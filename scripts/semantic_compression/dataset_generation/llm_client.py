"""Provider-agnostic structured-output client with the Validation Failure Policy.

Two responsibilities, deliberately kept out of ``generator.py``:

1. **Provider choice.** ``generator_config.yaml`` has always carried an
   ``llm.provider`` field; this module is what finally honours it. OpenAI,
   Anthropic and Google all expose structured output through LangChain's
   ``with_structured_output``, so the rest of the pipeline does not need to know
   which one is in use — which matters, because this project's provider decision
   is still open and should stay a config edit rather than a code change.

2. **Reject, regenerate, retry** (Architecture Section 7). A failed batch is not
   discarded wholesale: the specific reasons are fed back to the model and only
   that batch is regenerated, up to ``llm.max_retries``. Every rejection is
   logged so the failures can be inspected offline rather than vanishing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from pydantic import BaseModel

from seed_loader import GeneratorConfig, SeedCorpusError

SUPPORTED_PROVIDERS = ("openai", "anthropic", "google")


@dataclass
class Rejection:
    """One failed attempt, kept for the offline rejection log."""

    intent: str
    attempt: int
    reasons: list[str]
    raw: str = ""


@dataclass
class GenerationOutcome:
    """Result of one validated request."""

    value: Any = None
    attempts: int = 0
    rejections: list[Rejection] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None


def build_structured_llm(config: GeneratorConfig, schema: type[BaseModel]) -> Any:
    """Return a LangChain runnable that emits ``schema``.

    Imported lazily so ``--dry-run`` works with no provider SDK installed.
    """
    llm_cfg = config.llm
    provider = str(llm_cfg.get("provider", "openai")).lower()
    model = llm_cfg["model"]
    temperature = float(llm_cfg.get("temperature", 0.7))
    timeout = llm_cfg.get("request_timeout_seconds", 90)

    if provider not in SUPPORTED_PROVIDERS:
        raise SeedCorpusError(
            f"llm.provider must be one of {SUPPORTED_PROVIDERS}, got {provider!r}"
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        llm: Any = ChatOpenAI(model=model, temperature=temperature, timeout=timeout)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, temperature=temperature, timeout=timeout)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, timeout=timeout)

    return llm.with_structured_output(schema)


def required_api_key(config: GeneratorConfig) -> str:
    """Environment variable the configured provider needs."""
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }[str(config.llm.get("provider", "openai")).lower()]


def invoke_with_validation(
    chain: Any,
    payload: dict[str, Any],
    *,
    validate: Callable[[Any], Sequence[str]],
    config: GeneratorConfig,
    label: str,
    correction_key: str = "correction",
) -> GenerationOutcome:
    """Call the model until ``validate`` returns no errors or retries run out.

    ``validate`` returns human-readable failure reasons. Those reasons are
    written back into the prompt on the next attempt, so the model is told
    exactly what to fix rather than simply being asked again — a plain retry at
    the same temperature tends to reproduce the same mistake.
    """
    max_retries = int(config.llm.get("max_retries", 3))
    backoff = float(config.llm.get("retry_backoff_seconds", 2.0))
    outcome = GenerationOutcome()
    current = dict(payload)
    current.setdefault(correction_key, "")

    for attempt in range(1, max_retries + 1):
        outcome.attempts = attempt
        try:
            result = chain.invoke(current)
        except Exception as exc:  # noqa: BLE001 -- any SDK/transport failure
            outcome.rejections.append(Rejection(label, attempt, [f"call failed: {exc}"]))
            if attempt < max_retries:
                time.sleep(backoff * attempt)
            continue

        errors = list(validate(result))
        if not errors:
            outcome.value = result
            return outcome

        outcome.rejections.append(Rejection(label, attempt, errors))
        current = dict(current)
        current[correction_key] = (
            "\nYour previous attempt was REJECTED. Fix every one of these and "
            "return a fresh batch:\n" + "\n".join(f"- {e}" for e in errors) + "\n"
        )
        if attempt < max_retries:
            time.sleep(backoff * attempt)

    return outcome
