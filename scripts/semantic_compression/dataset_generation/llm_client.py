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

import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel

from seed_loader import GeneratorConfig, SeedCorpusError

SUPPORTED_PROVIDERS = ("openai", "azure", "anthropic", "google")

#: Azure serves OpenAI's models from a different endpoint, with a different key
#: and a per-resource DEPLOYMENT name. That deployment name is chosen by whoever
#: provisioned the resource and need not resemble the model at all, so
#: ``llm.model`` stays the REAL model name -- it is what
#: ``supports_temperature`` has to reason about -- and the deployment travels
#: separately in ``llm.azure.deployment``.
#:
#: Getting those two backwards breaks the temperature decision silently: a
#: deployment called "prod-nlu" matches no known prefix, so the code would
#: cheerfully send a parameter the underlying model rejects.

#: Model-name prefixes that REJECT the ``temperature`` parameter outright,
#: returning "Unsupported parameter: 'temperature' is not supported with this
#: model." rather than ignoring it. OpenAI's reasoning families removed manual
#: sampling control in favour of internal self-adjustment, so passing the
#: parameter is a hard 400 and would fail every call in the run.
#:
#: This matters more here than in most pipelines: temperature is the primary
#: lever for the output variability this project exists to maximise, so a model
#: that removes it is a real design trade-off, not just a config detail.
NO_TEMPERATURE_PREFIXES: tuple[str, ...] = ("gpt-5", "o1", "o3", "o4")

#: Families that DO accept temperature despite matching a prefix above.
#: ``gpt-5-chat`` is the chat-tuned variant and takes it unconditionally.
#: ``gpt-5.1`` and ``gpt-5.2`` take it only while ``reasoning_effort`` is
#: ``none``.
#:
#: WARNING -- this list describes what the MODEL accepts, NOT what this stack
#: currently delivers. An earlier version of this comment claimed
#: ``reasoning_effort`` defaults to ``none`` when the parameter is omitted, "as
#: it is here". That is false. Measured against langchain-openai 1.5.1, the
#: client NULLS temperature for the whole gpt-5.x family unless
#: ``reasoning_effort="none"`` is passed explicitly, and it does so silently --
#: no error, no warning:
#:
#:     model              temperature passed in      temperature on the client
#:     gpt-4o             0.9                        0.9
#:     gpt-5-chat-latest  0.9                        0.9
#:     gpt-5.1            0.9                        None
#:     gpt-5.1-mini       0.9                        None
#:     gpt-5              0.9                        None
#:     gpt-5.2            0.9                        None
#:
#:     gpt-5.1 + reasoning_effort="none"     0.9     0.9
#:     gpt-5.1 + reasoning_effort="minimal"  0.9     None
#:
#: So ``supports_temperature("gpt-5.1") is True`` does not mean 0.9 reaches the
#: API. Until ``build_structured_llm`` sends ``reasoning_effort="none"``, a run
#: on any gpt-5.x model other than ``gpt-5-chat`` has NO temperature control at
#: all -- and temperature is this project's primary lever for the output
#: variability it exists to produce (see ``generation.llm_overrides``). The
#: failure is silent, so it would surface as repetitive generated data long
#: after the run was paid for. Verify before choosing a gpt-5.x model.
#:
#: Separately, the boundary matching below is deliberate: a naive
#: ``"gpt-5" in model`` check also matches ``gpt-5.1`` and strips a parameter
#: that model does accept. That bug is open against LiteLLM
#: (BerriAI/litellm#17005), so the prefixes are anchored rather than substrings.
TEMPERATURE_EXCEPTIONS: tuple[str, ...] = ("gpt-5-chat", "gpt-5.1", "gpt-5.2")


def supports_temperature(model: str) -> bool:
    """Whether ``model`` accepts a ``temperature`` argument.

    Matching is on a version boundary, so ``gpt-5`` and ``gpt-5-mini`` are
    treated as reasoning models while ``gpt-5.1`` is not.
    """
    name = model.lower()
    if any(name.startswith(x) for x in TEMPERATURE_EXCEPTIONS):
        return True
    for prefix in NO_TEMPERATURE_PREFIXES:
        if name == prefix or name.startswith(prefix + "-"):
            return False
    return True


def temperature_kwargs(llm_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the temperature kwarg, omitting it where the model rejects it.

    Set ``llm.temperature: null`` in the config to omit it explicitly.
    """
    configured = llm_cfg.get("temperature")
    if configured is None:
        return {}
    if not supports_temperature(str(llm_cfg["model"])):
        return {}
    return {"temperature": float(configured)}


def reasoning_kwargs(llm_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the ``reasoning_effort`` kwarg, when the config asks for one.

    Config-driven rather than inferred from the model name, for two reasons.

    It is a real decision, not plumbing: ``reasoning_effort: none`` turns the
    model's reasoning OFF, so a gpt-5.x model runs as an ordinary chat model.
    That trade is worth making deliberately and worth seeing in the config,
    rather than discovering inside a code path.

    And it is the only way to keep temperature on gpt-5.x. langchain-openai
    nulls temperature for that family unless ``reasoning_effort="none"`` is sent
    explicitly -- silently, with no error (see TEMPERATURE_EXCEPTIONS above).
    Since temperature is this project's primary lever for output variability,
    omitting this on gpt-5.x means generating without that lever at all.

    Only OpenAI and Azure accept the parameter; other providers ignore it.
    """
    effort = llm_cfg.get("reasoning_effort")
    if effort is None:
        return {}
    return {"reasoning_effort": str(effort)}


def _assert_temperature_applied(llm: Any, llm_cfg: dict[str, Any], provider: str) -> None:
    """Fail if the SDK dropped a temperature the config asked for.

    This exists because the drop is SILENT. Passing ``temperature=0.9`` to a
    gpt-5.x client is accepted without complaint and then nulled, so a full run
    completes, is paid for, and produces the repetitive output this project
    exists to eliminate -- with nothing anywhere reporting a problem.

    Checking the constructed client rather than a table of model names keeps the
    guard independent of the SDK version: whatever the next release decides to
    rewrite, this still catches it, at construction time, before one paid call.

    ``temperature: null`` in the config means "do not send one", so nothing is
    requested and nothing is checked -- that remains the supported way to run a
    model with no temperature control.
    """
    requested = temperature_kwargs(llm_cfg).get("temperature")
    if requested is None:
        return
    # Default to `requested` so a client that simply does not expose the
    # attribute is not reported as a failure.
    landed = getattr(llm, "temperature", requested)
    if landed == requested:
        return
    raise SeedCorpusError(
        f"temperature {requested} was requested for model "
        f"{llm_cfg.get('model')!r} on provider {provider!r}, but the client "
        f"reports {landed!r} -- the SDK dropped it silently.\n"
        "  - On gpt-5.x this is expected unless `llm.reasoning_effort: none` "
        "is set; without it there is no temperature control at all.\n"
        "  - To run deliberately without temperature, set `llm.temperature: null`."
    )


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


def resolve_llm_config(config: GeneratorConfig, stage: str | None = None) -> dict[str, Any]:
    """Merge ``llm`` with any per-stage override block.

    Spec authoring and dataset generation want OPPOSITE settings from the same
    config file. Bootstrapping a specification is a precision task -- one right
    answer, low temperature. Generating 7,680 utterances is the reverse: the
    entire objective is variability, and a low temperature there produces the
    repetitive near-duplicate output this project exists to eliminate.

    So ``generation.llm_overrides`` layers on top of ``llm`` rather than either
    stage silently inheriting a value tuned for the other.
    """
    merged = dict(config.llm)
    if stage:
        overrides = (config.raw.get(stage) or {}).get("llm_overrides") or {}
        merged.update(overrides)
    return merged


def build_structured_llm(
    config: GeneratorConfig,
    schema: type[BaseModel],
    *,
    stage: str | None = None,
) -> Any:
    """Return a LangChain runnable that emits ``schema``.

    Imported lazily so ``--dry-run`` works with no provider SDK installed.
    """
    llm_cfg = resolve_llm_config(config, stage)
    provider = str(llm_cfg.get("provider", "openai")).lower()
    model = str(llm_cfg["model"])
    timeout = llm_cfg.get("request_timeout_seconds", 90)

    if provider not in SUPPORTED_PROVIDERS:
        raise SeedCorpusError(
            f"llm.provider must be one of {SUPPORTED_PROVIDERS}, got {provider!r}"
        )

    kwargs: dict[str, Any] = {"model": model, "timeout": timeout}
    kwargs.update(temperature_kwargs(llm_cfg))
    if provider in ("openai", "azure"):
        kwargs.update(reasoning_kwargs(llm_cfg))

    # Passed explicitly rather than left to the SDK's environment lookup, so a
    # key supplied through the secrets file works identically to one exported
    # into the shell.
    api_key = resolve_api_key(config)
    if api_key:
        kwargs["google_api_key" if provider == "google" else "api_key"] = api_key

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        llm: Any = ChatOpenAI(**kwargs)
    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI

        azure = llm_cfg.get("azure") or {}
        missing = [k for k in ("endpoint", "deployment", "api_version") if not azure.get(k)]
        if missing:
            raise SeedCorpusError(
                "llm.provider is 'azure' but llm.azure is missing: "
                f"{missing}. Endpoint, deployment and api_version are not secret "
                "and belong in the config; only AZURE_OPENAI_API_KEY comes from "
                "the environment."
            )
        # The whole pipeline runs through `with_structured_output`, which needs
        # an api_version new enough to expose structured outputs. An old one
        # fails on the first call rather than degrading quietly -- the better
        # failure, but cheaper to discover before a full run is launched.
        llm = AzureChatOpenAI(
            azure_endpoint=str(azure["endpoint"]),
            azure_deployment=str(azure["deployment"]),
            api_version=str(azure["api_version"]),
            **kwargs,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(**kwargs)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(**kwargs)

    _assert_temperature_applied(llm, llm_cfg, provider)
    return llm.with_structured_output(schema)


def required_api_key(config: GeneratorConfig) -> str:
    """Environment variable the configured provider needs."""
    return {
        "openai": "OPENAI_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }[str(config.llm.get("provider", "openai")).lower()]


def _inside_a_git_repository(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in path.parents)


def load_secrets(config: GeneratorConfig) -> dict[str, str]:
    """Read the local secrets file, keyed by environment-variable name.

    The file is keyed by the same names the environment uses
    (``AZURE_OPENAI_API_KEY`` and friends) so there is no mapping to get wrong
    and no ambiguity about which key belongs to which provider.

    ``paths.secrets_file`` defaults to a location OUTSIDE the repository. That
    is the point: this repository is public, and a file that never enters the
    working tree cannot be committed by a stray ``git add -f``, cannot ride
    along in a ``git stash``, and cannot appear in someone else's clone.
    ``.gitignore`` is a second line of defence, not the first one.

    Missing file is not an error -- the environment variable remains a
    perfectly good way to supply the key, and is what CI would use.
    """
    raw_path = (config.raw.get("paths") or {}).get("secrets_file")
    if not raw_path:
        return {}

    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (config.config_path.parent / path).resolve()
    if not path.is_file():
        return {}

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        print(
            f"WARNING: {path} is readable by other users (mode {mode:04o}). "
            "Run: chmod 600 <file>"
        )
    if _inside_a_git_repository(path):
        print(
            f"WARNING: {path} sits inside a git repository, and this one is "
            "PUBLIC. Prefer a path outside the working tree entirely."
        )

    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SeedCorpusError(f"{path} did not parse to a mapping of NAME: value")
    return {str(k): str(v).strip() for k, v in data.items() if v and str(v).strip()}


def resolve_api_key(config: GeneratorConfig) -> str | None:
    """The provider's API key: environment first, then the secrets file.

    Environment wins so a one-off run, a different key, or a CI job can override
    the file without editing it.
    """
    name = required_api_key(config)
    return os.environ.get(name) or load_secrets(config).get(name)


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
