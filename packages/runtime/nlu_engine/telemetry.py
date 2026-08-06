"""Telemetry event generation + on-device aggregation (Phase 3; ADR-003/005).

Responsibility split per runtime-contract-v1 / ADR-001 Part 5: EVENT
GENERATION (schema, redaction guarantees) lives here in the shared runtime —
one audited place that can never emit raw text; TRANSPORT (batching, upload,
consent gating) is native and out of scope.

Every event is keyed on `bundle_id` (the one-word answer to "what exactly
was the assistant?") and uses ONLY the closed enums from the bundle's
telemetry/schema.json. Aggregation matches the privacy stance: per-day
counters over (stage, result type, intent domain, outcome) — no utterances,
no per-user rows, no precise timestamps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

TELEMETRY_EVENT_VERSION = 1

# Closed enums (must stay subsets of telemetry/schema.json in the bundle;
# the compiler's stage-8 check owns that consistency once events ship).
STAGES = ("keyword", "tfidf", "semantic", "confirm", "uncertain_confirm",
          "slot_fill", "back_reference", "interrupt", "slot_abandon", "genai")
RESULT_TYPES = ("FULFILL", "PROMPT", "CONFIRM", "FALLBACK")
SECURITY_EVENTS = ("signature_invalid", "checksum_mismatch")


@dataclass
class TurnEvent:
    """One classified turn. NO raw text field exists — by construction."""

    bundle_id: str
    stage: str
    result_type: str
    intent_domain: str          # domain only ('device'), never the full intent
    confidence_bucket: str      # coarse '0.7-0.8'
    semantic_rescue: bool = False
    guarded: bool = False       # a polarity guard redirected the prediction
    v: int = TELEMETRY_EVENT_VERSION

    def __post_init__(self):
        assert self.stage in STAGES, self.stage
        assert self.result_type in RESULT_TYPES, self.result_type


def confidence_bucket(conf: float, width: float = 0.1) -> str:
    c = min(max(conf, 0.0), 0.9999)
    lo = int(c / width) * width
    return f"{lo:.1f}-{lo + width:.1f}"


_DOMAIN_SEPARATORS = re.compile(r"[._ ]")


def domain_of(intent: str | None) -> str:
    """The COARSE family of an intent — never the intent itself.

    Telemetry records a domain rather than a label so an uploaded event cannot
    say what a user asked about. That matters most for the help topics: tinnitus,
    fall alerts and heart rate are health-adjacent, and this product's users are
    a clinical population.

    It used to split on `.` alone and return the whole string when there was no
    dot. Under `domain.object.action` every label had a dot, so that was safe.
    The move to `Cmd.*` / `Help_*` broke it silently in the worst direction:
    `Help_Tinnitus` has no dot, so the guard returned `help_tinnitus` — the
    exact disclosure it exists to prevent — while `Cmd.VolumeMute` kept working.
    A privacy control that degrades quietly is worse than none, because nothing
    reports that it stopped.

    Splitting on any of `.`, `_` or a space keeps the first segment whatever
    convention the labels follow next.
    """
    if not intent:
        return "none"
    return _DOMAIN_SEPARATORS.split(intent, 1)[0].lower()


@dataclass
class TelemetryAggregator:
    """On-device aggregation: bounded, counters-only, flushable.

    The native TelemetryAgent drains :meth:`snapshot` on its own schedule
    and uploads under user consent; this class never does I/O unless asked
    to persist locally."""

    bundle_id: str
    counters: dict = field(default_factory=dict)
    day: str = field(default_factory=lambda: date.today().isoformat())

    def record(self, event: TurnEvent) -> None:
        key = "|".join([self.day, event.stage, event.result_type,
                        event.intent_domain, event.confidence_bucket,
                        "rescued" if event.semantic_rescue else "-",
                        "guarded" if event.guarded else "-"])
        self.counters[key] = self.counters.get(key, 0) + 1

    def record_security(self, kind: str) -> None:
        assert kind in SECURITY_EVENTS, kind
        key = f"{self.day}|security|{kind}"
        self.counters[key] = self.counters.get(key, 0) + 1

    def snapshot(self) -> dict:
        """The uploadable unit: bundle-keyed counters, nothing else."""
        return {"v": TELEMETRY_EVENT_VERSION, "bundle_id": self.bundle_id,
                "counters": dict(self.counters)}

    def flush_to(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2) + "\n")
        self.counters.clear()
