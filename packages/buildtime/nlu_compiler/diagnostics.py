"""Diagnostics for the shared validator (ADR-005 Part 5).

Every failure names the file (and therefore the owning team via CODEOWNERS),
the stage that caught it, and a stable code so CI can assert on specific
failures without string-matching messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Diagnostic:
    stage: int          # compiler stage that caught it (1–10)
    code: str           # stable machine code, e.g. "SCHEMA_INVALID"
    path: str           # bundle-relative file (or "<bundle>")
    message: str
    severity: Severity = Severity.ERROR

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[stage {self.stage}] {self.severity.value.upper()} {self.code} {self.path}: {self.message}"


def errors(diags: list[Diagnostic]) -> list[Diagnostic]:
    return [d for d in diags if d.severity is Severity.ERROR]
