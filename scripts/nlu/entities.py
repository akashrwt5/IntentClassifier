"""
Entity extraction — the @entity layer of Dialogflow, reimplemented on-device.

Two kinds of entities:
  - enum entities  (@memory, @recurrence, @remind): match against a value/synonym
    table loaded from data/nlu_entities.json. Supports exact, substring and
    optional fuzzy matching.
  - system entities (@sys.date-time, @sys.number-integer): parsed with portable
    rule-based parsers. If the `dateparser` package is installed it is used to
    widen coverage, but the regex parser is the reference implementation the
    mobile apps should port.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

# Optional, heavier dependency. The regex parser below is the portable fallback.
try:
    import dateparser  # type: ignore
    _HAS_DATEPARSER = True
except Exception:
    _HAS_DATEPARSER = False

BASE_DIR = Path(__file__).parent.parent.parent
ENTITIES_PATH = BASE_DIR / "data" / "nlu_entities.json"


def _levenshtein(a: str, b: str) -> int:
    """Small pure-Python edit distance for fuzzy enum matching."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = cur
    return prev[-1]


class EntityExtractor:
    def __init__(self, entities_path: Path = ENTITIES_PATH):
        self.entities = json.loads(Path(entities_path).read_text(encoding="utf-8"))
        # Pre-build synonym → canonical lookups for each enum entity
        self._lookup = {}
        for name, cfg in self.entities.items():
            if cfg.get("type") == "enum":
                table = {}
                for value, synonyms in cfg["values"].items():
                    table[value.lower()] = value
                    for syn in synonyms:
                        table[syn.lower()] = value
                self._lookup[name] = table

    # ---------------- enum entities ----------------
    def extract_enum(self, entity: str, text: str):
        """Return (canonical_value, matched_span) or (None, None)."""
        table = self._lookup.get(entity, {})
        t = text.lower()

        # 1. exact / substring match against any synonym (longest first)
        for syn in sorted(table, key=len, reverse=True):
            if re.search(rf"\b{re.escape(syn)}\b", t):
                return table[syn], syn

        # 2. fuzzy match per-token if entity allows it
        if self.entities.get(entity, {}).get("fuzzy"):
            tokens = re.findall(r"[a-z0-9]+", t)
            best, best_d = None, 99
            for syn, canon in table.items():
                if " " in syn or len(syn) < 3:
                    continue  # only fuzzy-match single, non-trivial tokens
                # allow ~30% of the word to differ (ASR / spelling errors),
                # at least 1 edit for short words
                limit = max(1, round(len(syn) * 0.3))
                for tok in tokens:
                    if abs(len(tok) - len(syn)) > limit:
                        continue
                    d = _levenshtein(tok, syn)
                    if d <= limit and d < best_d:
                        best, best_d = canon, d
            if best:
                return best, best

        return None, None

    # ---------------- system: number ----------------
    _NUM_WORDS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60,
    }

    def extract_number(self, text: str):
        t = text.lower()
        m = re.search(r"\b\d+\b", t)
        if m:
            return int(m.group()), m.group()
        for word, val in self._NUM_WORDS.items():
            if re.search(rf"\b{word}\b", t):
                return val, word
        return None, None

    # ---------------- system: date-time ----------------
    _WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday"]

    def extract_datetime(self, text: str, now: datetime = None):
        """
        Return (iso_string, matched_span) or (None, None).
        Reference rule-based parser; covers the common hearing-aid reminder
        phrasings ("at 5 pm", "tomorrow morning", "in 10 minutes", "8 a.m.").
        """
        now = now or datetime.now()
        t = text.lower().strip()

        # ---- relative: "in N minutes/hours/days" ----
        m = re.search(r"\bin\s+(\d+)\s*(minute|min|hour|hr|day|week)s?\b", t)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            delta = {
                "minute": timedelta(minutes=n), "min": timedelta(minutes=n),
                "hour": timedelta(hours=n), "hr": timedelta(hours=n),
                "day": timedelta(days=n), "week": timedelta(weeks=n),
            }[unit]
            return (now + delta).isoformat(timespec="minutes"), m.group()

        base_day = now
        span = None

        # ---- day anchors ----
        if re.search(r"\btomorrow\b", t):
            base_day = now + timedelta(days=1); span = "tomorrow"
        elif re.search(r"\btoday\b", t) or re.search(r"\btonight\b", t):
            base_day = now; span = "today"
        else:
            for i, wd in enumerate(self._WEEKDAYS):
                if re.search(rf"\b{wd}\b", t):
                    ahead = (i - now.weekday()) % 7
                    ahead = ahead or 7  # next occurrence
                    base_day = now + timedelta(days=ahead); span = wd
                    break

        # ---- clock time: "5 pm", "5:30 pm", "8 a.m.", "17:00" ----
        hour = minute = None
        tm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b", t)
        if tm:
            hour = int(tm.group(1)) % 12
            if tm.group(3) == "p":
                hour += 12
            minute = int(tm.group(2) or 0)
            span = tm.group()
        else:
            tm = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
            if tm:
                hour, minute = int(tm.group(1)), int(tm.group(2))
                span = tm.group()

        # ---- fuzzy day-parts ----
        if hour is None:
            if "morning" in t:   hour, minute, span = 8, 0, span or "morning"
            elif "afternoon" in t: hour, minute, span = 14, 0, span or "afternoon"
            elif "evening" in t: hour, minute, span = 18, 0, span or "evening"
            elif "night" in t:   hour, minute, span = 21, 0, span or "night"
            elif "noon" in t:    hour, minute, span = 12, 0, span or "noon"

        if hour is not None or span:
            dt = base_day.replace(
                hour=hour if hour is not None else 9,
                minute=minute or 0, second=0, microsecond=0,
            )
            # if only a time was given and it already passed today, roll forward
            if span and span not in ("tomorrow",) and dt < now and "tomorrow" not in t:
                if hour is not None and base_day.date() == now.date():
                    dt = dt + timedelta(days=1)
            return dt.isoformat(timespec="minutes"), span

        # ---- last resort: dateparser if available ----
        if _HAS_DATEPARSER:
            dt = dateparser.parse(t, settings={"PREFER_DATES_FROM": "future"})
            if dt:
                return dt.isoformat(timespec="minutes"), t

        return None, None

    # ---------------- open-ended (free-form) topic support ----------------
    def is_open(self, entity: str) -> bool:
        """True if the enum entity accepts free-form values (Dialogflow @sys.any)."""
        return bool(self.entities.get(entity, {}).get("open"))

    # time / recurrence expressions to peel off when deriving a free-form topic
    _TIME_PATTERNS = [
        r"\bin\s+\d+\s*(?:minute|min|hour|hr|day|week)s?\b",
        r"\b\d{1,2}(?::\d{2})?\s*[ap]\.?\s*m\.?\b",   # 5 pm, 8 a.m., 5:30 pm
        r"\b\d{1,2}:\d{2}\b",                          # 17:00
        r"\b(?:tomorrow|today|tonight)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b",
        r"\b(?:every|each)\s+\w+\b",                   # every morning, each day
        r"\b(?:morning|afternoon|evening|night|noon)\b",
        r"\b(?:at|on|by|this|next)\b",
    ]

    def strip_datetime(self, text: str) -> str:
        """Remove date/time/recurrence expressions, leaving the bare topic."""
        t = text
        for p in self._TIME_PATTERNS:
            t = re.sub(p, " ", t, flags=re.I)
        return re.sub(r"\s+", " ", t).strip(" .,")

    # ---------------- dispatch ----------------
    def extract(self, entity: str, text: str):
        """Extract a single entity of the given type from text."""
        if entity == "sys.date-time":
            return self.extract_datetime(text)
        if entity == "sys.number-integer":
            return self.extract_number(text)
        return self.extract_enum(entity, text)
