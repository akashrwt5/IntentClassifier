"""
Entity extraction — the @entity layer of Dialogflow, reimplemented on-device.

Two kinds of entities:
  - enum entities  (@memory, @recurrence, @remind): value/synonym table from nlu_entities.json
  - system entities (@sys.date-time, @sys.number-integer): rule-based parsers
"""

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import dateparser  # type: ignore
    _HAS_DATEPARSER = True
except Exception:
    _HAS_DATEPARSER = False

BASE_DIR = Path(__file__).parent.parent.parent
ENTITIES_PATH = BASE_DIR / "data" / "nlu_entities.json"


def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca!=cb)))
        prev = cur
    return prev[-1]


class EntityExtractor:
    def __init__(self, entities_path: Path = ENTITIES_PATH):
        self.entities = json.loads(Path(entities_path).read_text(encoding="utf-8"))
        self._lookup = {}
        for name, cfg in self.entities.items():
            if cfg.get("type") == "enum":
                table = {}
                for value, synonyms in cfg["values"].items():
                    table[value.lower()] = value
                    for syn in synonyms:
                        table[syn.lower()] = value
                self._lookup[name] = table

    # Fuzzy enum matching only considers synonyms at least this long. Short
    # memory names (Car, Gym, Pub, Mute, one…) collide with common ASR words
    # at edit-distance 1 (Car↔care/cab, Pub↔pup, Gym↔gum), so fuzzy matching
    # them silently selects the wrong memory. Exact/word-boundary matches still
    # work for short names; only the risky fuzzy path is length-gated.
    _FUZZY_MIN_LEN = 5

    def extract_enum(self, entity: str, text: str, fuzzy: bool = True):
        """Return (value, span, confidence) or (None, None, 0.0).

        Confidence reflects match quality:
          1.00 — exact / canonical match
          0.95 — synonym / substring match
          0.60–0.90 — fuzzy match (scales with edit distance ratio)

        fuzzy=False disables the approximate path entirely — used for one-shot
        full-sentence scanning, where a stray fuzzy hit on a common word is a
        wrong-action risk. Fuzzy is reserved for explicit slot-prompt answers.
        """
        table = self._lookup.get(entity, {})
        t = text.lower()
        for syn in sorted(table, key=len, reverse=True):
            if re.search(rf"\b{re.escape(syn)}\b", t):
                conf = 1.0 if syn == table[syn].lower() else 0.95
                return table[syn], syn, conf
        if fuzzy and self.entities.get(entity, {}).get("fuzzy"):
            tokens = re.findall(r"[a-z0-9]+", t)
            best, best_span, best_d, best_len = None, None, 99, 1
            for syn, canon in table.items():
                if " " in syn or len(syn) < self._FUZZY_MIN_LEN:
                    continue
                # The _FUZZY_MIN_LEN gate already excludes the short names that
                # collide with common words; keep the 0.3 ratio so genuine
                # typos on longer names ("restraunt"→"restaurant") still match.
                limit = max(1, round(len(syn) * 0.3))
                for tok in tokens:
                    if abs(len(tok) - len(syn)) > limit:
                        continue
                    d = _levenshtein(tok, syn)
                    if d <= limit and d < best_d:
                        best, best_span, best_d, best_len = canon, tok, d, len(syn)
            if best:
                # confidence scales from 0.90 (1 edit) down to 0.60 (at limit)
                fuzzy_conf = round(1.0 - (best_d / best_len), 2)
                fuzzy_conf = max(0.60, min(0.90, fuzzy_conf))
                return best, best_span, fuzzy_conf
        return None, None, 0.0

    _NUM_WORDS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60,
    }

    def extract_number(self, text: str):
        """Return (value, span, confidence) or (None, None, 0.0)."""
        t = text.lower()
        m = re.search(r"\b\d+\b", t)
        if m: return int(m.group()), m.group(), 1.0
        for word, val in self._NUM_WORDS.items():
            if re.search(rf"\b{word}\b", t):
                return val, word, 1.0
        return None, None, 0.0

    _WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday",
                 "friday", "saturday", "sunday"]

    # Word → digit for hours/minutes (ASR often outputs word numbers)
    _WORD_NUMS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50,
    }

    def _normalise_word_numbers(self, t: str) -> str:
        """Replace word numbers with digits so downstream regexes work uniformly."""
        for word, val in sorted(self._WORD_NUMS.items(), key=lambda x: -len(x[0])):
            t = re.sub(rf"\b{word}\b", str(val), t)
        return t

    def _pick_future_hour(self, h: int, minute: int, base_day: datetime,
                          now: datetime, period: str = None) -> datetime:
        """
        Given an hour (1-12) with optional period hint, return the next future
        datetime on base_day (or base_day+1 if needed).

        period: 'am' | 'pm' | 'morning' | 'afternoon' | 'evening' | 'night' | None
        """
        if period in ("am", "morning"):
            h24 = h % 12          # 12am → 0
        elif period in ("pm", "afternoon", "evening", "night"):
            h24 = h % 12 + 12     # 12pm → 12
        else:
            # No explicit period — prefer next future occurrence.
            # Hours 1-6 almost always mean PM (nobody sets 3am reminders).
            # Hours 7-11 and 12: try AM first then PM.
            if 1 <= h <= 6:
                h24 = h + 12
            else:
                h24 = h  # try AM (or 12 noon for h=12)

        dt = base_day.replace(hour=h24, minute=minute, second=0, microsecond=0)

        # If the chosen time is in the past and we're on today, try the other period.
        if dt <= now and base_day.date() == now.date() and period is None:
            alt = h24 + 12 if h24 < 12 else h24 - 12
            if 0 <= alt <= 23:
                dt_alt = base_day.replace(hour=alt, minute=minute, second=0, microsecond=0)
                if dt_alt > now:
                    return dt_alt

        # Still in the past → push to next day.
        if dt <= now:
            dt += timedelta(days=1)

        return dt

    @staticmethod
    def _to_utc_iso(dt: datetime) -> str:
        """
        Serialize a datetime as a UTC ISO 8601 string (e.g.
        '2026-06-14T14:00:00+00:00'). Reminders sync to a cloud backend, so a
        single unambiguous instant is stored and each client renders it in its
        own local zone. A naive datetime is assumed to be device-local.
        """
        if dt.tzinfo is None:
            dt = dt.astimezone()  # attach local tz
        return dt.astimezone(timezone.utc).isoformat(timespec="minutes")

    def extract_datetime(self, text: str, now: datetime = None):
        """Return (iso, span, confidence, time_explicit).

        time_explicit is True when the user actually specified a time-of-day
        (clock time, relative duration, or a named period like "morning").
        It is False when only a day was given ("tomorrow") and the time had to
        be defaulted — the engine uses this to prompt for the missing time while
        keeping the resolved day.
        """
        # Parse the user's spoken time in their LOCAL zone, then store UTC.
        # now must be timezone-aware; a naive now is treated as device-local.
        if now is None:
            now = datetime.now().astimezone()
        elif now.tzinfo is None:
            now = now.astimezone()
        t = text.lower().strip()

        # --- 1. Relative durations: "in 10 minutes", "in an hour", "in a few minutes" ---
        # Digit form
        m = re.search(r"\bin\s+(\d+)\s*(minute|min|hour|hr|day|week)s?\b", t)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            delta = {"minute": timedelta(minutes=n), "min": timedelta(minutes=n),
                     "hour":   timedelta(hours=n),   "hr":  timedelta(hours=n),
                     "day":    timedelta(days=n),     "week": timedelta(weeks=n)}[unit]
            return self._to_utc_iso(now + delta), m.group(), 1.0, True
        # "in an hour / in a minute"
        m = re.search(r"\bin\s+an?\s+(minute|min|hour|hr|day|week)s?\b", t)
        if m:
            unit = m.group(1)
            delta = {"minute": timedelta(minutes=1), "min": timedelta(minutes=1),
                     "hour":   timedelta(hours=1),   "hr":  timedelta(hours=1),
                     "day":    timedelta(days=1),     "week": timedelta(weeks=1)}[unit]
            return self._to_utc_iso(now + delta), m.group(), 1.0, True
        # "in a few / a couple of minutes/hours"
        m = re.search(r"\bin\s+(?:a\s+few|a\s+couple\s+(?:of\s+)?)\s*(minute|min|hour|hr)s?\b", t)
        if m:
            unit = m.group(1)
            n = 3 if "few" in m.group() else 2
            delta = {"minute": timedelta(minutes=n), "min": timedelta(minutes=n),
                     "hour":   timedelta(hours=n),   "hr":  timedelta(hours=n)}[unit]
            return self._to_utc_iso(now + delta), m.group(), 1.0, True
        # "in half an hour"
        if re.search(r"\bin\s+half\s+an?\s+hour\b", t):
            return self._to_utc_iso(now + timedelta(minutes=30)), "in half an hour", 1.0, True

        # --- 2. Explicit past-date rejection ---
        if re.search(r"\byesterday\b", t):
            return None, None, 0.0, False

        # --- 3. Normalise word numbers so "nine pm" → "9 pm", "nine thirty" → "9 30" ---
        t = self._normalise_word_numbers(t)

        # --- 4. Day anchor (day-after-tomorrow must be checked before tomorrow) ---
        base_day = now
        explicit_day = False
        if re.search(r"\bday\s+after\s+tomorrow\b", t):
            base_day = now + timedelta(days=2); explicit_day = True
        elif re.search(r"\btomorrow\b", t):
            base_day = now + timedelta(days=1); explicit_day = True
        elif re.search(r"\btoday\b", t) or re.search(r"\btonight\b", t):
            base_day = now; explicit_day = True
        elif re.search(r"\bnext\s+week\b", t):
            base_day = now + timedelta(weeks=1); explicit_day = True
        else:
            for i, wd in enumerate(self._WEEKDAYS):
                if re.search(rf"\b{wd}\b", t):
                    ahead = (i - now.weekday()) % 7 or 7
                    base_day = now + timedelta(days=ahead)
                    explicit_day = True
                    break

        # --- 4. Period hint from context words ---
        period = None
        if re.search(r"\bmorning\b", t):       period = "morning"
        elif re.search(r"\bafternoon\b", t):   period = "afternoon"
        elif re.search(r"\bevening\b", t):     period = "evening"
        elif re.search(r"\bnight\b", t):       period = "night"
        elif re.search(r"\btonight\b", t):     period = "evening"
        elif re.search(r"\bnoon\b", t):        period = "noon"
        elif re.search(r"\bmidnight\b", t):    period = "midnight"

        # --- 5. Named-only time (no digit) ---
        named_hour = {
            "morning": 8, "afternoon": 14, "evening": 18,
            "night": 21, "tonight": 21, "noon": 12, "midnight": 0,
        }

        # --- 6. Explicit time extraction ---
        hour = minute = None; span = None

        # "half past N" / "N thirty" → e.g. half past 9 = 9:30
        m = re.search(r"\bhalf\s+past\s+(\d{1,2})\b", t)
        if m:
            hour, minute = int(m.group(1)), 30; span = m.group()
        # "quarter past N"
        if hour is None:
            m = re.search(r"\bquarter\s+past\s+(\d{1,2})\b", t)
            if m:
                hour, minute = int(m.group(1)), 15; span = m.group()
        # "quarter to N"  (N must be a valid clock hour)
        if hour is None:
            m = re.search(r"\bquarter\s+to\s+(\d{1,2})\b", t)
            if m:
                h = int(m.group(1))
                if 1 <= h <= 12:
                    hour, minute = (h - 1) if h > 1 else 12, 45; span = m.group()
        # "N past M" (e.g. "20 past 9") — minutes 1-59, hour a valid clock hour
        if hour is None:
            m = re.search(r"\b(\d{1,2})\s+past\s+(\d{1,2})\b", t)
            if m:
                mm, hh = int(m.group(1)), int(m.group(2))
                if 0 <= mm <= 59 and 1 <= hh <= 12:
                    minute, hour = mm, hh; span = m.group()
        # "N to M" (e.g. "10 to 3" = 2:50) — minutes-to 1-59, hour a valid clock hour
        if hour is None:
            m = re.search(r"\b(\d{1,2})\s+to\s+(\d{1,2})\b", t)
            if m:
                mins_to, h = int(m.group(1)), int(m.group(2))
                if 1 <= mins_to <= 59 and 1 <= h <= 12:
                    hour = (h - 1) if h > 1 else 12
                    minute = 60 - mins_to; span = m.group()

        # Standard "9am", "9 am", "9:30pm"
        explicit_ampm = False
        if hour is None:
            tm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b", t)
            if tm:
                raw_h = int(tm.group(1))
                period_char = tm.group(3)
                hour = raw_h % 12 + (12 if period_char == "p" else 0)
                minute = int(tm.group(2) or 0)
                span = tm.group()
                explicit_ampm = True  # hour is already 0-23; skip _pick_future_hour

        # "9:30" with colon, no am/pm
        if hour is None:
            tm = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
            if tm:
                hour, minute = int(tm.group(1)), int(tm.group(2)); span = tm.group()

        # "at N" or bare number as entire input — no am/pm
        if hour is None:
            # "at N M" — hour and space-separated minutes e.g. "at 9 30"
            m = re.search(r"\bat\s+(\d{1,2})\s+(\d{2})\b", t)
            if m:
                hour, minute = int(m.group(1)), int(m.group(2)); span = m.group()
        if hour is None:
            # "at N" anywhere in text
            m = re.search(r"\bat\s+(\d{1,2})\b", t)
            if m is None:
                # Bare number is the whole input (slot answer like "9" or "11")
                m = re.match(r"^(\d{1,2})\s*$", t)
            if m:
                hour, minute = int(m.group(1)), 0; span = m.group()

        # Space-separated H MM without colon — e.g. "9 30" (ASR / word-number output)
        if hour is None:
            m = re.search(r"\b(\d{1,2})\s+(\d{2})\b", t)
            if m and 0 <= int(m.group(2)) <= 59:
                hour, minute = int(m.group(1)), int(m.group(2)); span = m.group()

        # Digit paired with a period context word: "9 in the morning", "9 tonight"
        # This runs before the named-only fallback so the digit overrides the default hour.
        if hour is None and period is not None and period not in ("am", "pm"):
            m = re.search(r"\b(\d{1,2})\b", t)
            if m:
                h = int(m.group(1))
                if 1 <= h <= 12:
                    hour, minute = h, 0; span = m.group()

        # Named time only (no digit found at all)
        if hour is None and period in named_hour:
            hour, minute = named_hour[period], 0; span = period

        # --- 7. Build datetime ---
        if hour is not None:
            minute = minute or 0
            # Final range guard: malformed input (e.g. ASR "0 to 3" → minute 60,
            # "quarter to 13" → hour 13) must yield a clean no-match, never a
            # ValueError from datetime.replace().
            if not (0 <= minute <= 59):
                return None, None, 0.0, False
            try:
                if 1 <= hour <= 12 and not explicit_ampm and period not in ("am",):
                    # Need disambiguation — use period hint
                    p = period if period in ("am","pm","morning","afternoon","evening","night") else period
                    dt = self._pick_future_hour(hour, minute, base_day, now, p)
                else:
                    # hour is already 0-23 (from am/pm regex or colon format)
                    if not (0 <= hour <= 23):
                        return None, None, 0.0, False
                    dt = base_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if not explicit_day and dt <= now:
                        dt += timedelta(days=1)
            except ValueError:
                return None, None, 0.0, False
            # Reject explicitly past dates (e.g. "yesterday", already-past specific date)
            if explicit_day and base_day.date() < now.date():
                return None, None, 0.0, False
            return self._to_utc_iso(dt), span, 1.0, True

        if explicit_day:
            # Day anchor found but NO time. Return the resolved day (defaulted to
            # 9am for backward-compatible callers) with time_explicit=False so the
            # engine can prompt for the missing time while keeping this day.
            dt = base_day.replace(hour=9, minute=0, second=0, microsecond=0)
            return self._to_utc_iso(dt), span, 1.0, False

        # --- 8. Dateparser fallback (stripped to avoid month/day misparse) ---
        if _HAS_DATEPARSER:
            # Only pass to dateparser if text looks like a time/date expression,
            # not a bare word that would be misread as a month or day number.
            if not re.match(r"^\d{1,2}$", t):  # bare number already handled above
                dt = dateparser.parse(
                    t,
                    settings={
                        "PREFER_DATES_FROM": "future",
                        "PARSERS": ["absolute-time", "relative-time"],
                    },
                )
                if dt:
                    return self._to_utc_iso(dt), t, 0.85, True

        return None, None, 0.0, False

    def is_open(self, entity: str) -> bool:
        return bool(self.entities.get(entity, {}).get("open"))

    _TIME_PATTERNS = [
        r"\bin\s+\d+\s*(?:minute|min|hour|hr|day|week)s?\b",
        r"\b\d{1,2}(?::\d{2})?\s*[ap]\.?\s*m\.?\b",
        r"\b\d{1,2}:\d{2}\b",
        # "at 5" / "by 7" — remove the connector AND the orphaned bare number
        # together, before the bare-connector strip below leaves the digit behind.
        r"\b(?:at|by)\s+\d{1,2}(?::\d{2})?\b",
        r"\b(?:tomorrow|today|tonight)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b",
        r"\b(?:every|each)\s+\w+\b",
        # consume "in the morning" as a phrase so no "in the" fragment dangles
        r"\b(?:in\s+the\s+)?(?:morning|afternoon|evening|night|noon)\b",
        r"\b(?:at|on|by|this|next)\b",
    ]

    def strip_datetime(self, text: str) -> str:
        t = text
        for p in self._TIME_PATTERNS:
            t = re.sub(p, " ", t, flags=re.I)
        return re.sub(r"\s+", " ", t).strip(" .,")

    def extract(self, entity: str, text: str, fuzzy: bool = True):
        """Return (value, span, confidence). confidence=0.0 means no match.

        fuzzy is forwarded to enum matching; pass fuzzy=False for one-shot
        full-sentence scans to avoid stray approximate matches on common words.
        """
        if entity == "sys.date-time":
            iso, span, conf, _time_explicit = self.extract_datetime(text)
            return iso, span, conf
        if entity == "sys.number-integer":  return self.extract_number(text)
        return self.extract_enum(entity, text, fuzzy=fuzzy)
