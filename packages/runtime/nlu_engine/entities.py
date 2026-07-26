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

BASE_DIR = Path(__file__).resolve().parents[3]
ENTITIES_PATH = BASE_DIR / "content" / "nlu_entities.json"
LOCALIZATION_DIR = BASE_DIR / "content" / "localization"

# Unicode-Latin-aware word boundaries for the lexicon-driven datetime parser.
# Must match EntityExtractor.swift exactly so Swift and Python agree on every
# golden fixture row. \b is avoided because its definition of "word" differs by
# engine; an explicit class keeps fr/de/da accents (é, ü, ø, å, æ) as letters.
_WB_L = r"(?<![0-9A-Za-zÀ-ÿ])"
_WB_R = r"(?![0-9A-Za-zÀ-ÿ])"
# English datetime grammar — the FALLBACK table used when a pack supplies none.
# `_DEFAULT_` is the language-neutrality guard's convention for an overridable
# DATA table (scripts/ci/check_language_neutral.py): the words live here as
# data, the interpretation lives in _build_en_dt_tables + extract_datetime, and
# a pack replaces the words without touching a line of engine code.
#
# Ported from the reference branch, which had already done this eviction. Only
# the vocabulary moved; the parser's branch ORDER and offsets are unchanged,
# and tests/test_datetime_parity_en.py (130 golden cases) proves it.
_DEFAULT_DT_GRAMMAR = {
    "day_anchors": {
        "day_after_tomorrow": ["day after tomorrow"],
        "tomorrow": ["tomorrow"],
        "today": ["today"],
        "tonight": ["tonight"],
        "next_week": ["next week"],
        "yesterday": ["yesterday"],
    },
    "time_of_day": {
        "morning":   {"names": ["morning"],   "hour": 8},
        "afternoon": {"names": ["afternoon"], "hour": 14},
        "evening":   {"names": ["evening"],   "hour": 18},
        "night":     {"names": ["night"],     "hour": 21},
        "noon":      {"names": ["noon"],      "hour": 12},
        "midnight":  {"names": ["midnight"],  "hour": 0},
    },
    "relative_markers": {"in": ["in"], "for": ["for"], "at": ["at"]},
    "relative_units": {
        "minute": ["minutes", "minute", "mins", "min"],
        "hour":   ["hours", "hour", "hrs", "hr"],
        "day":    ["days", "day"],
        "week":   ["weeks", "week"],
    },
    "articles": ["an", "a"],
    "quantifiers": {"few": {"phrases": ["a few"], "n": 3},
                    "couple": {"phrases": ["a couple of", "a couple"], "n": 2}},
    "clock_idioms": {
        "half_past": ["half past"], "quarter_past": ["quarter past"],
        "quarter_to": ["quarter to"], "past": ["past"], "to": ["to"],
        # Both articles: the original regex was `half\s+an?\s+hour`, so the
        # ungrammatical-but-common ASR output "half a hour" resolved too. The
        # reference branch's table listed only "half an hour" and silently
        # dropped it — caught by the golden corpus, not by review.
        "half_an_hour": ["half an hour", "half a hour"],
    },
    "am_pm": {"am": ["am", "a m", "a.m"], "pm": ["pm", "p m", "p.m"]},
    # Function words used only to strip date/time fragments out of a reminder
    # topic (cosmetic — never affects the resolved time).
    "strip": {
        "at_by": ["at", "by"],
        "connectors": ["at", "on", "by", "this", "next"],
        "recurrence": ["every", "each"],
        "the": ["the"],
    },
}

# Day-anchor offsets in days. Keys are canonical ROLE names, not words — a pack
# maps its own vocabulary onto these, so they are identifiers rather than
# match vocabulary (which is why the neutrality guard does not flag them).
_ANCHOR_OFFSET = {"today": 0, "tonight": 0, "tomorrow": 1,
                  "day_after_tomorrow": 2, "next_week": 7}


_WD_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"]


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
    def __init__(self, entities_path: Path = ENTITIES_PATH, *, weekdays=None, word_nums=None,
                 language: str = "en", lexicon_path: Path = None,
                 dt_grammar: dict = None):
        self.entities = json.loads(Path(entities_path).read_text(encoding="utf-8"))
        # English-path datetime vocabulary: pack table if supplied, else the
        # _DEFAULT_ fallback. Compiled once into matchers/alternations.
        self._build_en_dt_tables(dt_grammar or _DEFAULT_DT_GRAMMAR)
        self._lookup = {}
        for name, cfg in self.entities.items():
            if cfg.get("type") == "enum":
                table = {}
                for value, synonyms in cfg["values"].items():
                    table[value.lower()] = value
                    for syn in synonyms:
                        table[syn.lower()] = value
                self._lookup[name] = table
        if weekdays is not None:
            self._WEEKDAYS = weekdays
        if word_nums is not None:
            self._WORD_NUMS = word_nums

        # Lexicon-driven datetime parsing. Data-driven, NOT language-string
        # driven: if a datetime lexicon exists for this language it is used;
        # otherwise (English, which ships none, or a missing/undecodable file)
        # self._lex stays None and extract_datetime uses the table-driven path
        # fed by _DEFAULT_DT_GRAMMAR. Absence is the signal, so a new language
        # is a file rather than a branch.
        self._lex = None
        if lexicon_path is None and language:
            candidate = LOCALIZATION_DIR / f"nlu_lexicon.{language}.json"
            lexicon_path = candidate if candidate.exists() else None
        if lexicon_path is not None:
            try:
                self._lex = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
            except Exception:
                self._lex = None
        if self._lex is not None:
            self._build_lex_tables(self._lex)

    def _build_en_dt_tables(self, g: dict):
        """Compile the English day-anchor and period matchers from a grammar
        table (pack or default). The if/elif priority and offsets/hours of the
        original parser are preserved exactly; only the words come from data."""
        anchors = g.get("day_anchors", {})
        # Anchor match order preserves the original priority (day-after-tomorrow
        # before tomorrow so the shared substring never mis-fires).
        order = ["day_after_tomorrow", "tomorrow", "today", "tonight", "next_week"]
        self._anchor_specs = [
            (re.compile(rf"\b{re.escape(p)}\b"), key)
            for key in order for p in anchors.get(key, [])
        ]
        self._yesterday_res = [re.compile(rf"\b{re.escape(p)}\b")
                               for p in anchors.get("yesterday", [])]
        tod = g.get("time_of_day", {})
        # Period detection order matches the original: morning, afternoon,
        # evening, night, then tonight→evening, then noon, midnight.
        seq = []
        for name in ("morning", "afternoon", "evening", "night"):
            for p in tod.get(name, {}).get("names", []):
                seq.append((re.compile(rf"\b{re.escape(p)}\b"), name))
        for p in anchors.get("tonight", []):
            seq.append((re.compile(rf"\b{re.escape(p)}\b"), "evening"))
        for name in ("noon", "midnight"):
            for p in tod.get(name, {}).get("names", []):
                seq.append((re.compile(rf"\b{re.escape(p)}\b"), name))
        self._period_specs = seq
        self._named_hour = {name: spec["hour"] for name, spec in tod.items()}

        # ---- clock notation vocabulary (relative markers, units, idioms, am/pm) ----
        def _alt(words):
            # longest-first so multi-word phrases win over their prefixes
            return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))

        rel = g.get("relative_markers", {})
        self._alt_in = _alt(rel.get("in", []))
        self._alt_for = _alt(rel.get("for", []))
        self._alt_at = _alt(rel.get("at", []))
        self._alt_in_for = _alt(rel.get("in", []) + rel.get("for", []))

        units = g.get("relative_units", {})
        self._unit_canon = {w.lower(): canon for canon, arr in units.items() for w in arr}
        self._alt_unit = _alt(list(self._unit_canon))
        # "a few/couple of" originally applied to minute/hour only — preserve that.
        self._alt_unit_mh = _alt(units.get("minute", []) + units.get("hour", []))

        self._alt_article = _alt(g.get("articles", []))
        quant = g.get("quantifiers", {})
        self._quant_specs = [(_alt(spec.get("phrases", [])), spec.get("n", 1))
                             for spec in quant.values()]

        idi = g.get("clock_idioms", {})
        self._alt_half_past = _alt(idi.get("half_past", []))
        self._alt_quarter_past = _alt(idi.get("quarter_past", []))
        self._alt_quarter_to = _alt(idi.get("quarter_to", []))
        self._alt_past = _alt(idi.get("past", []))
        self._alt_to = _alt(idi.get("to", []))
        self._alt_half_an_hour = _alt(idi.get("half_an_hour", []))

        ap = g.get("am_pm", {})
        self._am_forms = {w.lower() for w in ap.get("am", [])}
        self._pm_forms = {w.lower() for w in ap.get("pm", [])}
        self._alt_ampm = _alt(list(self._am_forms) + list(self._pm_forms))

        # ---- topic-strip vocabulary (function words; cosmetic only) ----
        strip = g.get("strip", {})
        self._alt_strip_atby = _alt(strip.get("at_by", []))
        self._alt_strip_conn = _alt(strip.get("connectors", []))
        self._alt_strip_recur = _alt(strip.get("recurrence", []))
        self._alt_strip_the = _alt(strip.get("the", []))
        self._alt_strip_anchor = _alt(
            anchors.get("tomorrow", []) + anchors.get("today", []) + anchors.get("tonight", []))
        self._alt_strip_periods = _alt(
            [p for name, spec in tod.items() if name != "midnight" for p in spec.get("names", [])])
        self._strip_patterns_cache = None

    _UNIT_DELTA = {"minute": "minutes", "hour": "hours", "day": "days", "week": "weeks"}

    _UNIT_DELTA = {"minute": "minutes", "hour": "hours", "day": "days", "week": "weeks"}

    def _rel_delta(self, canon: str, n: int) -> timedelta:
        return timedelta(**{self._UNIT_DELTA[canon]: n})

    # ----- Lexicon-driven datetime: reverse-lookup tables (built once) -----

    def _en_strip_patterns(self):
        """Build the English topic-strip patterns from the grammar tables +
        weekdays (compiled once). Mirrors the previous hardcoded _TIME_PATTERNS
        order exactly — in particular '(at|by) N' is stripped before the bare
        connector so no orphan digit is left behind."""
        if self._strip_patterns_cache is not None:
            return self._strip_patterns_cache
        wd_alt = "|".join(re.escape(w) for w in self._WEEKDAYS)
        pats = [
            rf"\b(?:{self._alt_in_for})\s+\d+\s*(?:{self._alt_unit})\b",
            rf"\b\d{{1,2}}(?::\d{{2}})?\s*(?:{self._alt_ampm})\b",
            r"\b\d{1,2}:\d{2}\b",
            rf"\b(?:{self._alt_strip_atby})\s+\d{{1,2}}(?::\d{{2}})?\b",
            rf"\b(?:{self._alt_strip_anchor})\b",
            rf"\b(?:{wd_alt})s?\b",
            rf"\b(?:{self._alt_strip_recur})\s+\w+\b",
            rf"\b(?:{self._alt_in}\s+{self._alt_strip_the}\s+)?(?:{self._alt_strip_periods})\b",
            rf"\b(?:{self._alt_strip_conn})\b",
        ]
        self._strip_patterns_cache = [re.compile(p, re.I) for p in pats]
        return self._strip_patterns_cache

    # ----- Lexicon-driven datetime: reverse-lookup tables (built once) -----

    def _build_lex_tables(self, d):
        g = d.get("grammar", {})
        self._lex_is24h = g.get("time_format") == "24h"
        self._lex_conj = (g.get("conjunction") or "").lower() or None
        # (phrase, minutes, hour) sorted longest-phrase-first to avoid partial hits
        self._lex_idioms = sorted(
            [(i["phrase"].lower(), i.get("minutes"), i.get("hour"))
             for i in g.get("decimal_hour_idioms", []) if isinstance(i, dict) and "phrase" in i],
            key=lambda x: (-len(x[0]), x[0]))
        self._lex_weekday = {}                         # synonym -> 0..6 (Mon..Sun)
        for idx, name in enumerate(_WD_ORDER):
            for syn in d.get("weekdays", {}).get(name, []):
                self._lex_weekday[syn.lower()] = idx
        self._lex_month = {}                           # synonym -> 1..12
        for mi, name in enumerate(_MONTH_ORDER, 1):
            for syn in d.get("months", {}).get(name, []):
                self._lex_month[syn.lower()] = mi
        self._lex_number = {}                          # synonym -> int
        for k, syns in d.get("numbers_0_to_31", {}).items():
            for syn in syns:
                self._lex_number[syn.lower()] = int(k)
        for k, syns in d.get("ordinals_1_to_31", {}).items():
            for syn in syns:
                self._lex_number[syn.lower()] = int(k)
        self._lex_number_phrases = sorted(self._lex_number.items(), key=lambda x: (-len(x[0]), x[0]))
        self._lex_period = sorted(                     # (name, hour) longest-first
            [(n.lower(), e["hour"]) for e in d.get("time_of_day", {}).values() for n in e["names"]],
            key=lambda x: (-len(x[0]), x[0]))
        self._lex_day_anchor = sorted(                 # (phrase, key) longest-first
            [(p.lower(), key) for key, ps in d.get("day_anchors", {}).items() for p in ps],
            key=lambda x: (-len(x[0]), x[0]))
        self._lex_unit = {}                            # synonym -> canonical unit
        for canon, syns in d.get("relative_units", {}).items():
            for syn in syns:
                self._lex_unit[syn.lower()] = canon
        rm = d.get("relative_markers", {})
        self._lex_in = sorted([m.lower() for m in rm.get("in", [])], key=len, reverse=True)
        self._lex_at = sorted([m.lower() for m in rm.get("at", [])], key=len, reverse=True)
        # Explicit clock-hour markers ("18 h", "18 heures") — separate from relative_units.hour
        # so duration words ("time", "timer" in Danish; "Stunden" in German) are never treated
        # as clock-time suffixes. Only languages that set this key (currently French) use D1.
        self._lex_clock_hour = sorted(
            [m.lower() for m in d.get("clock_hour_markers", [])], key=len, reverse=True)

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
                # typos on longer names (“restraunt”→“restaurant”) still match.
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

    # ----- Lexicon-driven datetime parser (fr/de/da) -----

    @staticmethod
    def _lex_boundary(phrase: str) -> str:
        return _WB_L + re.escape(phrase) + _WB_R

    def _lex_normalise_numbers(self, t: str) -> str:
        """Replace spoken number words with digits (longest phrase first)."""
        for phrase, val in self._lex_number_phrases:
            t = re.sub(self._lex_boundary(phrase), str(val), t)
        return t

    def _lex_adjacent_hour(self, t: str, start: int, end: int):
        """Hour number adjacent to a decimal-hour idiom: look right (skipping the
        conjunction and hour-unit words), then left. Returns int or None."""
        skip = {self._lex_conj} if self._lex_conj else set()
        skip |= {syn for syn, c in self._lex_unit.items() if c == "hour"}

        def first_number(tokens):
            for tok in tokens:
                if tok in skip or tok == "":
                    continue
                if re.fullmatch(r"\d{1,2}", tok):
                    return int(tok)
                if tok in self._lex_number:
                    return self._lex_number[tok]
                return None  # first significant token is not a number -> stop
            return None

        rtok = re.findall(r"[0-9A-Za-zÀ-ÿ]+", t[end:])
        h = first_number(rtok)
        if h is not None:
            return h
        ltok = list(reversed(re.findall(r"[0-9A-Za-zÀ-ÿ]+", t[:start])))
        return first_number(ltok)

    def _extract_datetime_lex(self, text: str, now: datetime):
        """Returns (dt, time_explicit, explicit_day) or None. Mirrors
        EntityExtractor._extractDateTimeLexicon in Swift exactly."""
        t = text.lower().strip()

        def blank(s, a, b):
            return s[:a] + " " * (b - a) + s[b:]

        # A. Relative durations: "<in> N <unit>"
        if self._lex_in and self._lex_unit:
            inalt = "|".join(re.escape(m) for m in self._lex_in)
            ualt = "|".join(re.escape(u) for u in sorted(self._lex_unit, key=len, reverse=True))
            m = re.search(rf"{_WB_L}(?:{inalt})\s+(\d+)\s+({ualt}){_WB_R}", t)
            if m:
                n = int(m.group(1)); canon = self._lex_unit[m.group(2).lower()]
                delta = {"minute": timedelta(minutes=n), "hour": timedelta(hours=n),
                         "day": timedelta(days=n), "week": timedelta(weeks=n),
                         "month": timedelta(days=30 * n), "year": timedelta(days=365 * n)}[canon]
                return (now + delta, True, False)

        # B. yesterday / past rejection
        for phrase, key in self._lex_day_anchor:
            if key == "yesterday" and re.search(self._lex_boundary(phrase), t):
                return None

        t = self._lex_normalise_numbers(t)
        base_day = now; explicit_day = False; consumed_anchor = False

        # C. Day anchor (longest phrase first)
        for phrase, key in self._lex_day_anchor:
            if key == "yesterday":
                continue
            m = re.search(self._lex_boundary(phrase), t)
            if not m:
                continue
            off = {"today": 0, "tomorrow": 1, "day_after_tomorrow": 2,
                   "next_week": 7, "next_month": 30, "next_year": 365}.get(key)
            if key == "this_weekend":
                base_day = now + timedelta(days=(5 - now.weekday()) % 7)
            elif off is not None:
                base_day = now + timedelta(days=off)
            else:
                continue
            explicit_day = True; consumed_anchor = True
            t = blank(t, m.start(), m.end()); break

        # C2. Weekday names
        if not consumed_anchor:
            for syn, idx in sorted(self._lex_weekday.items(), key=lambda x: (-len(x[0]), x[0])):
                m = re.search(self._lex_boundary(syn), t)
                if m:
                    base_day = now + timedelta(days=((idx - now.weekday()) % 7 or 7))
                    explicit_day = True; consumed_anchor = True
                    t = blank(t, m.start(), m.end()); break

        # C3. Date of month "<day>[.] <month>"
        if not consumed_anchor and self._lex_month:
            malt = "|".join(re.escape(m) for m in sorted(self._lex_month, key=len, reverse=True))
            m = re.search(rf"{_WB_L}(\d{{1,2}})\.?\s+({malt}){_WB_R}", t)
            if m:
                day = int(m.group(1)); mon = self._lex_month[m.group(2).lower()]
                if 1 <= day <= 31:
                    cand = now.replace(month=mon, day=day, hour=9, minute=0, second=0, microsecond=0)
                    if cand.date() < now.date():
                        cand = cand.replace(year=now.year + 1)
                    base_day = cand; explicit_day = True
                    t = blank(t, m.start(), m.end())

        # D. Clock time
        hour = minute = None
        m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2)); t = blank(t, m.start(), m.end())
        if hour is None:
            m = re.search(r"\b(\d{1,2})h(\d{2})?\b", t)
            if m:
                hour = int(m.group(1)); minute = int(m.group(2)) if m.group(2) else 0
                t = blank(t, m.start(), m.end())
        if hour is None:
            m = re.search(r"\b(\d{1,2})\.(\d{2})\b", t)
            if m and 0 <= int(m.group(2)) <= 59:
                hour, minute = int(m.group(1)), int(m.group(2)); t = blank(t, m.start(), m.end())
        # D2. Absolute idioms (hour set: midi/minuit/Mitternacht/midnat)
        if hour is None:
            for phrase, mins, h in self._lex_idioms:
                if h is None:
                    continue
                m = re.search(self._lex_boundary(phrase), t)
                if m:
                    hour = h; minute = mins or 0; t = blank(t, m.start(), m.end()); break
        # D3. Decimal-hour idioms (minutes set) + adjacent hour number
        if hour is None:
            for phrase, mins, h in self._lex_idioms:
                if mins is None or h is not None:
                    continue
                m = re.search(self._lex_boundary(phrase), t)
                if not m:
                    continue
                hh = self._lex_adjacent_hour(t, m.start(), m.end())
                if hh is None:
                    continue
                if mins >= 0:
                    hour, minute = hh, mins
                else:
                    hour, minute = (hh - 1) % 24, 60 + mins
                t = blank(t, m.start(), m.end()); break
        # D1. Digit + spaced clock-hour marker: "18 h" / "18 heures".
        # Runs AFTER the decimal-hour idioms (D3) so a bare "N heures" doesn't
        # consume the hour before an adjacent "et demie" / "moins le quart" /
        # "et quart" adjustment can apply (otherwise the minutes are dropped).
        # Uses clock_hour_markers (not relative_units.hour) so duration words like
        # Danish "timer" or German "Stunden" are never misread as clock suffixes.
        if hour is None and self._lex_clock_hour:
            _hr_alt = "|".join(re.escape(s) for s in self._lex_clock_hour)
            m = re.search(rf"\b(\d{{1,2}})\s+(?:{_hr_alt})\b", t)
            if m:
                hour = int(m.group(1))
                minute = 0
                t = blank(t, m.start(), m.end())
        # D4. Named hour + conjunction ("15 Uhr 30", "drei Uhr")
        if hour is None and self._lex_conj:
            m = re.search(rf"\b(\d{{1,2}})\s+{re.escape(self._lex_conj)}(?:\s+(\d{{1,2}}))?\b", t)
            if m:
                hour = int(m.group(1)); minute = int(m.group(2)) if m.group(2) else 0
                t = blank(t, m.start(), m.end())
        # D5. "<at> <digit>[:mm]"
        if hour is None and self._lex_at:
            atalt = "|".join(re.escape(x) for x in self._lex_at)
            m = re.search(rf"{_WB_L}(?:{atalt})\s+(\d{{1,2}})(?::(\d{{2}}))?{_WB_R}", t)
            if m:
                hour = int(m.group(1)); minute = int(m.group(2)) if m.group(2) else 0
                t = blank(t, m.start(), m.end())
        # D6. Bare number as the whole input
        if hour is None:
            m = re.match(r"^\s*(\d{1,2})\s*$", t)
            if m:
                hour, minute = int(m.group(1)), 0

        # E. Period of day (longest name first)
        period_hour = None
        for name, ph in self._lex_period:
            if re.search(self._lex_boundary(name), t):
                period_hour = ph; break

        # F. Combine
        if hour is not None:
            minute = minute or 0
            if not (0 <= minute <= 59):
                return None
            # Explicit afternoon/evening/night shifts a 1-11 clock hour into PM.
            if period_hour is not None and period_hour >= 13 and 1 <= hour <= 11:
                hour += 12
            # 24h languages: never apply the bare 1-6 -> PM heuristic.
            if not (0 <= hour <= 23):
                return None
            dt = base_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if not explicit_day and dt <= now:
                dt += timedelta(days=1)
            if explicit_day and base_day.date() < now.date():
                return None
            return (dt, True, explicit_day)

        if period_hour is not None:
            dt = base_day.replace(hour=period_hour, minute=0, second=0, microsecond=0)
            if not explicit_day and dt <= now:
                dt += timedelta(days=1)
            return (dt, True, explicit_day)

        if explicit_day:
            return (base_day.replace(hour=9, minute=0, second=0, microsecond=0), False, True)

        return None

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

        # Non-English: drive the whole parse from the language lexicon. The
        # English path below is left byte-identical for language == "en".
        if self._lex is not None:
            res = self._extract_datetime_lex(text, now)
            if res is None:
                return None, None, 0.0, False, False
            dt, time_explicit, explicit_day = res
            return self._to_utc_iso(dt), text.strip(), 1.0, time_explicit, explicit_day

        t = text.lower().strip()

        # --- 1. Relative durations (markers/units/quantifiers from the grammar) ---
        # "in N <unit>" / "for N <unit>"
        m = re.search(rf"\b(?:{self._alt_in_for})\s+(\d+)\s*({self._alt_unit})\b", t)
        if m:
            n = int(m.group(1)); canon = self._unit_canon[m.group(2).lower()]
            return self._to_utc_iso(now + self._rel_delta(canon, n)), m.group(), 1.0, True, False
        # "in a/an <unit>"  (n = 1)
        m = re.search(rf"\b(?:{self._alt_in})\s+(?:{self._alt_article})\s+({self._alt_unit})\b", t)
        if m:
            canon = self._unit_canon[m.group(1).lower()]
            return self._to_utc_iso(now + self._rel_delta(canon, 1)), m.group(), 1.0, True, False
        # "in a few / a couple of <unit>" (quantifier count; minute/hour only,
        # preserving the original parser's scope)
        for qalt, qn in self._quant_specs:
            m = re.search(rf"\b(?:{self._alt_in})\s+(?:{qalt})\s*({self._alt_unit_mh})\b", t)
            if m:
                canon = self._unit_canon[m.group(1).lower()]
                return self._to_utc_iso(now + self._rel_delta(canon, qn)), m.group(), 1.0, True, False
        # "in half an hour"
        if self._alt_half_an_hour and re.search(
                rf"\b(?:{self._alt_in})\s+(?:{self._alt_half_an_hour})\b", t):
            return self._to_utc_iso(now + timedelta(minutes=30)), "in half an hour", 1.0, True, False

        # --- 2. Explicit past-date rejection (words from the grammar) ---
        if any(rex.search(t) for rex in self._yesterday_res):
            return None, None, 0.0, False, False

        # --- 3. Normalise word numbers so "nine pm" -> "9 pm", "nine thirty" -> "9 30" ---
        t = self._normalise_word_numbers(t)

        # --- 4. Day anchor (matchers + offsets from the grammar; match order
        # preserves day-after-tomorrow before tomorrow so the shared substring
        # cannot mis-fire) ---
        base_day = now
        explicit_day = False
        for rex, key in self._anchor_specs:
            if rex.search(t):
                base_day = now + timedelta(days=_ANCHOR_OFFSET.get(key, 0))
                explicit_day = True
                break
        else:
            for i, wd in enumerate(self._WEEKDAYS):
                if re.search(rf"\b{wd}\b", t):
                    ahead = (i - now.weekday()) % 7 or 7
                    base_day = now + timedelta(days=ahead)
                    explicit_day = True
                    break

        # --- 4b. Period hint from context words (matchers from the grammar) ---
        period = None
        for rex, name in self._period_specs:
            if rex.search(t):
                period = name
                break

        # --- 5. Named-only time (no digit); hours from the grammar table ---
        named_hour = dict(self._named_hour)

        # --- 6. Explicit time extraction ---
        hour = minute = None; span = None

        # Clock idioms — phrases from the grammar (half past / quarter past /
        # quarter to / N past M / N to M). Order and arithmetic unchanged.
        if self._alt_half_past:
            m = re.search(rf"\b(?:{self._alt_half_past})\s+(\d{{1,2}})\b", t)
            if m:
                hour, minute = int(m.group(1)), 30; span = m.group()
        if hour is None and self._alt_quarter_past:
            m = re.search(rf"\b(?:{self._alt_quarter_past})\s+(\d{{1,2}})\b", t)
            if m:
                hour, minute = int(m.group(1)), 15; span = m.group()
        if hour is None and self._alt_quarter_to:
            m = re.search(rf"\b(?:{self._alt_quarter_to})\s+(\d{{1,2}})\b", t)
            if m:
                h = int(m.group(1))
                if 1 <= h <= 12:
                    hour, minute = (h - 1) if h > 1 else 12, 45; span = m.group()
        # "N past M" (e.g. "20 past 9") — minutes 1-59, hour a valid clock hour
        if hour is None and self._alt_past:
            m = re.search(rf"\b(\d{{1,2}})\s+(?:{self._alt_past})\s+(\d{{1,2}})\b", t)
            if m:
                mm, hh = int(m.group(1)), int(m.group(2))
                if 0 <= mm <= 59 and 1 <= hh <= 12:
                    minute, hour = mm, hh; span = m.group()
        # "N to M" (e.g. "10 to 3" = 2:50) — minutes-to 1-59, hour a valid clock hour
        if hour is None and self._alt_to:
            m = re.search(rf"\b(\d{{1,2}})\s+(?:{self._alt_to})\s+(\d{{1,2}})\b", t)
            if m:
                mins_to, h = int(m.group(1)), int(m.group(2))
                if 1 <= mins_to <= 59 and 1 <= h <= 12:
                    hour = (h - 1) if h > 1 else 12
                    minute = 60 - mins_to; span = m.group()

        # Standard "9am", "9 am", "9:30pm" — am/pm markers from the grammar
        explicit_ampm = False
        if hour is None and self._alt_ampm:
            tm = re.search(rf"\b(\d{{1,2}})(?::(\d{{2}}))?\s*({self._alt_ampm})\b", t)
            if tm:
                raw_h = int(tm.group(1))
                is_pm = tm.group(3).lower() in self._pm_forms
                hour = raw_h % 12 + (12 if is_pm else 0)
                minute = int(tm.group(2) or 0)
                span = tm.group()
                explicit_ampm = True  # hour is already 0-23; skip _pick_future_hour

        # "9:30" with colon, no am/pm
        if hour is None:
            tm = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
            if tm:
                hour, minute = int(tm.group(1)), int(tm.group(2)); span = tm.group()

        # P0-6: Continental European "15h30", "9h" (French/German written clock)
        if hour is None:
            m = re.search(r"\b(\d{1,2})h(\d{2})?\b", t)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2)) if m.group(2) else 0
                span = m.group()

        # P0-6: Danish/German decimal notation "15.30" (minute block 00-59 avoids decimal false positives)
        if hour is None:
            m = re.search(r"\b(\d{1,2})\.(\d{2})\b", t)
            if m and 0 <= int(m.group(2)) <= 59:
                hour, minute = int(m.group(1)), int(m.group(2))
                span = m.group()

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
                return None, None, 0.0, False, False
            try:
                if 1 <= hour <= 12 and not explicit_ampm and period not in ("am",):
                    # Need disambiguation — use period hint
                    p = period if period in ("am","pm","morning","afternoon","evening","night") else period
                    dt = self._pick_future_hour(hour, minute, base_day, now, p)
                else:
                    # hour is already 0-23 (from am/pm regex or colon format)
                    if not (0 <= hour <= 23):
                        return None, None, 0.0, False, False
                    dt = base_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if not explicit_day and dt <= now:
                        dt += timedelta(days=1)
            except ValueError:
                return None, None, 0.0, False, False
            # Reject explicitly past dates (e.g. "yesterday", already-past specific date)
            if explicit_day and base_day.date() < now.date():
                return None, None, 0.0, False, False
            return self._to_utc_iso(dt), span, 1.0, True, explicit_day

        if explicit_day:
            # Day anchor found but NO time. Return the resolved day (defaulted to
            # 9am for backward-compatible callers) with time_explicit=False so the
            # engine can prompt for the missing time while keeping this day.
            dt = base_day.replace(hour=9, minute=0, second=0, microsecond=0)
            return self._to_utc_iso(dt), span, 1.0, False, True

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
                    return self._to_utc_iso(dt), t, 0.85, True, True

        return None, None, 0.0, False, False

    def is_open(self, entity: str) -> bool:
        return bool(self.entities.get(entity, {}).get("open"))

    def strip_datetime(self, text: str) -> str:
        if self._lex is not None:
            return self._strip_datetime_lex(text)
        t = text
        for pat in self._en_strip_patterns():
            t = pat.sub(" ", t)
        return re.sub(r"\s+", " ", t).strip(" .,")

    def _strip_datetime_lex(self, text: str) -> str:
        """Remove lexicon-recognised date/time fragments so the remainder is an
        open topic. Mirrors EntityExtractor.stripDateTime (lexicon path)."""
        t = text
        # Language-neutral clock forms.
        for p in (r"\b\d{1,2}:\d{2}\b", r"\b\d{1,2}h\d{0,2}\b", r"\b\d{1,2}\.\d{2}\b"):
            t = re.sub(p, " ", t, flags=re.I)
        # Spaced clock-hour forms: "18 h" / "18 heures" not caught by compact patterns above.
        if self._lex_clock_hour:
            _hr_alt = "|".join(re.escape(s) for s in self._lex_clock_hour)
            t = re.sub(rf"\b\d{{1,2}}\s+(?:{_hr_alt})\b", " ", t, flags=re.I)
        # Relative durations "<in> N <unit>".
        if self._lex_in and self._lex_unit:
            inalt = "|".join(re.escape(m) for m in self._lex_in)
            ualt = "|".join(re.escape(u) for u in sorted(self._lex_unit, key=len, reverse=True))
            t = re.sub(rf"{_WB_L}(?:{inalt})\s+\d+\s+(?:{ualt}){_WB_R}", " ", t, flags=re.I)
        # Phrase lists: anchors, weekdays, months, periods, idioms.
        phrases = ([p for p, _ in self._lex_day_anchor]
                   + list(self._lex_weekday.keys())
                   + list(self._lex_month.keys())
                   + [n for n, _ in self._lex_period]
                   + [p for p, _, _ in self._lex_idioms])
        for ph in sorted(phrases, key=len, reverse=True):
            t = re.sub(self._lex_boundary(ph), " ", t, flags=re.I)
        # Trailing markers + the conjunction.
        markers = self._lex_in + self._lex_at + ([self._lex_conj] if self._lex_conj else [])
        for mk in sorted(set(markers), key=len, reverse=True):
            t = re.sub(self._lex_boundary(mk), " ", t, flags=re.I)
        return re.sub(r"\s+", " ", t).strip(" .,")

    def extract(self, entity: str, text: str, fuzzy: bool = True):
        """Return (value, span, confidence). confidence=0.0 means no match.

        fuzzy is forwarded to enum matching; pass fuzzy=False for one-shot
        full-sentence scans to avoid stray approximate matches on common words.
        """
        if entity == "sys.date-time":
            iso, span, conf, _time_explicit, _explicit_day = self.extract_datetime(text)
            return iso, span, conf
        if entity == "sys.number-integer":  return self.extract_number(text)
        return self.extract_enum(entity, text, fuzzy=fuzzy)
