# Portable Regex Subset — normative definition (ADR-005 stage 10)

Every pattern shipped in a bundle (`keywords/*.json` rules and guards,
`entities/*` pattern entities, lexicon carriers) MUST conform to this subset.
It is the intersection that behaves **identically** in Python `re`, Swift
`NSRegularExpression` (ICU), Kotlin/Java `java.util.regex`, and Rust `regex` —
solving the ADR-001 §2.1 dialect problem by construction. Conformance is
proved by compiler stage 10 against the corpus below; runtimes never see a
pattern outside the subset.

## Allowed

| Construct | Syntax | Notes |
|---|---|---|
| Literals | `abc`, escaped metachars `\.` `\*` `\(` etc. | NFC-normalized before compile (stage 10) |
| Character classes | `[abc]`, `[a-z0-9]`, negated `[^abc]` | Ranges over ASCII + explicit non-ASCII literals only |
| Predefined classes | `\d`, `\w`, `\s` and negations `\D`, `\W`, `\S` | ASCII semantics (no Unicode-aware `\w`) |
| Any char | `.` | Never matches newline (patterns are single-line) |
| Greedy quantifiers | `*`, `+`, `?`, `{m}`, `{m,n}`, `{m,}` | `n ≤ 32` to bound backtracking |
| Alternation | `a\|b` | |
| Groups | `(...)` capturing, `(?:...)` non-capturing | Max nesting depth 4 |
| Anchors | `^`, `$` | |
| Word boundary | `\b`, `\B` | ASCII word-char definition |

## Forbidden

Backreferences (`\1`), lookahead/lookbehind (`(?=)`, `(?!)`, `(?<=)`, `(?<!)`),
lazy/possessive quantifiers (`*?`, `+?`, `?+`), named groups (`(?P<x>)`,
`(?<x>)`), inline flags (`(?i)`), conditionals, recursion, atomic groups,
Unicode property escapes (`\p{L}`), octal/hex char escapes except `\xNN`,
comments (`(?#…)`), and multiline/dotall semantics.

Rationale for the two most-missed features: **lookaround** differs subtly in
zero-width-assertion + anchor interaction across engines and defeats the
linear-time guarantee of Rust `regex` (which rejects it outright, making this
constraint future-proof for the ADR-001 shared core). **Case-insensitivity**
is expressed as the rule-level `case_insensitive` flag (data, not inline
syntax) so every engine applies its own well-tested top-level flag rather
than parsing inline modifiers.

## Semantics fixed by this spec

1. Matching is **single-line**: input contains no newlines (utterances are
   pre-normalized); `.` and `$` behave accordingly on every engine.
2. Patterns are matched against **NFC-normalized, casefolded-if-flagged**
   text; the runtime applies the same normalization declared in the bundle
   (`nfc_casefold`).
3. `find` semantics (leftmost match anywhere) unless the pattern is anchored.
4. Max pattern length 512 chars; compiled-size caps are a compiler concern.

## Conformance corpus (normative, machine-checked)

Compiler stage 10 and every runtime's regex shim MUST agree on all rows.
`spec/examples/` golden bundles embed patterns drawn only from this corpus's
allowed constructs.

| # | Pattern | Must match | Must NOT match |
|---|---|---|---|
| 1 | `^mute$` | `mute` | `unmute`, `mute please` |
| 2 | `\bvolume\b` | `the volume up`, `volume` | `voluminous` |
| 3 | `set (?:an? )?alarm` | `set an alarm`, `set alarm` | `preset alarms` *(unanchored find: matches `set alarm` inside? No — corpus text is exactly the cell)* |
| 4 | `[0-9]{1,2}:[0-9]{2}` | `8:30`, `12:05` | `123:456` *(no 3-digit hour match at same span)*, `8-30` |
| 5 | `louder\|quieter` | `louder`, `quieter` | `loud` |
| 6 | `remind(er)?s?\b` | `remind`, `reminders` | `remainder` |
| 7 | `\d+ ?(minutes?\|mins?)\b` | `10 minutes`, `5 min` | `minute 10` |
| 8 | `[^a-z]` | `A`, `9` | `a` |

Row 3's parenthetical illustrates the rule for reading the corpus: each cell
is the **entire input string**, matched with find semantics.

## Versioning

This file is part of the format-3.x contract. Additions to the allowed set
are a format **minor** bump (old patterns stay valid); removals or semantic
changes are a **major** bump (Part 8 of ADR-005).
