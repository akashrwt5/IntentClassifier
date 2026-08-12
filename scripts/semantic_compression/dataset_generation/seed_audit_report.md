# Seed Corpus Audit

_Generated 2026-08-12 18:49 UTC by `seed_audit.py` from `dialogflow-en-dataset/`._

## 1. Headline numbers

| Metric | Value |
|---|---|
| Seed files on disk | 63 |
| Excluded as entity lists | 3 |
| Merged into another intent | 1 |
| Dropped from taxonomy | 2 |
| **Resolved intents** | **57** |
| Raw non-empty lines | 3865 |
| Unique utterances after normalisation | 3472 |

## 2. Encodings

| Encoding | Files |
|---|---|
| `utf-16` | 61 |
| `utf-8-sig` | 2 |

> Reading these with `encoding='utf-8'` raises `UnicodeDecodeError`. Any
> caller that catches and skips on failure loses the intent silently.

## 3. Unexpected characters

| Codepoint | Occurrences |
|---|---|
| U+00A0 NO-BREAK SPACE | 2395 |
| U+2019 RIGHT SINGLE QUOTATION MARK | 26 |
| U+00F1 LATIN SMALL LETTER N WITH TILDE | 3 |

> `U+00A0 NO-BREAK SPACE` is used as a word separator in parts of the
> export. Tokenising on `str.split()` without NFKC normalisation
> collapses a whole utterance into a single token.

## 4. Cross-intent collisions

Utterances appearing under more than one intent. `% of smaller` is the
share of the smaller intent's unique utterances that the larger one also
claims; at 100% the smaller intent has no distinguishing evidence.

| Intent A | Intent B | Shared | % of smaller |
|---|---|---:|---:|
| `Help_HearingCareAnywhereConnect` | `Help_RemoteProgramming` | 72 | 100% |
| `Cmd.ActivityRun` | `Cmd.Health` | 30 | 100% |
| `Cmd.ActivityCycle` | `Cmd.Health` | 28 | 100% |
| `Help.Activity` | `Help_Activity` | 25 | 100% |
| `Cmd.ActivityWalk` | `Cmd.Health` | 23 | 100% |
| `Cmd.ActivityCalories` | `Cmd.Health` | 10 | 100% |
| `Cmd.ActivityStand` | `Cmd.Health` | 22 | 96% |
| `Cmd.ActivityAerobics` | `Cmd.Health` | 12 | 92% |
| `Cmd.ActivityStep` | `Cmd.Health` | 19 | 83% |
| `Cmd.ActivityExercise` | `Cmd.Health` | 11 | 65% |
| `Default Fallback Intent` | `_EntityRecurrence` | 10 | 22% |
| `Cmd.ActivityExercise` | `Help.Activity` | 2 | 12% |
| `Cmd.ActivityExercise` | `Help_Activity` | 2 | 12% |
| `Cmd.Health` | `Help_Activity` | 2 | 8% |
| `Cmd.Health` | `Help.Activity` | 2 | 7% |

## 5. Applied taxonomy rules

**Excluded (entity value lists):**

- `_EntityMemory` — entity value list, not an intent
- `_EntityRecurrence` — entity value list, not an intent
- `_EntityRemind` — entity value list, not an intent

**Merged:**

- `Help.Activity` → `Help_Activity`

**Dropped:**

- `Cmd.Health` — Rollup PARENT intent, not a sibling. 155 of its 160 unique utterances are drawn verbatim from the Cmd.Activity* children, leaving only 5 of its own. Keeping it would train the classifier on an impossible distinction (Cmd.Health vs Cmd.ActivityRun on identical text) and produce exactly the contradictory labels the blueprint forbids. Decision: keep the specific Cmd.Activity* intents; drop the parent.
- `Help_HearingCareAnywhereConnect` — Disabled in Dialogflow. All 72 of its unique utterances also appear under Help_RemoteProgramming (100% subset, zero distinguishing phrase).

## 6. Per-intent counts (resolved taxonomy)

| Intent | Family | Raw lines | Unique |
|---|---|---:|---:|
| `Cmd.ActivityAerobics` | ActivityTracking | 13 | 13 |
| `Cmd.ActivityCalories` | ActivityTracking | 10 | 10 |
| `Cmd.ActivityCycle` | ActivityTracking | 28 | 28 |
| `Cmd.ActivityExercise` | ActivityTracking | 17 | 17 |
| `Cmd.ActivityRun` | ActivityTracking | 30 | 30 |
| `Cmd.ActivityStand` | ActivityTracking | 24 | 23 |
| `Cmd.ActivityStep` | ActivityTracking | 25 | 23 |
| `Cmd.ActivityWalk` | ActivityTracking | 23 | 23 |
| `Cmd.BatteryLevel` | DeviceStatus | 11 | 11 |
| `Cmd.EdgeModeDeactivate` | EdgeMode | 16 | 16 |
| `Cmd.EdgeModeDecrease` | EdgeMode | 26 | 25 |
| `Cmd.EdgeModeIncrease` | EdgeMode | 152 | 152 |
| `Cmd.FindMyPhone` | DeviceLocate | 43 | 43 |
| `Cmd.StreamingStart` | Streaming | 43 | 43 |
| `Cmd.StreamingStop` | Streaming | 26 | 25 |
| `Cmd.TranscribeStart` | SpeechServices | 16 | 16 |
| `Cmd.TranslationStart` | SpeechServices | 22 | 22 |
| `Cmd.VolumeDecrease` | AudioControl | 62 | 60 |
| `Cmd.VolumeIncrease` | AudioControl | 71 | 68 |
| `Cmd.VolumeMute` | AudioControl | 33 | 33 |
| `Cmd.VolumeUnmute` | AudioControl | 32 | 32 |
| `Default Fallback Intent` | Fallback | 614 | 613 |
| `Help_Accessories` | HelpDeviceCare | 98 | 96 |
| `Help_Activity` | HelpHealth | 57 | 29 |
| `Help_AppSettings` | HelpAppSettings | 23 | 23 |
| `Help_Battery` | HelpDeviceCare | 19 | 18 |
| `Help_ChangingMemories` | HelpAppSettings | 92 | 91 |
| `Help_CleanCare` | HelpDeviceCare | 25 | 25 |
| `Help_Customize` | HelpAppSettings | 36 | 33 |
| `Help_DemoMode` | HelpAppSettings | 31 | 28 |
| `Help_DeviceSettings` | HelpAppSettings | 91 | 88 |
| `Help_EdgeMode` | HelpAudio | 20 | 19 |
| `Help_FallAlert` | HelpHealth | 108 | 108 |
| `Help_FindMyHearingAids` | HelpFind | 106 | 105 |
| `Help_Health` | HelpHealth | 23 | 23 |
| `Help_HearShare` | HelpConnectivity | 26 | 26 |
| `Help_HeartRate` | HelpHealth | 19 | 19 |
| `Help_HeartRateRecovery` | HelpHealth | 22 | 22 |
| `Help_Home` | HelpAppSettings | 23 | 23 |
| `Help_InsertDevice` | HelpDeviceCare | 30 | 30 |
| `Help_IntelliVoice` | HelpAudio | 42 | 41 |
| `Help_MaskMode` | HelpAudio | 18 | 16 |
| `Help_MemoryOptions` | HelpAppSettings | 64 | 61 |
| `Help_Pairing` | HelpConnectivity | 139 | 135 |
| `Help_Reminder` | Reminders | 23 | 23 |
| `Help_RemoteProgramming` | HelpConnectivity | 232 | 232 |
| `Help_SelfCheck` | HelpDeviceCare | 66 | 65 |
| `Help_ThriveScore` | HelpHealth | 65 | 65 |
| `Help_Tinnitus` | HelpAudio | 130 | 129 |
| `Help_Transcribe` | HelpSpeechServices | 37 | 35 |
| `Help_Translate` | HelpSpeechServices | 43 | 43 |
| `Help_VoiceAssistant` | HelpSpeechServices | 30 | 30 |
| `Help_Volume` | HelpAudio | 122 | 121 |
| `Help_WhatsNew` | HelpAppSettings | 37 | 37 |
| `Help_WiCROS` | HelpDeviceCare | 34 | 34 |
| `reminders.add` | Reminders | 255 | 254 |
| `reminders.complete` | Reminders | 20 | 19 |

## 7. Suspected transcription noise (review candidates)

Noise guard is **REPORT-ONLY — nothing is withheld** (`seed_sampling.noise_guard.enabled`).

Phrases carrying several words unique to their intent. Max-diversity
sampling is attracted to outliers, so these are among the likeliest
lines to reach the LLM. The heuristic is *not* reliable enough to
filter on: it penalises lexical novelty, which is the very signal this
project wants. Treat the counts as a human review queue, not a verdict.

125 phrases flagged across 22 intents.

| Intent | Flagged |
|---|---:|
| `Default Fallback Intent` | 89 |
| `Help_SelfCheck` | 4 |
| `Help_Transcribe` | 3 |
| `Help_Volume` | 3 |
| `reminders.add` | 3 |
| `Cmd.EdgeModeIncrease` | 2 |
| `Cmd.StreamingStart` | 2 |
| `Cmd.VolumeIncrease` | 2 |
| `Help_DeviceSettings` | 2 |
| `Help_MemoryOptions` | 2 |
| `Help_Translate` | 2 |
| `Cmd.ActivityRun` | 1 |
| `Cmd.FindMyPhone` | 1 |
| `Cmd.VolumeDecrease` | 1 |
| `Help_ChangingMemories` | 1 |
| `Help_FallAlert` | 1 |
| `Help_InsertDevice` | 1 |
| `Help_Pairing` | 1 |
| `Help_Tinnitus` | 1 |
| `Help_VoiceAssistant` | 1 |
| `Help_WhatsNew` | 1 |
| `Help_WiCROS` | 1 |

Phrase text is omitted by default. These are raw production ASR
transcripts and this repository is public; the flagged lines are
disproportionately the ones carrying names, addresses and other
personal detail, precisely because the heuristic selects for rare
words. Run `seed_audit.py --include-phrases` locally to read them,
and do not commit that output.

## 8. Thin intents

Fewer than 15 unique seed utterances. The bootstrapper has little
evidence to reverse-engineer boundaries from, so review these specs by
hand before Stage 1 generation.

- `Cmd.ActivityAerobics` (13)
- `Cmd.ActivityCalories` (10)
- `Cmd.BatteryLevel` (11)
