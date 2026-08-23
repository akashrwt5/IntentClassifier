# Error Analysis

Total errors captured across 7 suites: **408**

## By cause

| category | n | share |
|---|---|---|
| stt_error | 238 | 58.3% |
| other_embedding_or_classifier_weakness | 50 | 12.3% |
| same_family_opposite_or_sibling | 26 | 6.4% |
| negation | 21 | 5.1% |
| low_margin_uncertainty | 19 | 4.7% |
| result_request_vs_feature_question | 18 | 4.4% |
| confident_and_wrong_calibration_issue | 13 | 3.2% |
| insufficient_training_coverage | 13 | 3.2% |
| long_context | 6 | 1.5% |
| ood_false_acceptance | 4 | 1.0% |

## By suite

| suite | confident_and_wrong_calibration_issue | insufficient_training_coverage | long_context | low_margin_uncertainty | negation | ood_false_acceptance | other_embedding_or_classifier_weakness | result_request_vs_feature_question | same_family_opposite_or_sibling | stt_error |
|---|---|---|---|---|---|---|---|---|---|---|
| contextual | 0 | 0 | 6 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| hard_negatives | 0 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 3 | 0 |
| minimal_pairs | 0 | 3 | 0 | 0 | 0 | 0 | 4 | 2 | 2 | 0 |
| negation | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 |
| ood | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| standard_test | 13 | 10 | 0 | 19 | 6 | 0 | 45 | 15 | 21 | 0 |
| stt | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 238 |

## Most frequent confusions (true -> predicted)

| true | predicted | n |
|---|---|---|
| `Default Fallback Intent` | `Cmd.StreamingStart` | 14 |
| `Default Fallback Intent` | `Cmd.MemoryChange` | 13 |
| `Default Fallback Intent` | `reminders.add` | 9 |
| `Help_Accessories` | `Help_Pairing` | 8 |
| `Default Fallback Intent` | `Cmd.VolumeDecrease` | 8 |
| `Default Fallback Intent` | `Help_DeviceSettings` | 8 |
| `Help_Battery` | `Cmd.BatteryLevel` | 7 |
| `Default Fallback Intent` | `Help_Home` | 5 |
| `Cmd.VolumeDecrease` | `Cmd.VolumeUnmute` | 5 |
| `Cmd.VolumeDecrease` | `Default Fallback Intent` | 5 |
| `Cmd.VolumeIncrease` | `Cmd.VolumeDecrease` | 5 |
| `Cmd.SendMessage` | `Default Fallback Intent` | 5 |
| `Help_RemoteProgramming` | `Default Fallback Intent` | 5 |
| `Default Fallback Intent` | `Cmd.StreamingStop` | 5 |
| `Default Fallback Intent` | `Cmd.VolumeUnmute` | 5 |
| `Cmd.MemoryChange` | `Default Fallback Intent` | 5 |
| `Default Fallback Intent` | `Help_DemoMode` | 5 |
| `Help_HeartRate` | `Help_HeartRateRecovery` | 5 |
| `Default Fallback Intent` | `Help_RemoteProgramming` | 5 |
| `Cmd.VolumeMute` | `Cmd.VolumeDecrease` | 4 |
| `Cmd.TranslationStart` | `Help_Translate` | 4 |
| `Default Fallback Intent` | `Help_SelfCheck` | 4 |
| `Default Fallback Intent` | `Cmd.VolumeMute` | 4 |
| `Help_MemoryOptions` | `Cmd.MemoryChange` | 4 |
| `Cmd.ActivityStep` | `Default Fallback Intent` | 4 |

## Errors that the safety gate ACCEPTED (false executions)

These are the dangerous ones: the model was wrong and the gate let it through. Count: **45**

| text | true | predicted | conf | margin |
|---|---|---|---|---|
| uh a setting for verywindy places | `Help_EdgeMode` | `Cmd.MemoryChange` | 0.998 | 0.997 |
| and push it down for | `Default Fallback Intent` | `Cmd.VolumeDecrease` | 0.995 | 0.993 |
| not the recovery number, just my heart rate | `Help_HeartRate` | `Help_HeartRateRecovery` | 0.992 | 0.985 |
| er can you and push it down for | `Default Fallback Intent` | `Cmd.VolumeDecrease` | 0.991 | 0.987 |
| um can i snooze a reminda | `Help_Reminder` | `reminders.add` | 0.991 | 0.986 |
| creating a demo with your capabilities | `Default Fallback Intent` | `Help_DemoMode` | 0.987 | 0.984 |
| numbers | `Default Fallback Intent` | `Cmd.MemoryChange` | 0.987 | 0.976 |
| hear it as they got to go. | `Default Fallback Intent` | `Cmd.StreamingStart` | 0.984 | 0.979 |
| can i charge my battery? | `Help_Battery` | `Cmd.BatteryLevel` | 0.983 | 0.973 |
| uh hear it as they got to | `Default Fallback Intent` | `Cmd.StreamingStart` | 0.981 | 0.967 |
| can you listen to music | `Default Fallback Intent` | `Cmd.MemoryChange` | 0.979 | 0.961 |
| i want silence | `Cmd.VolumeDecrease` | `Cmd.VolumeMute` | 0.978 | 0.966 |
| uhright on | `Default Fallback Intent` | `Cmd.VolumeUnmute` | 0.974 | 0.962 |
| and push it down for dramatics | `Default Fallback Intent` | `Cmd.VolumeDecrease` | 0.974 | 0.966 |
| a setting for very windy places | `Help_EdgeMode` | `Cmd.MemoryChange` | 0.973 | 0.953 |
| do not tell me how, just change the volume | `Cmd.VolumeDecrease` | `Help_Volume` | 0.973 | 0.954 |
| uh canyou listen to music | `Default Fallback Intent` | `Cmd.MemoryChange` | 0.973 | 0.951 |
| hearing aid not charging | `Help_SelfCheck` | `Help_Battery` | 0.971 | 0.949 |
| can you creating a demo with your capabilities | `Default Fallback Intent` | `Help_DemoMode` | 0.968 | 0.951 |
| um this mode is not working for my surrounding | `Cmd.MemoryChange` | `Help_MaskMode` | 0.967 | 0.940 |
| this mode isn't working for my surroundings | `Cmd.MemoryChange` | `Help_MaskMode` | 0.965 | 0.936 |
| please i need help with my | `Help_DeviceSettings` | `Help_Accessories` | 0.959 | 0.945 |
| how to share my wellness data? | `Help_HearShare` | `Help_HearingCareAnywhereConnect` | 0.955 | 0.929 |
| can you and push it down for dramatics | `Default Fallback Intent` | `Cmd.VolumeDecrease` | 0.954 | 0.939 |
| er creatinga demo with your capabilities | `Default Fallback Intent` | `Help_DemoMode` | 0.951 | 0.920 |
| please and push it down for dramatics | `Default Fallback Intent` | `Cmd.VolumeDecrease` | 0.949 | 0.924 |
| how much energy did my gym session burn | `Cmd.ActivityCalories` | `Cmd.ActivityExercise` | 0.945 | 0.905 |
| explain the personalization options | `Help_Customize` | `Help_MemoryOptions` | 0.942 | 0.923 |
| you help forspeech to text | `Help_Transcribe` | `Cmd.SendMessage` | 0.940 | 0.891 |
| er can yourecord this meeting | `Cmd.TranscribeStart` | `reminders.add` | 0.935 | 0.912 |

## Next targeted data batch

Ordered by how many errors each cause explains. Per Phase 23, the next batch addresses these and nothing else.

- **stt_error** (238) — add the observed corruption pattern to the STT augmentation set
- **other_embedding_or_classifier_weakness** (50) — candidate for a stronger encoder
- **same_family_opposite_or_sibling** (26) — add minimal pairs inside that family
- **negation** (21) — extend the P2/P3 negation templates with new openers and objects
- **low_margin_uncertainty** (19) — raise the margin threshold or add separating examples
- **result_request_vs_feature_question** (18) — add matched result-request / feature-question pairs for the affected topic (policy P1)
- **confident_and_wrong_calibration_issue** (13) — re-check calibration; consider per-class temperature
- **insufficient_training_coverage** (13) — collect or generate examples for the tail intents
- **long_context** (6) — add long conversational forms for the affected intents
- **ood_false_acceptance** (4) — add near-OOD training examples for the accepting intent
