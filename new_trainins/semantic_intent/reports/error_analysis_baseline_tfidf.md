# Error Analysis

Total errors captured across 7 suites: **513**

## By cause

| category | n | share |
|---|---|---|
| stt_error | 265 | 51.7% |
| command_vs_help_confusion | 136 | 26.5% |
| same_family_opposite_or_sibling | 41 | 8.0% |
| negation | 36 | 7.0% |
| other_embedding_or_classifier_weakness | 12 | 2.3% |
| insufficient_training_coverage | 11 | 2.1% |
| ood_false_acceptance | 8 | 1.6% |
| low_margin_uncertainty | 3 | 0.6% |
| confident_and_wrong_calibration_issue | 1 | 0.2% |

## By suite

| suite          |   command_vs_help_confusion |   confident_and_wrong_calibration_issue |   insufficient_training_coverage |   low_margin_uncertainty |   negation |   ood_false_acceptance |   other_embedding_or_classifier_weakness |   same_family_opposite_or_sibling |   stt_error |
|:---------------|----------------------------:|----------------------------------------:|---------------------------------:|-------------------------:|-----------:|-----------------------:|-----------------------------------------:|----------------------------------:|------------:|
| contextual     |                          13 |                                       0 |                                0 |                        0 |          2 |                      0 |                                        0 |                                 0 |           0 |
| hard_negatives |                           0 |                                       0 |                                0 |                        0 |         19 |                      0 |                                        0 |                                 2 |           0 |
| minimal_pairs  |                          10 |                                       0 |                                3 |                        1 |          0 |                      0 |                                        0 |                                 6 |           0 |
| negation       |                           1 |                                       0 |                                0 |                        0 |          8 |                      0 |                                        0 |                                 0 |           0 |
| ood            |                           0 |                                       0 |                                0 |                        0 |          0 |                      8 |                                        0 |                                 0 |           0 |
| standard_test  |                         112 |                                       1 |                                8 |                        2 |          7 |                      0 |                                       12 |                                33 |           0 |
| stt            |                           0 |                                       0 |                                0 |                        0 |          0 |                      0 |                                        0 |                                 0 |         265 |

## Most frequent confusions (true -> predicted)

| true | predicted | n |
|---|---|---|
| `Default Fallback Intent` | `Cmd.VolumeIncrease` | 14 |
| `Cmd.VolumeIncrease` | `Cmd.VolumeUnmute` | 13 |
| `Default Fallback Intent` | `Cmd.VolumeDecrease` | 12 |
| `Default Fallback Intent` | `Cmd.BatteryLevel` | 12 |
| `Default Fallback Intent` | `reminders.add` | 11 |
| `Cmd.MemoryChange` | `Default Fallback Intent` | 11 |
| `Default Fallback Intent` | `Cmd.MemoryChange` | 10 |
| `Default Fallback Intent` | `Cmd.ListenMessage` | 10 |
| `Cmd.StreamingStart` | `Default Fallback Intent` | 9 |
| `Help_SelfCheck` | `Default Fallback Intent` | 9 |
| `Cmd.VolumeDecrease` | `Default Fallback Intent` | 9 |
| `Default Fallback Intent` | `Help_Home` | 8 |
| `Default Fallback Intent` | `Help_Pairing` | 8 |
| `Help_Home` | `Default Fallback Intent` | 8 |
| `Cmd.VolumeIncrease` | `Cmd.VolumeDecrease` | 7 |
| `Help_ChangingMemories` | `Default Fallback Intent` | 7 |
| `Cmd.VolumeMute` | `Cmd.VolumeDecrease` | 7 |
| `Cmd.SendMessage` | `Default Fallback Intent` | 7 |
| `Default Fallback Intent` | `Help_DemoMode` | 6 |
| `Cmd.VolumeDecrease` | `Cmd.VolumeUnmute` | 5 |
| `Default Fallback Intent` | `Cmd.StreamingStart` | 5 |
| `Default Fallback Intent` | `Help_CleanCare` | 5 |
| `Help_MemoryOptions` | `Cmd.MemoryChange` | 5 |
| `Help_FindMyHearingAids` | `Help_Home` | 5 |
| `Default Fallback Intent` | `Help_IntelliVoice` | 4 |

## Errors that the safety gate ACCEPTED (false executions)

These are the dangerous ones: the model was wrong and the gate let it through. Count: **29**

| text | true | predicted | conf | margin |
|---|---|---|---|---|
| please i said other program | `Default Fallback Intent` | `Cmd.MemoryChange` | 0.991 | 0.989 |
| um you i said other program | `Default Fallback Intent` | `Cmd.MemoryChange` | 0.987 | 0.983 |
| can you i said other program | `Default Fallback Intent` | `Cmd.MemoryChange` | 0.987 | 0.983 |
| not the recovery number, just my heart rate | `Help_HeartRate` | `Help_HeartRateRecovery` | 0.985 | 0.978 |
| i said other program | `Default Fallback Intent` | `Cmd.MemoryChange` | 0.982 | 0.975 |
| turn the volumeup on the hearing | `Cmd.VolumeIncrease` | `Cmd.VolumeUnmute` | 0.981 | 0.973 |
| i said other pro gram | `Default Fallback Intent` | `Cmd.MemoryChange` | 0.979 | 0.975 |
| um what do i need to knowabout this new app | `Help_WhatsNew` | `Help_Home` | 0.966 | 0.936 |
| can you creatinga demo with your capabilitie | `Default Fallback Intent` | `Help_DemoMode` | 0.960 | 0.926 |
| get down south | `Default Fallback Intent` | `Cmd.VolumeDecrease` | 0.960 | 0.946 |
| personalize my hearingaid setting | `Help_MemoryOptions` | `Cmd.MemoryChange` | 0.956 | 0.941 |
| er creatinga demo with your capabilities | `Default Fallback Intent` | `Help_DemoMode` | 0.948 | 0.903 |
| how would i jump to setting number three | `Help_ChangingMemories` | `Cmd.MemoryChange` | 0.946 | 0.938 |
| not edge mode, i meant mask mode | `Help_MaskMode` | `Help_EdgeMode` | 0.942 | 0.896 |
| please turn the volume up on the hearing aid | `Cmd.VolumeIncrease` | `Cmd.VolumeUnmute` | 0.938 | 0.909 |
| i need a car battery | `Default Fallback Intent` | `Cmd.BatteryLevel` | 0.935 | 0.881 |
| how do i make the sound suit me | `Help_Customize` | `Help_Volume` | 0.924 | 0.892 |
| why cannot i hear anything through hearin aids | `Help_SelfCheck` | `Help_Pairing` | 0.922 | 0.904 |
| creating a demo with your capabilities | `Default Fallback Intent` | `Help_DemoMode` | 0.921 | 0.847 |
| can you turn the volume up on the hearing aid | `Cmd.VolumeIncrease` | `Cmd.VolumeUnmute` | 0.920 | 0.882 |
| i want to hearmy massage | `Cmd.ListenMessage` | `Cmd.SendMessage` | 0.919 | 0.893 |
| relay this information | `Cmd.SendMessage` | `Cmd.MemoryChange` | 0.919 | 0.900 |
| can you personalize hearing aid setting | `Help_MemoryOptions` | `Cmd.MemoryChange` | 0.917 | 0.881 |
| can you creating a demo with your capabilities | `Default Fallback Intent` | `Help_DemoMode` | 0.914 | 0.834 |
| please i need a car battery | `Default Fallback Intent` | `Cmd.BatteryLevel` | 0.914 | 0.841 |
| i turned it down too far, take it back up | `Cmd.VolumeIncrease` | `Cmd.VolumeDecrease` | 0.914 | 0.844 |
| do i have a normal memory | `Help_ChangingMemories` | `Help_MemoryOptions` | 0.905 | 0.844 |
| how do i balance volume between | `Help_Volume` | `Help_WiCROS` | 0.903 | 0.848 |
| can you thanks for changing | `Default Fallback Intent` | `Cmd.MemoryChange` | 0.902 | 0.836 |

## Next targeted data batch

Ordered by how many errors each cause explains. Per Phase 23, the next batch addresses these and nothing else.

- **stt_error** (265) — add the observed corruption pattern to the STT augmentation set
- **command_vs_help_confusion** (136) — add matched imperative/interrogative pairs for the affected intent families
- **same_family_opposite_or_sibling** (41) — add minimal pairs inside that family
- **negation** (36) — extend the P2/P3 negation templates with new openers and objects
- **other_embedding_or_classifier_weakness** (12) — candidate for a stronger encoder
- **insufficient_training_coverage** (11) — collect or generate examples for the tail intents
- **ood_false_acceptance** (8) — add near-OOD training examples for the accepting intent
- **low_margin_uncertainty** (3) — raise the margin threshold or add separating examples
- **confident_and_wrong_calibration_issue** (1) — re-check calibration; consider per-class temperature
