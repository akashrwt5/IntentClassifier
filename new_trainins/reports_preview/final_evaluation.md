# Final Evaluation

Model: **tfidf-svd + mlp**

## Evaluation matrix

| suite | purpose | result |
|---|---|---|
| Standard test | general classification | acc **0.6636**, macro-F1 **0.4807** |
| Contextual | long/natural requests | acc 0.0000 |
| Minimal pairs | opposite-intent separation | item 0.4205, both-sides 0.2273 |
| Hard negatives | shortcut resistance | acc 0.1667 |
| Negation | scope handling (P2/P3) | acc 0.1538 |
| STT | recognition noise | acc 0.4332 |
| OOD | unknown rejection | rejection 1.0000, false acceptance 0.0000 |
| Calibration | trustworthy confidence | ECE 0.0291 |

## The production statement

> At the selected operating threshold, accepted predictions achieve **99.62% precision** on the held-out test set, with **17.51% coverage**.

- false execution rate (wrong **and** accepted): **0.0007**
- false rejection rate (right but gated out): 0.4891

## Negation policy breakdown

| policy | accuracy |
|---|---|
| P2_bare_negation | 0.1429 |
| P3_corrective | 0.1667 |

## Minimal pairs by axis

| axis | item accuracy |
|---|---|
| activity_type | 0.1667 |
| cmd_vs_help | 0.3000 |
| feature_swap | 0.5000 |
| find_target | 0.2500 |
| health_metric | 1.0000 |
| message_direction | 0.5000 |
| mute_direction | 0.3750 |
| reminder_direction | 0.7500 |
| stream_direction | 0.6667 |
| volume_direction | 0.4375 |

## Python vs ONNX parity

| build | max abs delta | top-1 agreement | gate agreement | size MB | within tolerance |
|---|---|---|---|---|---|
| fp32 | 9.537e-07 | 1.00000 | 1.00000 | 0.455 | True |
| int8 | 1.709e-01 | 0.98820 | 0.99034 | 0.118 | False |

## Comparison against the existing baseline

| metric | baseline (INT8 semantic student) | this run |
|---|---|---|
| intents | 11 | 57 |
| model size | 0.236 MB | see parity table |
| contextual accuracy | 0.9062 | 0.0000 |
| OOD rejection | 0.3438 | 1.0000 |

> These are **not** like-for-like. The baseline numbers are for an 11-intent problem; this run is 57 intents on a leakage-controlled split with harder held-out suites. A 57-way problem with a 35x class imbalance is a strictly harder task, so the baseline's accuracy is not a ceiling this run failed to reach — it is a different measurement. The only honest comparison is to re-run the baseline model against these same suites.
