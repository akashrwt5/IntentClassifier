# Final Evaluation

Model: **bge-small-en-v1.5 + mlp**

## Evaluation matrix

| suite | purpose | result |
|---|---|---|
| Standard test | general classification | acc **0.9141**, macro-F1 **0.9038** |
| Contextual | long/natural requests | acc 0.5379 |
| Minimal pairs | opposite-intent separation | item 0.8295, both-sides 0.7045 |
| Hard negatives | shortcut resistance | acc 0.6823 |
| Negation | scope handling (P2/P3) | acc 0.5897 |
| STT | recognition noise | acc 0.8509 |
| OOD | unknown rejection | rejection 0.8667, false acceptance 0.1333 |
| Calibration | trustworthy confidence | ECE 0.0146 |

## The production statement

> At the selected operating threshold, accepted predictions achieve **98.32% precision** on the held-out test set, with **70.79% coverage**.

- false execution rate (wrong **and** accepted): **0.0119**
- false rejection rate (right but gated out): 0.2181

## Negation policy breakdown

| policy | accuracy |
|---|---|
| P2_bare_negation | 0.5794 |
| P3_corrective | 0.6333 |

## Minimal pairs by axis

| axis | item accuracy |
|---|---|
| activity_type | 0.7500 |
| cmd_vs_help | 0.7500 |
| feature_swap | 1.0000 |
| find_target | 1.0000 |
| health_metric | 0.7500 |
| message_direction | 0.7500 |
| mute_direction | 0.8750 |
| reminder_direction | 1.0000 |
| stream_direction | 0.8333 |
| volume_direction | 0.8125 |

## Python vs ONNX parity

| build | max abs delta | top-1 agreement | gate agreement | size MB | within tolerance |
|---|---|---|---|---|---|
| fp32 | 6.491e-05 | 1.00000 | 1.00000 | 17.185 | True |
| int8 | 3.692e-02 | 0.99483 | 0.99806 | 4.752 | True |

## Comparison against the existing baseline

| metric | baseline (INT8 semantic student) | this run |
|---|---|---|
| intents | 11 | 57 |
| model size | 0.236 MB | see parity table |
| contextual accuracy | 0.9062 | 0.5379 |
| OOD rejection | 0.3438 | 0.8667 |

> These are **not** like-for-like. The baseline numbers are for an 11-intent problem; this run is 57 intents on a leakage-controlled split with harder held-out suites. A 57-way problem with a 35x class imbalance is a strictly harder task, so the baseline's accuracy is not a ceiling this run failed to reach — it is a different measurement. The only honest comparison is to re-run the baseline model against these same suites.
