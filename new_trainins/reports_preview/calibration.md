# Confidence Calibration and the Operating Point

## What 'calibrated' means here

> 90% confidence must not mean 'the model feels 90% sure'. It must mean that among predictions in the 90% band, roughly 90% are correct.

The table below is the direct test of that claim on held-out data.


## Temperature scaling

- fitted temperature **T = 0.6409** (validation split only)
- validation ECE **0.1760 -> 0.0246**

T < 1 means the raw scores were **under-confident** and had to be sharpened.

## Reliability diagram (held-out test)

| confidence bin | n | mean confidence | actual accuracy | gap |
|---|---|---|---|---|
| 0.0–0.1 | 5 | 0.090 | 0.000 | +0.090 |
| 0.1–0.2 | 99 | 0.156 | 0.222 | -0.066 |
| 0.2–0.3 | 137 | 0.254 | 0.299 | -0.045 |
| 0.3–0.4 | 156 | 0.350 | 0.423 | -0.073 |
| 0.4–0.5 | 136 | 0.445 | 0.471 | -0.025 |
| 0.5–0.6 | 132 | 0.550 | 0.583 | -0.034 |
| 0.6–0.7 | 124 | 0.649 | 0.613 | +0.036 |
| 0.7–0.8 | 128 | 0.750 | 0.766 | -0.016 |
| 0.8–0.9 | 133 | 0.851 | 0.820 | +0.031 |
| 0.9–1.0 | 463 | 0.976 | 0.974 | +0.002 |

Test ECE **0.0291**, MCE 0.0733, Brier 0.4604.

## Operating point

Chosen on validation for precision first: on a hearing aid, executing the wrong command is worse than asking the user to repeat.

- confidence threshold **0.985**
- top1-top2 margin threshold **0.0**
- validation precision among accepted **1.0000** at coverage **0.1916**
- target of 97% precision met: **False**

### Coverage / precision trade-off

| threshold | coverage | precision among accepted | n |
|---|---|---|---|
| 0.30 | 0.844 | 0.7156 | 1273 |
| 0.35 | 0.792 | 0.7387 | 1194 |
| 0.40 | 0.747 | 0.7655 | 1126 |
| 0.45 | 0.695 | 0.7958 | 1048 |
| 0.50 | 0.649 | 0.8200 | 978 |
| 0.55 | 0.602 | 0.8436 | 908 |
| 0.60 | 0.563 | 0.8622 | 849 |
| 0.65 | 0.526 | 0.8865 | 793 |
| 0.70 | 0.490 | 0.8958 | 739 |
| 0.75 | 0.458 | 0.9204 | 691 |
| 0.80 | 0.403 | 0.9473 | 607 |
| 0.85 | 0.361 | 0.9633 | 545 |
| 0.90 | 0.314 | 0.9789 | 473 |
| 0.95 | 0.264 | 0.9899 | 398 |

## Why margin is checked as well as confidence

```text
increase = 0.51
decrease = 0.48
margin   = 0.03
```

Top-1 confidence alone would call that a decision. The margin says it is a coin flip between two opposite commands, which is exactly the case where a hearing aid should ask again.
