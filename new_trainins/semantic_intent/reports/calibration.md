# Confidence Calibration and the Operating Point

## What 'calibrated' means here

> 90% confidence must not mean 'the model feels 90% sure'. It must mean that among predictions in the 90% band, roughly 90% are correct.

The table below is the direct test of that claim on held-out data.


## Temperature scaling

- fitted temperature **T = 2.1483** (validation split only)
- validation ECE **0.0781 -> 0.0201**

T > 1 means the raw scores were **over-confident** and had to be softened.

## Reliability diagram (held-out test)

| confidence bin | n | mean confidence | actual accuracy | gap |
|---|---|---|---|---|
| 0.1–0.2 | 5 | 0.149 | 0.000 | +0.149 |
| 0.2–0.3 | 13 | 0.260 | 0.308 | -0.048 |
| 0.3–0.4 | 14 | 0.343 | 0.214 | +0.129 |
| 0.4–0.5 | 26 | 0.466 | 0.500 | -0.034 |
| 0.5–0.6 | 42 | 0.553 | 0.429 | +0.124 |
| 0.6–0.7 | 35 | 0.653 | 0.743 | -0.090 |
| 0.7–0.8 | 46 | 0.751 | 0.804 | -0.054 |
| 0.8–0.9 | 103 | 0.854 | 0.835 | +0.019 |
| 0.9–1.0 | 1229 | 0.977 | 0.973 | +0.004 |

Test ECE **0.0146**, MCE 0.1291, Brier 0.1282.

## Operating point

Chosen on validation for precision first: on a hearing aid, executing the wrong command is worse than asking the user to repeat.

- confidence threshold **0.87**
- top1-top2 margin threshold **0.0**
- validation precision among accepted **0.9766** at coverage **0.8368**
- target of 97% precision met: **True**

### Coverage / precision trade-off

| threshold | coverage | precision among accepted | n |
|---|---|---|---|
| 0.30 | 0.993 | 0.9011 | 1497 |
| 0.35 | 0.989 | 0.9041 | 1491 |
| 0.40 | 0.979 | 0.9106 | 1476 |
| 0.45 | 0.971 | 0.9147 | 1465 |
| 0.50 | 0.962 | 0.9201 | 1451 |
| 0.55 | 0.946 | 0.9306 | 1426 |
| 0.60 | 0.932 | 0.9381 | 1406 |
| 0.65 | 0.923 | 0.9418 | 1392 |
| 0.70 | 0.905 | 0.9479 | 1364 |
| 0.75 | 0.885 | 0.9543 | 1335 |
| 0.80 | 0.865 | 0.9594 | 1304 |
| 0.85 | 0.830 | 0.9688 | 1251 |
| 0.90 | 0.773 | 0.9794 | 1165 |
| 0.95 | 0.678 | 0.9824 | 1022 |

## Why margin is checked as well as confidence

```text
increase = 0.51
decrease = 0.48
margin   = 0.03
```

Top-1 confidence alone would call that a decision. The margin says it is a coin flip between two opposite commands, which is exactly the case where a hearing aid should ask again.
