# Decision Record — English tiny semantic student

**Date:** 2026-08-10
**Status:** Accepted
**Chosen model:** `unkaug` config, **seed 1**, operating threshold **0.40**

---

## Decision

Ship `models/en/student_unkaug_s1.pt` with a confidence gate at **0.40**.

| Metric (mean of 3 seeds) | Value |
|---|---:|
| OOD reject @ 0.40 | **0.9504** |
| In-scope accuracy @ 0.40 | **0.8333** |
| Locked accuracy (raw) | 0.8891 ± 0.0036 |
| Locked macro recall | 0.8635 ± 0.0045 |
| Stress accuracy | 0.7422 ± 0.0037 |
| Size | 0.79 MB fp32 |

Config: `--unk-aug 0.04` (921 synthetic fallback rows), class weights ON,
word tokenizer, E5-small-v2 teacher, `MAX_LEN 24`, d64 / 2-layer / nhead 4.

**Seed 1 was chosen because it is the MEDIAN of the three by locked accuracy.**
Picking seed 7 (the best) would have re-introduced the exact selection bias this
whole exercise was about.

---

## Why, in one line

At every threshold from 0.30 to 0.50, `unkaug` gives **more OOD rejection AND
more in-scope accuracy** than the baseline — it is not a trade-off, it dominates.
It is also ~10× more reproducible.

| thr | base OODrej / in-scope | unkaug OODrej / in-scope |
|---:|---:|---:|
| 0.30 | 0.7494 / 0.8578 | **0.9313 / 0.8614** |
| 0.40 | 0.8379 / 0.8296 | **0.9504 / 0.8333** |

---

## What this cost

`unkaug` loses **5.5 points of stress accuracy** (0.7422 vs 0.7976). Stress
measures novel phrasings of in-scope commands — precisely what semantic rescue
exists to catch. This is a real, statistically confirmed regression and it is
accepted deliberately: in a hearing-aid product, executing the wrong command is
worse than failing to act. It should be revisited when more data exists.

**This model does NOT pass the pre-registered gate** (`stress >= 0.78`). Neither
did any other config. The gate was shipped past on an explicit product judgement,
not because the number improved.

---

## The reproducibility finding (the most important result here)

Identical config, three seeds:

| config | locked sd | stress sd | **OOD raw sd** |
|---|---:|---:|---:|
| base | 0.0195 | 0.0252 | **0.2317** |
| unkaug | 0.0036 | 0.0037 | **0.0224** |
| both | 0.0100 | 0.0000 | 0.0108 |

The baseline's OOD fallback rate ranged **0.2134 – 0.6551 across seeds with no
config change**. A 44-point spread.

**Consequence: every single-seed comparison in this project before this point was
uninterpretable.** Conclusions drawn from v1–v5 about which intervention "helped"
or "hurt" were mostly noise. They have been discarded.

Rule going forward: **no config is compared on one seed.** Use
`scripts/run_seeds.py`. A gap smaller than 2× the pooled sd is not evidence.

---

## What survived the noise floor

Only two effects exceeded 2× pooled sd:

1. UNK augmentation **raises OOD rejection** (+0.465 raw). Real.
2. UNK augmentation **lowers stress accuracy** (−0.055). Real.

Everything else — fallback weight floor, fallback oversampling, subword
tokenizer, the `--unk-robust` counter-example — showed **no significant effect**.

---

## Hypotheses that were tested and failed

| Hypothesis | Result |
|---|---|
| Fallback class is starved (4% of data) → upweight it | No significant effect |
| Oversampling real fallback rows helps | No significant effect |
| `[UNK]` collapse is the bottleneck → subword tokenizer removes it | UNK went 22.5% → 0% on OOD, but **no improvement**; stress got worse |
| A counter-example (`--unk-robust`) recovers stress while keeping OOD | No significant effect on either |

The subword result is worth remembering: removing `[UNK]` entirely did not help.
The 64-dim / 2-layer student appears unable to recover word meaning from pieces
with only 24k sentences, and loses the direct word→meaning mapping that worked.

---

## Known weaknesses of this decision

- **OOD eval set is 403 rows.** Every OOD number here has a wide confidence
  interval. This is the single biggest limitation.
- **23 of 57 intents have under 50 training rows.** `Help_AppSettings` has 9
  locked-test rows; its recall figure means very little.
- **Stress test covers only 11 of 57 intents.** It is not a general
  novel-phrasing measure.
- **921 synthetic rows** are in the training data (`synthetic_text: true`).
  Repo manifests track this field — keep it accurate downstream.
- The locked test's phrasing diversity has never been audited.

---

## Export result (2026-08-10)

| artifact | size | parity vs PyTorch |
|---|---:|---|
| **FP32 ONNX** | **0.953 MB** | max delta 4.3e-06, **0** argmax mismatches, **0** gate disagreements (2,654 utterances) |
| INT8 ONNX | 0.349 MB | max delta 2.3e-01, 5 argmax mismatches, 9 gate disagreements |

### CORRECTION (2026-08-10) — the size figures here were wrong

This section first recorded **0.166 MB** and concluded *"INT8 is rejected — it is
2x LARGER than FP32."* **Both statements were false.**

`torch.onnx.export` writes initializers to a sidecar (`student_unkaug_s1.onnx.data`,
0.787 MB) once they pass a size threshold. Only the graph lands in the `.onnx`.
The reported 0.166 MB was the graph alone; the real artifact is **0.953 MB**. The
INT8 comparison then pitted a graph-only file against a self-contained one:

| | claimed | actual |
|---|---:|---:|
| FP32 | 0.166 MB | **0.953 MB** |
| INT8 | 0.349 MB | 0.349 MB |
| verdict | "INT8 2x larger" | **INT8 is 2.7x SMALLER** |

Shipping the `.onnx` without its `.data` sidecar would have deployed a model with
no weights. Parity could not catch it — both sides of that check load the same
weights in the same process.

`export_onnx.py` now folds external data back into a single file and aborts if
any sidecar remains. INT8 is still not shipped, but for the reason that always
held: **5 argmax flips and 9 gate disagreements**. The size argument was backwards.

Shipping artifact: `models/en/student_unkaug_s1.onnx`, **0.953 MB self-contained**,
static shape (1, 24), no dynamic axes, logits out, gate applied at runtime at 0.40.

## Cascade correction (2026-08-10) — supersedes the operating point above

Everything above evaluated the student as a STANDALONE classifier on 100% of
inputs. Production never asks that question: the student is Stage 3 and only
sees turns where Stage 2 (TF-IDF, gate 0.7) is unsure.

Measured in that position, the shipped policy — student replaces Stage 2's
answer — makes the pipeline WORSE on every axis:

| | Stage 2 alone | + student (`replace`) |
|---|---:|---:|
| stress | 0.8442 | 0.7080 |
| locked | 0.9573 | 0.9282 |
| OOD reject | — | — |

On the stress handover subset Stage 2 scores 0.6923 and the student 0.3632. The
policy was discarding a 0.69 signal to substitute a 0.36 one. That is a pipeline
bug, not a model deficiency.

### New policy: `dual@s3=0.40, s2=0.30`

```
answer  = argmax of whichever model is more confident
accept  = (student_conf >= 0.40) OR (stage2_conf >= 0.30)
else    = fallback
```

Selected on a **dev half** of the eval sets (stratified, seed 20260810, saved at
`data/eval/dev_test_split.json`), averaged over 3 seeds, by max harmonic mean of
stress and OOD. TEST half read once afterwards.

| policy | test stress | test OOD |
|---|---:|---:|
| s2_only (no Stage 3) | 0.8333 | 0.4527 |
| replace (previously shipped) | 0.6856 | 0.9088 |
| **dual@s3=0.40,s2=0.30** | **0.8298 ± 0.0094** | **0.8590 ± 0.0057** |

vs the previously shipped policy: **+14.4 stress, −5.0 OOD**.
vs having no Stage 3 at all: −0.4 stress (within noise), **+40.6 OOD**.

**The dev→test drop was −0.023 for the selected policy and −0.022 to −0.023 for
every other policy including the bar.** A uniform drop is a dev/test difficulty
difference, not selection bias — had we overfit dev, the winner would have
dropped further than the others. It did not.

### A failed attempt worth recording

The first dual gate used `stage2_conf >= 0.7` as the second clause. That is
vacuous: the handover subset is DEFINED by Stage 2 scoring below 0.7, so the
clause was always false and the policy silently degenerated into `replace`.
Stage 2 needs a secondary, lower threshold on this subset. This cost one full
round of experiments.

### What the policy does NOT fix

No policy reached stress >= 0.85 AND OOD >= 0.90. The frontier is real. Closing
it needs the model to be trained on the HANDOVER distribution — the rows where
Stage 2 is unsure — rather than on the full distribution it currently sees.

## Next

```
[ ] Implement dual@s3=0.40,s2=0.30 in the engine (behaviour change - needs approval)
[ ] Train the student on the handover distribution (the real remaining fix)
[ ] Wire into packages/runtime/nlu_engine/semantic.py
[ ] OOD eval set 403 -> 1000+   <- highest-value data work
[ ] 23 small classes -> 150+ rows each
[ ] Re-run run_seeds.py on the new data; revisit the stress regression
```
