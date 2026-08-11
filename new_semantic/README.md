# new_semantic

Semantic (Stage 3) ka saara kaam yahan hota hai. **English pehle, phir multilingual.**

Purana `semantic_project/` (200+ files, v1–v7) research archive hai — usko touch
mat karo. Yahan ek hi pipeline hai, ek hi config, aur har script leak guard ke
saath aati hai.

---

## Abhi ki state

| | |
|---|---|
| Training data | `data/en/train.csv` — **23,989 rows / 57 intents** (= `train_merged.csv`, byte-for-byte) |
| Locked test | `data/eval/locked_test_en.csv` — 1,686 rows |
| Stress test | `data/eval/stress_test_en.csv` — 565 rows |
| OOD test | `data/eval/ood_test_en.csv` — 403 rows |
| Leak status | locked / stress / OOD — sab training se **disjoint** ✅ |

### Honest baseline (numpy TF-IDF + LogReg, koi torch nahi)

| Metric | Value |
|---|---:|
| Locked accuracy | **0.8007** |
| Locked macro recall | 0.8178 |
| Stress accuracy | 0.6637 |
| OOD fallback rate | **0.2184** |

Ye **floor** hai. Neural student ko isse aaram se jeetna chahiye, warna woh
justify nahi hota.

> Purana `98.87%` ka number **invalid** tha — locked test ki 100% rows `train.csv`
> mein thi. Detail: `docs/semantic-tiny-model-plan.md` §10.1.

---

## Sabse badi problem: OOD rejection

Baseline sirf **21.8%** OOD queries reject karta hai — 403 mein se 315 nikal
jaati hain, mostly `Help_Home`, `reminders.add`, `Help_Customize` mein.

Threshold sweep dikhata hai ki trade-off kitna crooked hai:

| threshold | OOD reject | in-scope acc |
|---:|---:|---:|
| 0.0 | 0.218 | 0.801 |
| 0.3 | 0.948 | 0.565 |
| 0.5 | 0.978 | 0.423 |
| 0.8 | 1.000 | 0.144 |

Yaani 95% OOD reject karne ke liye in-scope accuracy 80% se 56% pe girti hai.
**Threshold se ye theek nahi hoga — data aur model se hoga.** Isiliye neural
student ka asli kaam accuracy nahi, *separation* hai.

---

## Pipeline

```
train_merged.csv
   │  scripts/prepare_data.py          leak guard + dedup
   ▼
data/en/train.csv  (23,989)
   │  scripts/train_en.py              E5 teacher -> tiny student
   ▼
models/en/student_<tag>.pt  (~0.75 MB)
   │  scripts/evaluate.py              locked + stress + OOD + ship bar
   ▼
reports/eval_<tag>.json
```

### Commands

```bash
cd new_semantic

# 1. data prep (already done — re-run if train_merged.csv badle)
python scripts/prepare_data.py --protect training

# 2. dependency-free floor (chalti hai kahin bhi, ~2 min)
python scripts/baseline_numpy.py

# 3. asli student (torch chahiye — aapki machine pe)
pip install torch sentence-transformers scikit-learn
python scripts/train_en.py --tag v1

# 4. evaluate
python scripts/evaluate.py --tag v1
python scripts/evaluate.py --tag v1 --threshold 0.55

# ablations
python scripts/train_en.py --tag no_weights --no-class-weights
python scripts/train_en.py --tag no_distill --no-distill
```

---

## Design decisions (aur kyun)

### 1. Tokenizer hi identity hai

```python
re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
```

Punctuation poori tarah discard hoti hai — `"volume up"` aur `"volume up?"`
model ke liye **ek hi input** hain. Isliye dedup aur leak guard raw text pe
nahi, **tokenizer ke view pe** chalte hain. Raw text pe check karne se 55
duplicates aur 23 locked-test leaks bach nikli thin.

### 2. Data uncapped hai, class weights se balance hota hai

`train.csv` deliberately imbalanced hai (55×): 11 intents ke paas ~1,850 rows,
23 intents ke paas 50 se kam. Rows delete karke balance karna data waste karna
hai. Iske bajaye:

```python
w[c] = n_total / (n_classes * n[c])       # Help_DemoMode 12.36, Cmd.VolumeMute 0.22
loss = CE_WEIGHT * weighted_CE + KD_WEIGHT * KL(student ‖ teacher, T=2)
```

Weights `reports/prepare_data.json` mein pre-computed hain.

### 3. Macro recall dekho, accuracy nahi

23 intents 50 rows se neeche hain. Accuracy unhe chhupa deti hai — model bade
classes pe achha karke high accuracy dikha sakta hai. `evaluate.py` dono report
karta hai aur gap 5% se zyada ho toh warn karta hai.

### 4. Eval sets vs training — `--protect`

Ek phrase dono jagah nahi ho sakta. `--protect training` (default) training
corpus bachata hai aur eval set se colliding rows nikaalta hai. Isi wajah se
stress test 595 → 565 hua.

---

## Aage kya (English)

```
[ ] student train karo (train_en.py) — baseline 0.80 ko beat karo
[ ] class-weights on/off ablation
[ ] OOD data 403 -> 1000+ (yahi sabse bada blocker hai)
[ ] 23 chhoti classes ko 150+ rows tak lao
[ ] threshold calibrate karo
[ ] ONNX export (static batch=1, seq=24) + parity
[ ] INT8 + parity
[ ] packages/runtime/nlu_engine/semantic.py mein wire karo
```

Multilingual tab shuru hoga jab English ship bar clear ho jaaye. Plan:
`docs/semantic-tiny-model-plan.md` §6.

---

## Files

| Path | Kya hai |
|---|---|
| `config.py` | Single source of truth — architecture, hyperparams, paths, ship bar |
| `scripts/common.py` | Tokenizer, vocab, leak guard, class weights |
| `scripts/prepare_data.py` | train_merged.csv → leak-free train.csv |
| `scripts/train_en.py` | E5 teacher → tiny student distillation |
| `scripts/evaluate.py` | Locked + stress + OOD + ship bar |
| `scripts/baseline_numpy.py` | Dependency-free floor baseline |
| `scripts/build_merged_train.py` | xlsx + balanced CSV merge (upstream) |
| `reports/` | Har run ka JSON summary |
