# Data Correction Plan — Sprint 3 Semantic Head Retraining

## Context

The semantic classification head (MiniLM-L6-v2 + logistic regression) is underperforming on the holdout set (0.60 accuracy) due to:

1. **Class imbalance**: MemoryChange (1821 rows) vs Help_Battery (18 rows) — 100× range
2. **Near-duplicate spam**: 70% of MemoryChange in 3-gram buckets ("can you change X" ×185 variants)
3. **Undersized OOS class**: 156 out-of-scope phrases vs 6,800 in-scope (2.2% of corpus)
4. **Overfitting hyperparameter**: C=10.0 in high-dimensional (384-dim) space
5. **Blind holdout measurement**: semantic_holdout_100.csv covers only 10 of 60 intents (50 completely unmeasured)

## Changes Implemented in This Branch

### ✅ Completed (Lever B + C Code Changes)

**File: `scripts/train_semantic_head.py`**

1. **Lever A: Per-class capping (lines 117–128)**
   - Cap in-scope intents at MAX_PER_INTENT=250 (down from raw counts)
   - MemoryChange: 1821 → ~250 (removes ~70% near-duplicate spam)
   - Help_Battery: 18 → 18 (preserves tiny tail classes)
   - Use deterministic keep-newest + shuffle (same pattern as train.py)

2. **Lever B: Hyperparameter tuning (lines 171–173, 211)**
   - Reduce C from 10.0 → 1.5 (prevents overfitting of tail classes)
   - Regularization strength increases (less overfitting, more bias)
   - Applied in both held-out split training and final retrain

3. **Lever C: Per-class metrics (lines 181–193)**
   - Per-intent accuracy showing what each class achieves
   - Macro-F1 score (unweighted per-class average)
   - Exposes tail-class regressions (e.g., Help_Battery 0% accuracy)

### ⏳ Pending (Lever A Data Changes — User Responsibility)

**Gap Analysis** (run `python scripts/scaffold_data_correction.py`):
- **Tail classes needing diversity**: 22 intents < 40 rows → need 252 total new paraphrases
- **OOS expansion**: 156 → 400–600 phrases → need 244–344 more
- **Holdout rebuild**: 10 intents → 60 intents → need 50 new intent coverages + 5–10 rows each

### 📋 Data Files to Update

#### 1. `data/01_source_base_training_data.csv` — Add diversity to tail classes

**Target**: All intents ≥ 40 rows (currently 22 intents < 40)
**Need**: 252 new paraphrases across tail intents

**Example tail intents**:
- Help_Battery (18 rows) → add 22 paraphrases
- Help_EdgeMode (19 rows) → add 21 paraphrases
- Cmd.ActivityAerobics (21 rows) → add 19 paraphrases
- ... (22 total intents, each needing 1–22 new examples)

**How to add**: Append new rows with format:
```
text,intent
my battery is draining fast,Help_Battery
how long does battery last,Help_Battery
need a new battery,Help_Battery
...
```

**Validation**: Review for accuracy; ensure phrases are genuine paraphrases, not permutations.

#### 2. `data/semantic_oos.csv` — Expand hard negatives

**Current**: 156 phrases  
**Target**: 400–600 phrases  
**Need**: 244–344 hard-negative phrases (queries that look in-domain but are OOS)

**Suggested categories**:
- Cooking/food: "how do I make pasta", "pasta recipe", "cooking tips"
- Weather: "what's the weather", "will it rain", "forecast"
- General knowledge: "who won world cup", "capital of france", "tell me about X"
- Unit conversion: "convert 10 dollars to euros", "10 km in miles"
- Product Q&A (near boundary): "how to clean my device", "battery specs", "device lifespan"
- Entertainment: "tell me a joke", "sing a song", "dance moves"

**Template provided**: `data/semantic_oos_expansion_template.csv` (15 examples)

**How to add**: Append rows with format:
```
text,intent
how do I make pasta,Default Fallback Intent
what's the weather today,Default Fallback Intent
...
```

**Validation**: Ensure all phrases are genuinely OOS (not in-domain) and diverse.

#### 3. `data/semantic_holdout_100.csv` → Rebuild for 60 intents

**Current**: 100 rows across 10 intents (MemoryChange, Volume controls, reminders, etc.)  
**Target**: 300–600 rows across all 60 intents  
**Missing**: 50 intents completely unmeasured (Help_Battery, Cmd.ActivityRun, Help_Tinnitus, etc.)

**Template provided**: `data/semantic_holdout_expansion_template.csv` (140 examples covering all 60)

**How to rebuild**:
1. Keep existing 100 rows from current holdout
2. Add 5–10 rows per intent for the 50 unmeasured intents
3. For well-represented intents (MemoryChange, reminders), add edge cases and paraphrases
4. Tag `difficulty`: "easy" (in training data) vs "hard" (paraphrase/edge case)

**Format**:
```
utterance,expected_intent,difficulty
mute the audio,Cmd.VolumeMute,easy
silence please,Cmd.VolumeMute,hard
...
```

**Validation**: Ensure coverage across all 60 intents; mix easy (training data) and hard (paraphrases).

---

## Expected Impact

### Before Retraining
- Holdout accuracy: 0.60 (measured on 10 intents only; 50 intents blind)
- Tail-class performance: unknown (zero per-class metrics reported)
- Training samples: 6,787 (heavy imbalance + spam)

### After Changes
- **Holdout accuracy**: 0.65–0.72 (on honest all-60-intent coverage, not just 10)
- **Tail-class performance**: Visible per-intent metrics expose regressions
- **Training samples**: ~4,600 (post-cap); ~4,900 (post-cap + OOS expansion)
- **Class balance**: MemoryChange 250 vs Help_Battery 18 → better weighting effect
- **OOS rejection**: Larger, more diverse OOS set improves decision boundary

### Validation Signals
1. Per-class accuracy should flatten (tail classes stop at 0%)
2. Macro-F1 should improve (less 0% classes dragging it down)
3. Real holdout (60-intent) should show honest baseline for future improvements

---

## Timeline

### Phase 1: Data Preparation (User)
1. Add 252 paraphrases to 01_source_base_training_data.csv (tail class diversity)
2. Expand semantic_oos.csv by 244–344 phrases (OOS hard negatives)
3. Rebuild semantic_holdout_100.csv for 60-intent coverage
4. Review all new data for accuracy

### Phase 2: Retrain (Automated)
```bash
python scripts/train_semantic_head.py
```

**Output**:
- Per-class accuracy report (shows which intents improved/regressed)
- Macro-F1 vs aggregate accuracy comparison
- Rejection curve (threshold tuning guidance)
- Updated models/semantic_head.npz and models/semantic_head.json

### Phase 3: Validation
1. Run `python scripts/test_semantic.py` to check if paraphrase rescues work better
2. Verify no regressions in existing passing cases
3. Review per-class metrics for unexpected failures

---

## Files in This Branch

**Modified**:
- `scripts/train_semantic_head.py` — Lever A/B/C code changes

**New (helpers)**:
- `scripts/scaffold_data_correction.py` — Gap analysis tool
- `data/semantic_oos_expansion_template.csv` — 15 example OOS phrases
- `data/semantic_holdout_expansion_template.csv` — 140 example holdout rows (all 60 intents)
- `DATA_CORRECTION_PLAN.md` — This file

**To be modified**:
- `data/01_source_base_training_data.csv` — Add tail-class diversity
- `data/semantic_oos.csv` — Expand hard negatives
- `data/semantic_holdout_100.csv` — Rebuild for 60-intent coverage

---

## Next Steps (When User Is Ready)

1. **Run gap analysis**:
   ```bash
   python scripts/scaffold_data_correction.py
   ```

2. **Update data files** (use templates as reference):
   - 01_source_base_training_data.csv: Add 252 paraphrases
   - semantic_oos.csv: Add 244+ hard negatives
   - semantic_holdout_100.csv: Rebuild for 60 intents

3. **Retrain**:
   ```bash
   python scripts/train_semantic_head.py
   ```

4. **Validate**:
   ```bash
   python scripts/test_semantic.py
   python scripts/test_holdout.py  # if it exists
   ```

5. **Commit** (when satisfied with results):
   ```bash
   git add data/01_source_base_training_data.csv data/semantic_oos.csv data/semantic_holdout_100.csv
   git commit -m "data: expand semantic training with tail-class diversity and full-coverage holdout"
   git push -u origin feature/Adv2/AddSemanticUnderstanding-4-adding-coreML-supports-DataCorrection
   ```

---

## Safety Gates

- ✅ **Leakage guard**: train_semantic_head.py checks no OOS phrase is in holdout
- ✅ **Accuracy gate**: scripts/train.py MIN_TEST_ACCURACY=0.85 (TF-IDF)
- ✅ **Manifest checksum**: All model artifacts tracked in models/manifest.json
- ⚠️ **Data review**: All new training/holdout data should be reviewed by domain expert before committing

---

## References

- **Intent distribution**: `python scripts/scaffold_data_correction.py` → shows per-intent gaps
- **Training logic**: scripts/train_semantic_head.py (lines 102–182)
- **Semantic head**: scripts/nlu/semantic.py (61 classes: 60 intents + "Default Fallback Intent")
- **Test suite**: scripts/test_semantic.py (validates phase 3 semantic rescue cases)
- **Previous analysis**: See conversation history for class imbalance and near-duplicate spam audit
