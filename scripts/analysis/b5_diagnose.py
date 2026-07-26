"""Diagnostics for the four wrong-action remediation options.

MEASUREMENT ONLY — nothing is written, no threshold is changed. The goal is to
find out which options are even viable before any of them is designed, and to
cost the ones that are.
"""
import csv
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/home/user/IntentClassifier")
sys.path.insert(0, str(REPO / "packages" / "runtime"))
sys.path.insert(0, str(REPO / "packages" / "buildtime"))

from nlu_engine import NLUEngine  # noqa: E402
from nlu_training.wrong_action_harness import is_actionable, is_read_only  # noqa: E402

SCHEMA = json.loads((REPO / "content" / "nlu_schema.json").read_text())
GATED = set(SCHEMA["uncertain_confirm"]["intents"])

with open(REPO / "datasets/en/holdout_honest.csv", encoding="utf-8-sig", newline="") as f:
    ROWS = list(csv.DictReader(f))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ENG = NLUEngine(model_name="en", language="en", semantic_enabled=False)

# ---- one pass, capture everything -------------------------------------------
turns = []
for i, row in enumerate(ROWS):
    text, truth = row["text"], row["intent"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ENG.handle(f"diag-{i}", text)
    # raw model view of the same utterance, independent of the engine's stage
    logits = ENG.classifier.backend.tfidf_logits(text)
    if isinstance(logits, dict):
        logits = np.array([logits[l] for l in ENG.classifier.labels], dtype=float)
    order = np.argsort(logits)[::-1]
    margin = float(logits[order[0]] - logits[order[1]])
    turns.append({
        "text": text, "truth": truth, "type": r.type, "intent": r.intent or "",
        "conf": float(r.confidence or 0.0), "stage": ENG.classifier.last_stage,
        "tier": ENG.classifier.last_keyword_tier,
        "margin": margin,
        "model_top": ENG.classifier.labels[order[0]],
    })


def is_wrong_action(t):
    return (t["type"] == "FULFILL" and t["intent"] != t["truth"]
            and is_actionable(t["intent"]) and not is_read_only(t["intent"]))


def is_good_fire(t):
    return t["type"] == "FULFILL" and t["intent"] == t["truth"]


wrong = [t for t in turns if is_wrong_action(t)]
good = [t for t in turns if is_good_fire(t)]
print(f"turns={len(turns)}  wrong_actions={len(wrong)}  correct_fulfills={len(good)}")

# ---- Q1: does the top1-top2 logit margin separate the failures? --------------
print("\n=== Q1  logit-margin separation (is an abstention margin viable?) ===")
wm = np.array([t["margin"] for t in wrong])
gm = np.array([t["margin"] for t in good])
print(f"wrong actions  margin: median {np.median(wm):.2f}  "
      f"p25 {np.percentile(wm,25):.2f}  min {wm.min():.2f}  max {wm.max():.2f}")
print(f"correct fires  margin: median {np.median(gm):.2f}  "
      f"p25 {np.percentile(gm,25):.2f}  min {gm.min():.2f}  max {gm.max():.2f}")
for thr in (0.5, 1.0, 1.5, 2.0, 3.0):
    print(f"  margin<{thr:<4} would block {int((wm<thr).sum())}/{len(wm)} wrong "
          f"and {int((gm<thr).sum())}/{len(gm)} correct "
          f"({(gm<thr).mean()*100:.1f}% of good traffic)")

# ---- Q2: how much of the residue is already in the confirm list? -------------
print("\n=== Q2  confirm-gate coverage of the failures ===")
in_gate = [t for t in wrong if t["intent"] in GATED]
print(f"{len(in_gate)}/{len(wrong)} wrong actions fired an intent ALREADY in "
      f"uncertain_confirm, but at conf >= below_confidence(0.80)")
print("  not in the gate list:",
      Counter(t["intent"] for t in wrong if t["intent"] not in GATED))

print("\n  cost of raising below_confidence (gated intents only):")
for b in (0.80, 0.90, 0.95, 0.99, 1.01):
    blocked = sum(1 for t in wrong if t["intent"] in GATED and t["conf"] < b)
    extra = sum(1 for t in good if t["intent"] in GATED and t["conf"] < b)
    tot_gated_good = sum(1 for t in good if t["intent"] in GATED)
    print(f"   below={b:<5} catches {blocked}/{len(wrong)} wrong;  "
          f"{extra}/{tot_gated_good} correct fires become confirmations "
          f"({extra/max(tot_gated_good,1)*100:.1f}% of gated good traffic)")

# ---- Q3: what would adding the missing intents to the gate cost? -------------
print("\n=== Q3  extending the gate list to every state-changing intent ===")
allsc = {t["intent"] for t in turns
         if is_actionable(t["intent"]) and not is_read_only(t["intent"])}
for b in (0.90, 0.95, 1.01):
    blocked = sum(1 for t in wrong if t["conf"] < b)
    extra = sum(1 for t in good
                if t["intent"] in allsc and t["conf"] < b)
    tot = sum(1 for t in good if t["intent"] in allsc)
    print(f"   all {len(allsc)} state-changing, below={b:<5} catches "
          f"{blocked}/{len(wrong)} wrong;  {extra}/{tot} correct fires "
          f"become confirmations ({extra/max(tot,1)*100:.1f}%)")

# ---- Q4: fire-threshold sweep (the blunt instrument) ------------------------
print("\n=== Q4  raising the fire threshold alone ===")
for thr in (0.70, 0.80, 0.90, 0.95, 0.99):
    blocked = sum(1 for t in wrong if t["conf"] < thr)
    lost = sum(1 for t in good if t["conf"] < thr)
    print(f"   threshold={thr:<5} blocks {blocked}/{len(wrong)} wrong  "
          f"but DEFLECTS {lost}/{len(good)} correct ({lost/len(good)*100:.1f}%)")

# ---- Q5: which stage produced the failures? ---------------------------------
print("\n=== Q5  stage attribution ===")
print("  wrong actions by stage:", Counter((t["stage"], t["tier"]) for t in wrong))
print("  correct fires by stage:", Counter(t["stage"] for t in good))

# ---- Q6: OOS data volume ----------------------------------------------------
print("\n=== Q6  OOS coverage in training data ===")
tr = list(csv.DictReader(open(REPO / "datasets/en/train.csv",
                              encoding="utf-8-sig", newline="")))
c = Counter(r["intent"] for r in tr)
print(f"  train rows {len(tr)}; sys.oos.fallback = {c['sys.oos.fallback']} "
      f"({c['sys.oos.fallback']/len(tr)*100:.1f}%)")
print(f"  mean rows per in-scope intent: "
      f"{np.mean([v for k,v in c.items() if k!='sys.oos.fallback']):.0f}")
oos_hold = [t for t in turns if t["truth"] == "sys.oos.fallback"]
caught = sum(1 for t in oos_hold if t["type"] == "FALLBACK"
             or not is_actionable(t["intent"]))
print(f"  OOS turns in holdout: {len(oos_hold)}; correctly deflected: {caught} "
      f"({caught/len(oos_hold)*100:.1f}%)")
print(f"  unused OOS pools: oos.csv=156  oos_2.csv=367 rows (not in train.csv?)")
tr_texts = {r["text"].lower().strip() for r in tr}
for name in ("oos.csv", "oos_2.csv"):
    p = REPO / "datasets/en" / name
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))
    new = [r for r in rows if r["text"].lower().strip() not in tr_texts]
    print(f"    {name}: {len(rows)} rows, {len(new)} NOT already in train.csv")
