"""Viability check for cheaper abstention signals than threshold tuning.

Tests three scores that need NO new model, only the logits the engine already
computes:
  (a) max logit  (energy-style OOD score)
  (b) rank of Default Fallback Intent in the logit ordering
  (c) softmax prob assigned to Default Fallback Intent

Question is only 'does the signal separate?', not 'what threshold?' — the
parameter must be fit on a dev split, never on this holdout.
"""
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path("/home/user/IntentClassifier")
sys.path.insert(0, str(REPO / "packages" / "runtime"))
sys.path.insert(0, str(REPO / "packages" / "buildtime"))

from nlu_engine import NLUEngine  # noqa: E402
from nlu_training.wrong_action_harness import is_actionable, is_read_only  # noqa: E402

rows = list(csv.DictReader(open(REPO / "datasets/en/holdout_honest.csv",
                                encoding="utf-8-sig", newline="")))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ENG = NLUEngine(model_name="en", language="en", semantic_enabled=False)
labels = list(ENG.classifier.labels)
OOS = labels.index("Default Fallback Intent")
T = ENG.classifier.temperature

rec = []
for i, r in enumerate(rows):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ENG.handle(f"s-{i}", r["text"])
    lg = ENG.classifier.backend.tfidf_logits(r["text"])
    if isinstance(lg, dict):
        lg = np.array([lg[l] for l in labels], dtype=float)
    order = np.argsort(lg)[::-1]
    z = lg / T
    p = np.exp(z - z.max())
    p = p / p.sum()
    rec.append({
        "truth": r["intent"], "type": res.type, "intent": res.intent or "",
        "conf": float(res.confidence or 0),
        "maxlogit": float(lg.max()),
        "oos_rank": int(np.where(order == OOS)[0][0]),
        "oos_prob": float(p[OOS]),
    })

wrong = [t for t in rec if t["type"] == "FULFILL" and t["intent"] != t["truth"]
         and is_actionable(t["intent"]) and not is_read_only(t["intent"])]
good = [t for t in rec if t["type"] == "FULFILL" and t["intent"] == t["truth"]]
print(f"wrong={len(wrong)} good={len(good)}\n")


def sweep(name, key, cmp_lo=True, grid=()):
    w = np.array([t[key] for t in wrong])
    g = np.array([t[key] for t in good])
    print(f"=== {name} ===")
    print(f"  wrong: median {np.median(w):.3f}   good: median {np.median(g):.3f}")
    for v in grid:
        if cmp_lo:
            bw, bg = int((w < v).sum()), int((g < v).sum())
            op = "<"
        else:
            bw, bg = int((w > v).sum()), int((g > v).sum())
            op = ">"
        print(f"   block if {key}{op}{v:<7} -> catches {bw}/{len(w)} wrong, "
              f"costs {bg}/{len(g)} correct ({bg/len(g)*100:.1f}%)")
    print()


sweep("(a) max logit  (energy-style)", "maxlogit", True,
      (0.0, 0.5, 1.0, 1.5, 2.0, 2.5))
sweep("(b) OOS class rank (block if OOS ranks high)", "oos_rank", True,
      (1, 2, 3, 5, 10))
sweep("(c) OOS class probability", "oos_prob", False,
      (0.005, 0.01, 0.02, 0.05, 0.10))
