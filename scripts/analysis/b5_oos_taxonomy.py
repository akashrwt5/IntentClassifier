"""How much of the OOS class is actually in-scope speech wearing an OOS label?

Nearest in-scope neighbour by TF-IDF cosine for every Default Fallback Intent row.
A high-similarity pair means the same words carry both labels, which is not a
hard example — it is a contradictory training signal sitting on the decision
boundary of the intent it resembles.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path("/home/user/IntentClassifier")

train = list(csv.DictReader(open(REPO / "datasets/en/train.csv",
                                 encoding="utf-8-sig", newline="")))
oos = [r for r in train if r["intent"] == "Default Fallback Intent"]
ins = [r for r in train if r["intent"] != "Default Fallback Intent"]
print(f"train: {len(train)} rows -> {len(oos)} OOS / {len(ins)} in-scope")

vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(
    [r["text"] for r in train])
A = vec.transform([r["text"] for r in oos])
B = vec.transform([r["text"] for r in ins])
S = (A @ B.T).toarray()
best = S.argmax(axis=1)
sim = S.max(axis=1)

print("\n=== OOS rows by similarity to their nearest IN-SCOPE neighbour ===")
for lo, hi in ((0.9, 1.01), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.0, 0.6)):
    m = (sim >= lo) & (sim < hi)
    print(f"  cosine {lo:.1f}-{hi:.1f}: {m.sum():5d} rows "
          f"({m.sum()/len(oos)*100:5.1f}%)")

print("\n=== the most contradictory pairs (cosine >= 0.75) ===")
idx = np.argsort(sim)[::-1]
shown = 0
for i in idx:
    if sim[i] < 0.75 or shown >= 25:
        break
    print(f"  {sim[i]:.2f}  OOS {oos[i]['text']!r}")
    print(f"        vs {ins[best[i]]['intent']:<26} {ins[best[i]]['text']!r}")
    shown += 1

print("\n=== which in-scope intents are most shadowed by OOS rows? ===")
c = Counter(ins[best[i]]["intent"] for i in range(len(oos)) if sim[i] >= 0.7)
for lab, n in c.most_common(12):
    print(f"  {n:4d}  {lab}")

# same test for the 355 candidate additions
print("\n" + "=" * 70)
print("=== the 355 candidate OOS additions (oos_2.csv) ===")
pool = list(csv.DictReader(open(REPO / "datasets/en/oos_2.csv",
                                encoding="utf-8-sig", newline="")))
P = vec.transform([r["text"] for r in pool])
SP = (P @ B.T).toarray()
pb, ps = SP.argmax(axis=1), SP.max(axis=1)
for lo, hi in ((0.7, 1.01), (0.5, 0.7), (0.0, 0.5)):
    m = (ps >= lo) & (ps < hi)
    print(f"  cosine {lo:.1f}-{hi:.1f}: {m.sum():4d} / {len(pool)} rows")
print("\n  most contradictory candidates:")
for i in np.argsort(ps)[::-1][:12]:
    print(f"  {ps[i]:.2f}  {pool[i]['text']!r}")
    print(f"        vs {ins[pb[i]]['intent']:<26} {ins[pb[i]]['text']!r}")
