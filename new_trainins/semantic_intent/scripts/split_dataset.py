"""Phase 3 + Phase 10 — cleaning, leakage-free grouping, and splitting.

Leakage control is the point of this script. Two sentences that are near
paraphrases must land in the SAME split, otherwise the test set is measuring
memorization. We build groups by:
  1. exact normalized text
  2. leakage_key (sorted content stems)
  3. fuzzy token_sort_ratio >= 92 within an intent
and take connected components over those relations.

Outputs:
  data/cleaned/dataset_clean.csv
  data/train.csv  data/validation.csv  data/test.csv
  data/stt_test.csv  data/hard_negative_test.csv  data/ood_test.csv
  data/minimal_pair_test.csv  data/negation_test.csv  data/contextual_test.csv
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent))
from build_challenge_sets import stt_corrupt  # noqa: E402
from common import leakage_key, normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15
FUZZ_THRESHOLD = 92


class DSU:
    def __init__(self) -> None:
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_groups(df: pd.DataFrame) -> pd.Series:
    dsu = DSU()
    for i in df.index:
        dsu.find(i)
    # relation 1+2: shared normalized text or leakage key
    for col in ("norm", "key"):
        buckets = defaultdict(list)
        for i, v in df[col].items():
            buckets[v].append(i)
        for idxs in buckets.values():
            for j in idxs[1:]:
                dsu.union(idxs[0], j)
    # relation 3: fuzzy within intent
    for _, sub in df.groupby("intent"):
        idxs = sub.index.tolist()
        texts = sub["norm"].tolist()
        if len(texts) < 2:
            continue
        m = process.cdist(texts, texts, scorer=fuzz.token_sort_ratio, workers=-1)
        ii, jj = np.where(np.triu(m, k=1) >= FUZZ_THRESHOLD)
        for a, b in zip(ii, jj):
            dsu.union(idxs[a], idxs[b])
    return df.index.to_series().map(dsu.find)


def split_groups(df: pd.DataFrame, rng: random.Random) -> pd.Series:
    """Stratified-by-intent split at GROUP level."""
    assign = pd.Series(index=df.index, dtype=object)
    for intent, sub in df.groupby("intent"):
        groups = list(sub.groupby("group").size().items())
        rng.shuffle(groups)
        n_total = len(sub)
        want_val = max(1, int(round(n_total * VAL_FRAC)))
        want_test = max(1, int(round(n_total * TEST_FRAC)))
        got_val = got_test = 0
        for gid, gsize in groups:
            if got_test < want_test:
                part, got_test = "test", got_test + gsize
            elif got_val < want_val:
                part, got_val = "validation", got_val + gsize
            else:
                part = "train"
            assign.loc[sub.index[sub["group"] == gid]] = part
        # never leave an intent without train rows
        if (assign.loc[sub.index] == "train").sum() == 0:
            biggest = max(groups, key=lambda t: t[1])[0]
            assign.loc[sub.index[sub["group"] == biggest]] = "train"
    return assign


def main() -> None:
    rng = random.Random(SEED)
    df = pd.read_csv(DATA / "raw" / "en.csv").dropna(subset=["text", "intent"])
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df["norm"] = df["text"].map(normalize)
    df = df[df["norm"].str.len() > 0]

    before = len(df)
    # drop normalized duplicates that agree on label (keep first)
    df = df.drop_duplicates(subset=["norm", "intent"], keep="first")
    # drop normalized texts carrying more than one label (unresolvable)
    multi = df.groupby("norm")["intent"].nunique()
    ambiguous = set(multi[multi > 1].index)
    df = df[~df["norm"].isin(ambiguous)]
    df = df.reset_index(drop=True)
    df["key"] = df["text"].map(leakage_key)
    print(f"clean: {before} -> {len(df)} rows "
          f"(dropped {before - len(df)}; ambiguous texts removed: {len(ambiguous)})")

    df["group"] = build_groups(df)
    n_groups = df["group"].nunique()
    print(f"leakage groups: {n_groups} (avg {len(df)/n_groups:.2f} rows/group, "
          f"largest {df['group'].value_counts().max()})")

    df["split"] = split_groups(df, rng)
    (DATA / "cleaned").mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA / "cleaned" / "dataset_clean.csv", index=False)

    for part in ("train", "validation", "test"):
        sub = df[df["split"] == part][["text", "intent"]]
        sub.to_csv(DATA / f"{part}.csv", index=False)
        print(f"{part:11s} {len(sub):5d} rows, {sub['intent'].nunique()} intents")

    # --- leakage verification -------------------------------------------
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    va = df[df["split"] == "validation"]
    overlap_key = len(set(te["key"]) & set(tr["key"]))
    overlap_grp = len(set(te["group"]) & set(tr["group"]))
    print(f"VERIFY test/train shared leakage keys = {overlap_key} "
          f"(expect 0); shared groups = {overlap_grp} (expect 0)")
    assert overlap_key == 0 and overlap_grp == 0

    # --- challenge suites -------------------------------------------------
    ch = DATA / "challenge"
    pd.read_csv(ch / "hard_negatives.csv").to_csv(DATA / "hard_negative_test.csv", index=False)
    pd.read_csv(ch / "ood.csv").to_csv(DATA / "ood_test.csv", index=False)
    pd.read_csv(ch / "minimal_pairs.csv").to_csv(DATA / "minimal_pair_test.csv", index=False)
    pd.read_csv(ch / "negation.csv").to_csv(DATA / "negation_test.csv", index=False)
    pd.read_csv(ch / "contextual.csv").to_csv(DATA / "contextual_test.csv", index=False)
    pd.read_csv(ch / "accessories.csv").to_csv(DATA / "accessories_test.csv", index=False)

    # --- STT suite: corrupt the held-out test split ----------------------
    srng = random.Random(SEED + 1)
    rows = []
    for _, r in te.iterrows():
        corrupted, ops = stt_corrupt(r["norm"], srng)
        if corrupted == r["norm"]:
            continue
        rows.append(dict(text=corrupted, intent=r["intent"],
                         clean_text=r["norm"], ops=ops))
    stt = pd.DataFrame(rows)
    # Corruption can coincidentally collapse a test sentence onto a training
    # sentence's leakage key. Those rows would no longer be held out, so drop them.
    train_keys = set(tr["key"])
    n_before = len(stt)
    stt = stt[~stt["text"].map(leakage_key).isin(train_keys)].reset_index(drop=True)
    stt.to_csv(DATA / "stt_test.csv", index=False)
    print(f"stt_test    {len(stt):5d} rows built from held-out test split "
          f"({n_before - len(stt)} dropped for colliding with train after corruption)")

    counts = df.groupby(["intent", "split"]).size().unstack(fill_value=0)
    counts.to_csv(ROOT / "reports" / "split_counts.csv")
    weak = counts[(counts.get("test", 0) < 5)]
    if len(weak):
        print(f"NOTE: {len(weak)} intents have <5 test rows: {list(weak.index)}")


if __name__ == "__main__":
    main()
