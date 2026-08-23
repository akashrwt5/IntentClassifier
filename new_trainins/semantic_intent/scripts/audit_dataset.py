"""Phase 1 — Dataset audit.

Reads data/raw/en.csv and produces:
  reports/dataset_audit.md
  data/cleaned/dataset_issues.csv
No model / no network required.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent))
from common import leakage_key, normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "en.csv"
OUT_REPORT = ROOT / "reports" / "dataset_audit.md"
OUT_ISSUES = ROOT / "data" / "cleaned" / "dataset_issues.csv"
OUT_JSON = ROOT / "reports" / "dataset_audit.json"

FALLBACK_LABEL = "Default Fallback Intent"


def main() -> None:
    df = pd.read_csv(RAW).dropna(subset=["text", "intent"])
    df["text"] = df["text"].astype(str)
    df["intent"] = df["intent"].astype(str).str.strip()
    df["norm"] = df["text"].map(normalize)
    df["key"] = df["text"].map(leakage_key)
    df = df[df["norm"].str.len() > 0].reset_index(drop=True)

    issues: list[dict] = []
    stats: dict = {}

    # --- 3.1 intent distribution ---------------------------------------
    vc = df["intent"].value_counts()
    stats["n_rows"] = int(len(df))
    stats["n_intents"] = int(len(vc))
    stats["max_class"] = int(vc.max())
    stats["min_class"] = int(vc.min())
    stats["imbalance_ratio"] = round(float(vc.max() / vc.min()), 2)
    stats["median_class"] = int(vc.median())
    stats["classes_under_60"] = [c for c, n in vc.items() if n < 60]
    stats["distribution"] = {k: int(v) for k, v in vc.items()}

    # --- 3.2 duplicates -------------------------------------------------
    exact = df[df.duplicated(subset=["text"], keep=False)]
    stats["exact_dupe_rows"] = int(len(exact))
    stats["exact_dupe_groups"] = int(exact["text"].nunique())

    norm_dupe = df[df.duplicated(subset=["norm"], keep=False)]
    stats["normalized_dupe_rows"] = int(len(norm_dupe))
    stats["normalized_dupe_groups"] = int(norm_dupe["norm"].nunique())

    # conflicting labels: same normalized text, >1 intent
    conf = df.groupby("norm")["intent"].nunique().reset_index(name="n_intent")
    conflicting = set(conf.loc[conf["n_intent"] > 1, "norm"])
    stats["conflicting_texts"] = int(len(conflicting))
    conflict_rows = df[df["norm"].isin(conflicting)].sort_values("norm")
    for _, r in conflict_rows.iterrows():
        labels = sorted(df.loc[df["norm"] == r["norm"], "intent"].unique())
        issues.append(
            dict(
                issue="conflicting_label",
                text=r["text"],
                intent=r["intent"],
                detail="|".join(labels),
            )
        )

    # duplicate rows that agree on label -> drop-able
    dup_same = df[df.duplicated(subset=["norm", "intent"], keep="first")]
    for _, r in dup_same.iterrows():
        issues.append(
            dict(
                issue="duplicate_same_label",
                text=r["text"],
                intent=r["intent"],
                detail="normalized duplicate",
            )
        )

    # --- 3.3 near duplicates across DIFFERENT intents --------------------
    # This is the dangerous kind: near-identical text, different label.
    near_conflicts = []
    by_key = defaultdict(set)
    for k, i in zip(df["key"], df["intent"]):
        by_key[k].add(i)
    cross_key = {k for k, v in by_key.items() if len(v) > 1}
    stats["leakage_key_collisions_cross_intent"] = int(len(cross_key))
    for k in sorted(cross_key):
        sub = df[df["key"] == k]
        labels = sorted(sub["intent"].unique())
        near_conflicts.append(dict(key=k, labels=labels, examples=sub["text"].head(4).tolist()))
        for _, r in sub.iterrows():
            issues.append(
                dict(
                    issue="near_dup_cross_intent",
                    text=r["text"],
                    intent=r["intent"],
                    detail="|".join(labels),
                )
            )
    stats["near_conflict_examples"] = near_conflicts[:40]

    # --- fuzzy near-duplicates within the corpus (sampled) --------------
    # Full O(n^2) on ~10k is fine with rapidfuzz cdist on a sample per intent.
    fuzzy_pairs = 0
    for intent, sub in df.groupby("intent"):
        texts = sub["norm"].tolist()
        if len(texts) < 2:
            continue
        sample = texts[:400]
        m = process.cdist(sample, sample, scorer=fuzz.token_sort_ratio, workers=-1)
        iu = np.triu_indices(len(sample), k=1)
        fuzzy_pairs += int((m[iu] >= 92).sum())
    stats["fuzzy_near_dupe_pairs_within_intent"] = fuzzy_pairs

    # --- 3.4 vocabulary --------------------------------------------------
    global_df = Counter()
    per_intent_tokens: dict[str, Counter] = {}
    for intent, sub in df.groupby("intent"):
        c = Counter()
        for t in sub["norm"]:
            c.update(set(t.split()))
        per_intent_tokens[intent] = c
        global_df.update(c.keys())

    n_intents = len(per_intent_tokens)
    distinctive: dict[str, list] = {}
    for intent, c in per_intent_tokens.items():
        n = max(1, int(vc[intent]))
        scored = []
        for w, cnt in c.items():
            if cnt < 3:
                continue
            p_in = cnt / n
            in_k_intents = sum(1 for cc in per_intent_tokens.values() if cc.get(w, 0) > 0)
            idf = np.log(n_intents / in_k_intents)
            scored.append((w, round(p_in * idf, 3), cnt, in_k_intents))
        scored.sort(key=lambda x: -x[1])
        distinctive[intent] = scored[:8]

    # single-token shortcut risk: token that appears in >=60% of an intent's
    # rows and in <=2 intents overall
    shortcut_risk = []
    for intent, c in per_intent_tokens.items():
        n = max(1, int(vc[intent]))
        for w, cnt in c.items():
            in_k = sum(1 for cc in per_intent_tokens.values() if cc.get(w, 0) > 0)
            if cnt / n >= 0.6 and in_k <= 2 and w not in {"the", "a", "to"}:
                shortcut_risk.append(
                    dict(
                        intent=intent, token=w, coverage=round(cnt / n, 3), intents_containing=in_k
                    )
                )
    shortcut_risk.sort(key=lambda d: -d["coverage"])
    stats["shortcut_risk_tokens"] = shortcut_risk[:40]

    # --- length profile ---------------------------------------------------
    lens = df["norm"].str.split().str.len()
    stats["len_words"] = dict(
        p10=int(lens.quantile(0.10)),
        p50=int(lens.quantile(0.50)),
        p90=int(lens.quantile(0.90)),
        p99=int(lens.quantile(0.99)),
        max=int(lens.max()),
    )
    stats["short_rows_under_3_words"] = int((lens < 3).sum())

    # --- fallback class ----------------------------------------------------
    stats["fallback_rows"] = int((df["intent"] == FALLBACK_LABEL).sum())

    # --- write outputs ------------------------------------------------------
    OUT_ISSUES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(issues).to_csv(OUT_ISSUES, index=False)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2))

    md = render_markdown(stats, distinctive, vc)
    OUT_REPORT.write_text(md)
    print(
        f"rows={stats['n_rows']} intents={stats['n_intents']} "
        f"issues={len(issues)} -> {OUT_REPORT}"
    )


def render_markdown(stats, distinctive, vc) -> str:
    L: list[str] = []
    a = L.append
    a("# Dataset Audit — en.csv\n")
    a("Phase 1 of the robustness plan. No model involved; pure data inspection.\n")

    a("## 1. Scale and balance\n")
    a("| metric | value |")
    a("|---|---|")
    a(f"| rows (after empty-text drop) | {stats['n_rows']} |")
    a(f"| intents | {stats['n_intents']} |")
    a(f"| largest class | {stats['max_class']} |")
    a(f"| median class | {stats['median_class']} |")
    a(f"| smallest class | {stats['min_class']} |")
    a(f"| imbalance ratio (max/min) | {stats['imbalance_ratio']}x |")
    a(f"| classes with <60 examples | {len(stats['classes_under_60'])} |")
    a("")
    a(
        "Classes under 60 examples (these carry the most risk of a weak, "
        "over-confident decision boundary):\n"
    )
    a("```text")
    for c in stats["classes_under_60"]:
        a(f"{c:40s} {stats['distribution'][c]}")
    a("```\n")

    a("### Full distribution\n")
    a("| intent | n | share |")
    a("|---|---|---|")
    total = stats["n_rows"]
    for k, n in stats["distribution"].items():
        a(f"| `{k}` | {n} | {100*n/total:.1f}% |")
    a("")

    a("## 2. Duplicates\n")
    a("| check | value |")
    a("|---|---|")
    a(f"| exact duplicate rows | {stats['exact_dupe_rows']} |")
    a(f"| exact duplicate groups | {stats['exact_dupe_groups']} |")
    a(f"| normalized duplicate rows | {stats['normalized_dupe_rows']} |")
    a(f"| normalized duplicate groups | {stats['normalized_dupe_groups']} |")
    a(
        f"| fuzzy near-dupe pairs within an intent (>=92 token_sort) | {stats['fuzzy_near_dupe_pairs_within_intent']} |"
    )
    a("")

    a("## 3. Label consistency\n")
    a(f"- Normalized texts carrying more than one label: **{stats['conflicting_texts']}**")
    a(
        f"- Leakage-key collisions across different intents: **{stats['leakage_key_collisions_cross_intent']}**\n"
    )
    if stats["near_conflict_examples"]:
        a("Examples of cross-intent collisions (same content words, different label):\n")
        a("```text")
        for ex in stats["near_conflict_examples"][:20]:
            a(f"labels: {', '.join(ex['labels'])}")
            for t in ex["examples"]:
                a(f"  - {t}")
        a("```\n")

    a("## 4. Vocabulary shortcut risk\n")
    a(
        "Tokens covering >=60% of one intent's rows while appearing in at most "
        "2 intents overall. These are exactly the shortcuts the plan warns about "
        "(Section 5 / Section 23): the model can learn the token instead of the "
        "meaning, and then negation or context flips break it.\n"
    )
    if stats["shortcut_risk_tokens"]:
        a("| intent | token | coverage | intents containing |")
        a("|---|---|---|---|")
        for d in stats["shortcut_risk_tokens"][:25]:
            a(
                f"| `{d['intent']}` | `{d['token']}` | {d['coverage']:.0%} | {d['intents_containing']} |"
            )
    else:
        a("_None found._")
    a("")

    a("### Most distinctive tokens per intent\n")
    a("```text")
    for intent in list(distinctive)[:60]:
        toks = ", ".join(f"{w}({s})" for w, s, _, _ in distinctive[intent][:6])
        a(f"{intent:36s} {toks}")
    a("```\n")

    a("## 5. Length profile\n")
    lw = stats["len_words"]
    a(
        f"p10={lw['p10']} · p50={lw['p50']} · p90={lw['p90']} · p99={lw['p99']} · max={lw['max']} words"
    )
    a(f"\nRows shorter than 3 words: {stats['short_rows_under_3_words']}\n")

    a("## 6. Fallback / OOD class already present\n")
    a(
        f"`{FALLBACK_LABEL}` has {stats['fallback_rows']} rows. This is a supervised "
        "reject class, which is useful, but it is not a substitute for OOD "
        "evaluation: it only covers unsupported phrasings someone thought of in "
        "advance. Phase 9 still needs a held-out OOD suite, including near-OOD.\n"
    )
    return "\n".join(L)


if __name__ == "__main__":
    main()
