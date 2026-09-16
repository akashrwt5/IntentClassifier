#!/usr/bin/env python3
"""Fit the plural-folding table by ABLATION, not by a hand-chosen rule.

WHY THIS EXISTS
---------------
The TF-IDF featurizer has no stemmer (`train.py`: a stock `TfidfVectorizer`),
so `program` and `programs` occupy two unrelated vocabulary slots. When one form
dominates a label and the other dominates a different label, PLURALITY ITSELF
becomes the intent predictor. Measured on en: `program` runs 623 rows under
`Cmd.MemoryChange` against 23 under `Help_ChangingMemories`, while `programs`
runs 6 against 44 — which is why "change programs" classified as a help question
and "change program" as a command.

Folding the two forms together fixes that pair. The hard part is deciding WHICH
pairs to fold, and two obvious answers were tried and MEASURED WRONG:

  * A ratio rule ("fold when the plural is the minority form"). Arbitrary. At
    0.30 it selected 54 pairs, of which ablation later showed `batteries`,
    `restaurants`, `goals` and `settings` all HURT. End to end that table cost
    5 holdout rows and raised out-of-scope actions from 6 to 9.

  * Distributional similarity (Jensen-Shannon divergence between the two forms'
    label distributions). It gives the OPPOSITE of the right answer:
    `memories` scores 0.49 and `programs` 0.61 — "used differently" — but that
    divergence IS the bug being fixed, not evidence against fixing it. Mean-
    while `aids` scores 0.10 ("safe to fold") and folding it costs 2 rows.

Neither proxy can distinguish "two different concepts" from "one concept with
skewed sampling". Only the model can, so ask it: fold the pair, refit, and
measure. That is this module.

METHOD
------
1. CANDIDATES — a crude suffix rule (-s / -es / -ies), then a vocabulary guard:
   fold only onto a stem the corpus actually contains. The guard is what stops
   `this -> thi`, `tinnitus -> tinnitu` and `does -> doe`; without it the rule
   invents a non-word 149 times out of 250.
2. ABLATION — for each surviving candidate, fit the pipeline with ONLY that one
   fold applied and score it on a held-out split, averaged over several seeds.
   Keep the pair when it beats the unfolded baseline by `--min-gain` rows.
3. Selection uses train.csv splits ONLY. The permanent holdout is never read
   here — tuning against it is Review-F5 blocker B9.

The fits are independent, so the result is a first-order (non-interacting)
estimate. That is deliberate: a greedy or exhaustive search over 2^n subsets
would overfit the splits long before it paid for the compute.

COST — one fit is a few seconds and the default is 2 seeds per candidate, so a
full run is minutes, not hours. It is a build step, not a per-turn cost:
nothing here runs at inference, which only applies the finished map.

    python -m nlu_training.fit_lemmas --lang en            # report only
    python -m nlu_training.fit_lemmas --lang en --write    # write lemmas.json

`--cache` makes a run resumable: scores already computed are reused, so an
interrupted run continues instead of restarting.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_SEEDS = (42, 7)
DEFAULT_MIN_COUNT = 3        # a rarer plural cannot move a held-out split
DEFAULT_MIN_GAIN = 1.0       # rows, on the held-out split, averaged over seeds
_TOKEN_RE = re.compile(r"\b\w\w+\b")          # sklearn's default token pattern


def singularise(word: str) -> str:
    """The suffix rule. Deliberately crude — the vocabulary guard filters it."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _pipeline():
    # MUST mirror train.py. A different featurizer here would rank candidates
    # for a model that is never built.
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0)),
    ])


def _score(texts, labels, seeds):
    from sklearn.model_selection import train_test_split
    accs, n = [], 0
    for seed in seeds:
        x_tr, x_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.2, random_state=seed, stratify=labels)
        model = _pipeline()
        model.fit(x_tr, y_tr)
        accs.append(float(np.mean(model.predict(x_te) == np.array(y_te))))
        n = len(y_te)
    return float(np.mean(accs)), n


def candidates(texts, min_count: int):
    counts = Counter(w for t in texts for w in _TOKEN_RE.findall(t))
    pairs = [(w, singularise(w)) for w in counts
             if singularise(w) != w and singularise(w) in counts
             and counts[w] >= min_count]
    return sorted(pairs, key=lambda p: -counts[p[0]]), counts


def fit(lang: str = "en", seeds=DEFAULT_SEEDS, min_count: int = DEFAULT_MIN_COUNT,
        min_gain: float = DEFAULT_MIN_GAIN, cache: Path | None = None,
        budget: int | None = None, verbose: bool = True) -> dict:
    from nlu_engine.text_norm import normalize_text

    pack = BASE_DIR / "language_packs" / lang
    with (pack / "train.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Unfolded base forms: the candidate search and every ablation start here.
    texts = [normalize_text(r["text"], lemmas={}) for r in rows]
    labels = [r["intent"].strip() for r in rows]

    pairs, counts = candidates(texts, min_count)
    scores: dict = json.loads(cache.read_text()) if cache and cache.exists() else {}

    if "__BASELINE__" not in scores:
        acc, n = _score(texts, labels, seeds)
        scores["__BASELINE__"] = {"acc": acc, "n": n}
        if cache:
            cache.write_text(json.dumps(scores), encoding="utf-8")
    base_acc = scores["__BASELINE__"]["acc"]
    n_eval = scores["__BASELINE__"]["n"]

    todo = [p for p in pairs if p[0] not in scores]
    if budget is not None:
        todo = todo[:budget]
    for word, stem in todo:
        rx = re.compile(r"\b" + re.escape(word) + r"\b")
        acc, _ = _score([rx.sub(stem, t) for t in texts], labels, seeds)
        scores[word] = {"stem": stem, "acc": acc,
                        "delta_rows": round((acc - base_acc) * n_eval, 2),
                        "count": counts[word], "stem_count": counts[stem]}
        if cache:
            cache.write_text(json.dumps(scores), encoding="utf-8")
        if verbose:
            print(f"  {word:16} -> {stem:16} {scores[word]['delta_rows']:+6.1f} rows")

    remaining = len([p for p in pairs if p[0] not in scores])
    kept = {w: v["stem"] for w, v in scores.items()
            if w != "__BASELINE__" and v["delta_rows"] >= min_gain}
    return {"table": dict(sorted(kept.items())), "scores": scores,
            "baseline_acc": base_acc, "n_eval": n_eval,
            "n_candidates": len(pairs), "remaining": remaining}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="nlu_training.fit_lemmas", description=__doc__)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                    help="skip a plural rarer than this (default 3)")
    ap.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN,
                    help="keep a fold only if it wins at least this many "
                         "held-out rows (default 1.0)")
    ap.add_argument("--budget", type=int, default=None,
                    help="score at most N candidates this run (use with --cache "
                         "to split a long run across invocations)")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--write", action="store_true",
                    help="write language_packs/<lang>/lemmas.json")
    args = ap.parse_args(argv)

    res = fit(args.lang, tuple(args.seeds), args.min_count, args.min_gain,
              args.cache, args.budget)

    print(f"\n  baseline held-out accuracy {res['baseline_acc']:.4f} "
          f"on {res['n_eval']} rows ({len(args.seeds)} seeds)")
    print(f"  candidates {res['n_candidates']}, unscored {res['remaining']}")
    ranked = sorted(((v["delta_rows"], w, v["stem"]) for w, v in res["scores"].items()
                     if w != "__BASELINE__"), reverse=True)
    print(f"\n  {'delta':>7}  {'plural':16} {'singular':16}")
    for d, w, s in ranked:
        if abs(d) >= args.min_gain:
            print(f"  {d:>7.1f}  {w:16} {s:16}{'   KEEP' if d >= args.min_gain else ''}")
    print(f"\n  KEEPING {len(res['table'])}: {json.dumps(res['table'])}")

    if res["remaining"]:
        print(f"\n  {res['remaining']} candidates still unscored — rerun with "
              f"--cache to finish before trusting --write.")
    if args.write:
        out = BASE_DIR / "language_packs" / args.lang / "lemmas.json"
        out.write_text(json.dumps(res["table"], indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"  wrote {out.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
