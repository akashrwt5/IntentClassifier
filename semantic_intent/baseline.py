"""
TF-IDF baseline, measured under exactly the same protocol as the semantic model.

This exists to keep the project honest. A 24 MB encoder has to earn its size
against a 1.6 MB bag-of-words model, and on a template-generated corpus it
very often does not win on accuracy. Run this before claiming it does.

    python -m semantic_intent.baseline --data datasets/balanced_intents_final.xlsx

Reports accuracy on the same grouped split, the same hard paraphrase set, the
same antonym pairs, and — stratified by how much of each test utterance's
vocabulary the training set actually contains — where the two models diverge.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from . import data as data_mod
from .eval_sets import ANTONYM_PAIRS, HARD_PARAPHRASES, OUT_OF_SCOPE

# Function words carry no intent signal; excluding them makes the overlap
# statistic reflect real vocabulary novelty rather than grammar.
STOPWORDS = set(
    "the a an is it its this that to of in on for me my i you your can could "
    "would please and be are was am here now so at do how what with".split()
)


def build(train_texts, train_labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=15.0)),
        ]
    )
    return pipe.fit(train_texts, train_labels)


def vocabulary_overlap(text: str, vocab: set[str]) -> float:
    """Fraction of a sentence's content words that appear in training."""
    words = [w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOPWORDS]
    return sum(w in vocab for w in words) / max(len(words), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0, help="must match training seed")
    ap.add_argument(
        "--compare-semantic",
        action="store_true",
        help="also load models/semantic_intent.onnx and diff",
    )
    ap.add_argument(
        "--augment",
        type=int,
        default=0,
        help="apply the same contrastive augmentation to the baseline",
    )
    args = ap.parse_args()

    df = data_mod.grouped_split(data_mod.load(args.data), seed=args.seed)
    if args.augment:
        from .augment import augment_training_split

        df = augment_training_split(df, per_intent=args.augment, seed=args.seed)

    fit = df[df.split.isin(["train", "dev"])]
    test = df[df.split == "test"]
    model = build(fit.text, fit.intent)
    vocab = {w for t in fit.text for w in re.findall(r"[a-z']+", t.lower())}

    sem = None
    if args.compare_semantic:
        from .predict import SemanticIntentClassifier

        sem = SemanticIntentClassifier()

    def semantic(texts):
        return [sem.predict(t).intent for t in texts]

    # ---------------------------------------------------------- accuracy
    acc = model.score(test.text, test.intent)
    print(f"grouped test accuracy   TF-IDF={acc:.4f}  n={len(test)}")
    if sem:
        s_acc = np.mean(np.array(semantic(test.text.tolist())) == test.intent.values)
        print(f"                        semantic={s_acc:.4f}")

    # -------------------------------------------------- hard paraphrases
    texts = [t for t, _ in HARD_PARAPHRASES]
    gold = [g for _, g in HARD_PARAPHRASES]
    p_tf = model.predict(texts)
    conf_tf = model.predict_proba(texts).max(1)
    p_sem = semantic(texts) if sem else None
    conf_sem = [sem.predict(t).confidence for t in texts] if sem else None

    print(
        f"\nhard paraphrases        TF-IDF={sum(a == b for a, b in zip(p_tf, gold))}"
        f"/{len(gold)}",
        end="",
    )
    if sem:
        print(f"  semantic={sum(a == b for a, b in zip(p_sem, gold))}/{len(gold)}")
    else:
        print()

    print("\nstratified by training-vocabulary overlap:")
    overlaps = np.array([vocabulary_overlap(t, vocab) for t in texts])
    for lo, hi, name in [(0.0, 0.7, "low  (<70% content words seen)"), (0.7, 1.01, "high (>=70%)")]:
        m = (overlaps >= lo) & (overlaps < hi)
        if not m.any():
            continue
        n_tf = sum(a == b for a, b, k in zip(p_tf, gold, m) if k)
        line = f"  {name:34s} n={m.sum():2d}  TF-IDF={n_tf}/{m.sum()}"
        if sem:
            n_s = sum(a == b for a, b, k in zip(p_sem, gold, m) if k)
            line += f"  semantic={n_s}/{m.sum()}"
        print(line)

    # ------------------------------------------------------- confidence
    print(
        f"\nconfidence on hard set  TF-IDF mean={conf_tf.mean():.2f}  "
        f"below 0.6: {(conf_tf < 0.6).sum()}/{len(texts)}"
    )
    if sem:
        cs = np.array(conf_sem)
        print(
            f"                        semantic mean={cs.mean():.2f}  "
            f"below 0.6: {(cs < 0.6).sum()}/{len(texts)}"
        )
    print(
        "  (a correct answer below the gate threshold is still routed to"
        " fallback — this is where the encoder pays for itself)"
    )

    # ---------------------------------------------------- antonym pairs
    ok = 0
    print("\nantonym pairs:")
    for a, ga, b, gb in ANTONYM_PAIRS:
        pa, pb = model.predict([a, b])
        good = pa == ga and pb == gb
        ok += good
        print(f"  {'OK  ' if good else 'FAIL'} {a[:36]:38s} -> {pa}")
        if not good:
            print(f"       {b[:36]:38s} -> {pb}")
    print(f"  TF-IDF: {ok}/{len(ANTONYM_PAIRS)}")

    # ----------------------------------------------------- out of scope
    oos = [t for t in OUT_OF_SCOPE if t.strip()]
    c_oos = model.predict_proba(oos).max(1)
    print(
        f"\nout-of-scope: TF-IDF has no rejection mechanism. "
        f"mean max-prob={c_oos.mean():.2f}, "
        f"{(c_oos > 0.7).sum()}/{len(oos)} would pass a 0.7 gate."
    )
    if sem:
        rej = sum(not sem.predict(t).accepted for t in oos)
        print(f"              semantic rejects {rej}/{len(oos)} via the prototype gate.")


if __name__ == "__main__":
    main()
