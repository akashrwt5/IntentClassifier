#!/usr/bin/env python3
"""
Robust BIO Tagging Prototype for IntentClassifier (Slot Extraction).

Feature building, preprocessing and BIO decoding all come from
``nlu_engine.slot_tagger`` -- this script owns the CORPUS and the TRAINING only.
Never redefine a feature here; change :data:`nlu_engine.slot_tagger.FEATURE_SPEC`
so the runtime, the exporter and the mobile ports move with it.

Trained and hardened against 7 real-world failure categories:
1. Spelling mistakes (e.g., medcine, docotr, tomorow, perscription)
2. Slang & colloquialisms (e.g., gimme a ping, drop a reminder, hit me up)
3. Speech-to-text (ASR) errors (e.g., remind her, remind me two, at for pm)
4. Missing punctuation (e.g., run-ons like please thanks, dont forget)
5. Contractions (e.g., don't, can't, i've gotta, i'd like)
6. Conversational fillers (e.g., um, uhhh, you know, well basically)
7. Word variations (e.g., schedule, log an alert, keep track of)

Usage:
    python scripts/bio_tagger_prototype.py
    python scripts/bio_tagger_prototype.py "bro set reminder to take medicine tomorrow 5 pm"
"""

import csv
import random
import sys
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "runtime"))

from nlu_engine import NLUEngine  # noqa: E402
from nlu_engine.slot_tagger import (  # noqa: E402
    FEATURE_SPEC,
    extract_slot_title,
    extract_title_span,
    extract_token_features,
    prepare,
    robust_preprocess,
    tokenize,
)
from nlu_langpack import load_pack  # noqa: E402

# Backwards compatibility aliases
extract_features = extract_token_features

# Augmentation draws on this, so a model is reproducible from its seed alone.
SEED = 42
HOLDOUT_FRACTION = 0.15

BEGIN = FEATURE_SPEC["labels"]["begin"]
INSIDE = FEATURE_SPEC["labels"]["inside"]
OUTSIDE = FEATURE_SPEC["labels"]["outside"]

TYPO_MAP = {
    "doctor": "docotr",
    "medicine": "medcine",
    "prescription": "perscription",
    "groceries": "groseries",
    "tomorrow": "tomorow",
    "reminder": "remndr",
    "appointment": "apointment",
}

FILLERS = [
    "um like",
    "you know uhhh",
    "well basically",
    "yo bro",
    "listen buddy",
    "can someone please",
    "hey",
    "thanks",
    "gimme a ping to",
    "drop a reminder to",
]

RUN_ONS = ["dont forget", "please thanks", "thanks"]


def best_span(tokens: list[str], title_tokens: list[str]) -> tuple[int, int] | None:
    """
    Locate the title inside the text, tolerating a title the rule engine mangled.

    The engine builds a title by DELETING date words, so "call mom on christmas"
    comes back as "call mom christmas" -- a string that no longer occurs in the
    sentence. Matching it whole therefore fails, and 53 of 707 sentences used to
    become all-"O" because of it, teaching the model that a normal reminder has no
    title. Instead, take the best contiguous run of title tokens that DOES occur.

    Ranked by real words first, then length, then earliest. Words-before-length is
    what keeps a leaked carrier ("reminder for 4 .") from beating the real title
    ("visit grandma please") on token count alone.
    """
    lowered = [t.lower() for t in tokens]
    wanted = [t.lower() for t in title_tokens]

    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    for first in range(len(wanted)):
        for last in range(first + 1, len(wanted) + 1):
            piece = wanted[first:last]
            for start in range(len(lowered) - len(piece) + 1):
                if lowered[start : start + len(piece)] != piece:
                    continue
                words = sum(1 for token in piece if any(c.isalpha() for c in token))
                key = (words, len(piece), -start)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (start, start + len(piece))
                break
    return best


def annotate(text: str, title: str | None) -> tuple[list[str], list[str]] | None:
    """
    Turn one (text, title) pair into BIO tags over the PREPROCESSED tokens.

    Returns None only when nothing at all could be located, so a failure is dropped
    and counted rather than quietly becoming a negative example.
    """
    tokens = prepare(text)
    if not tokens:
        return None

    tags = [OUTSIDE] * len(tokens)
    if not title:
        # A genuine negative: "every day", "set up a reminder". Worth training on.
        return tokens, tags

    title_tokens = prepare(title)
    if not title_tokens:
        return tokens, tags

    span = best_span(tokens, title_tokens)
    if span is None:
        return None

    start, end = span
    tags[start] = BEGIN
    for offset in range(start + 1, end):
        tags[offset] = INSIDE
    return tokens, tags


def apply_typos(text: str) -> str:
    """Swap known words for their common misspellings, leaving everything else alone."""
    return " ".join(TYPO_MAP.get(word.lower(), word) for word in text.split())


def build_corpus() -> tuple[list[tuple[list[str], list[str]]], list[tuple[str, str | None]]]:
    """
    Build token-level training data from train.csv, with multi-category augmentation.

    Labels come from the rule-based NLUEngine, so the tagger inherits whatever that
    pipeline gets right AND wrong. Every sentence it cannot label is counted and
    reported below rather than silently becoming a negative example.
    """
    rng = random.Random(SEED)
    pack = load_pack(str(PROJECT_ROOT / "dist" / "nlu_pack"))
    engine = NLUEngine(pack=pack)

    csv_path = PROJECT_ROOT / "language_packs" / "en" / "train.csv"
    with open(csv_path, "r", encoding="utf-8") as handle:
        sentences = [
            row["text"].strip()
            for row in csv.DictReader(handle)
            if row["intent"] == "reminders.add"
        ]

    # 1. Base annotation, straight from the engine.
    pairs: list[tuple[str, str | None]] = []
    for text in sentences:
        result = engine.handle("build_session", text)
        pairs.append((text, result.parameters.get("name")))
        engine.reset("build_session")

    # 2. Augmentation at TEXT level, so preprocessing sees it exactly as production will.
    augmented: list[tuple[str, str | None]] = []
    for text, title in pairs:
        if rng.random() < 0.3:
            augmented.append((f"{rng.choice(FILLERS)} {text}", title))

        typo_text = apply_typos(text)
        if typo_text != text:
            augmented.append((typo_text, apply_typos(title) if title else title))

        if rng.random() < 0.2:
            augmented.append((f"{text} {rng.choice(RUN_ONS)}", title))

    corpus: list[tuple[list[str], list[str]]] = []
    dropped: list[tuple[str, str | None]] = []
    for text, title in pairs + augmented:
        annotated = annotate(text, title)
        if annotated is None:
            dropped.append((text, title))
        else:
            corpus.append(annotated)

    total = len(pairs) + len(augmented)
    negatives = sum(1 for _, tags in corpus if all(tag == OUTSIDE for tag in tags))
    print(f"  base sentences           : {len(pairs)}")
    print(f"  augmented sentences      : {len(augmented)}")
    print(
        f"  dropped (title not locatable): {len(dropped)}  "
        f"({len(dropped) * 100 // max(total, 1)}%)"
    )
    print(f"  kept                     : {len(corpus)}")
    print(f"  of which all-O negatives : {negatives}")
    if dropped:
        print("  examples the engine could not align:")
        for text, title in dropped[:5]:
            print(f"     {text!r}\n        title={title!r}")

    return corpus, pairs


def to_instances(corpus):
    """Flatten sequences into per-token (features, label) pairs."""
    features, labels = [], []
    for tokens, tags in corpus:
        for index in range(len(tokens)):
            features.append(extract_token_features(tokens, index))
            labels.append(tags[index])
    return features, labels


def train_bio_tagger(corpus):
    """Train a LogisticRegression token classifier and report held-out quality."""
    train_seqs, test_seqs = train_test_split(corpus, test_size=HOLDOUT_FRACTION, random_state=SEED)
    x_train, y_train = to_instances(train_seqs)

    print(f"\n  training tokens: {len(x_train)}  (held out {len(test_seqs)} sequences)")
    vec = DictVectorizer(sparse=True)
    x_vec = vec.fit_transform(x_train)

    # Roughly 70% of tokens are "O", so without balancing the model is happiest
    # predicting "no title here" -- which is the failure this tagger exists to fix.
    clf = LogisticRegression(max_iter=500, C=15.0, class_weight="balanced")
    clf.fit(x_vec, y_train)

    evaluate(vec, clf, test_seqs)
    return vec, clf


def predict_tags(tokens, vec, clf):
    """Predict BIO tags for a list of already-prepared tokens."""
    features = [extract_token_features(tokens, i) for i in range(len(tokens))]
    return list(clf.predict(vec.transform(features)))


def evaluate(vec, clf, test_seqs) -> None:
    """Report token-level precision/recall and, more usefully, exact span accuracy."""
    if not test_seqs:
        print("  no held-out data -- skipping evaluation")
        return

    x_test, y_test = to_instances(test_seqs)
    predicted = list(clf.predict(vec.transform(x_test)))
    print("\n  token-level:")
    print(classification_report(y_test, predicted, zero_division=0, digits=3))

    exact = 0
    for tokens, tags in test_seqs:
        gold = extract_title_span(tokens, tags)
        guess = extract_title_span(tokens, predict_tags(tokens, vec, clf))
        if gold == guess:
            exact += 1
    share = exact * 100 / len(test_seqs)
    print(f"  exact title match: {exact}/{len(test_seqs)}  ({share:.1f}%)")


def test_custom_phrase(phrase: str, vec, clf) -> None:
    """Show the tokens, their tags and the extracted title for one phrase."""
    tokens = prepare(phrase)
    tags = predict_tags(tokens, vec, clf)

    print("\n" + "=" * 65)
    print(f'Input : "{phrase}"')
    print("-" * 65)
    for token, tag in zip(tokens, tags):
        print(f"   {token:<15} -> {tag if tag != OUTSIDE else 'O'}")
    print("-" * 65)
    print(f'Title : "{extract_title_span(tokens, tags) or "(None)"}"')
    print("=" * 65)


def main() -> None:
    print("=== Training Robust BIO Slot Tagging Model ===")
    corpus, _ = build_corpus()
    vec, clf = train_bio_tagger(corpus)

    import joblib

    joblib.dump(
        {"vectorizer": vec, "classifier": clf}, PROJECT_ROOT / "models" / "bio_slot_tagger.pkl"
    )
    print("\nUpdated models/bio_slot_tagger.pkl")

    args = sys.argv[1:]
    if "--interactive" in args or "-i" in args:
        print("\n=== Interactive BIO Slot Tester (type 'exit' to quit) ===\n")
        while True:
            try:
                phrase = input("Enter phrase: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not phrase or phrase.lower() in ("exit", "quit"):
                break
            test_custom_phrase(phrase, vec, clf)
        return

    if args:
        test_custom_phrase(" ".join(args), vec, clf)


if __name__ == "__main__":
    main()
