"""Phase 22 — structured error analysis.

Every wrong prediction is saved with text / true / predicted / confidence /
top2 / margin, then bucketed into a cause so the next data batch targets a
known failure mode instead of being random extra sentences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from calibration import margin_of  # noqa: E402
from common import df_to_markdown  # noqa: E402
from pipeline import DATA, IntentModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FALLBACK = "Default Fallback Intent"

SUITES = {
    "standard_test": "test.csv",
    "contextual": "contextual_test.csv",
    "accessories": "accessories_test.csv",
    "minimal_pairs": "minimal_pair_test.csv",
    "hard_negatives": "hard_negative_test.csv",
    "negation": "negation_test.csv",
    "stt": "stt_test.csv",
    "ood": "ood_test.csv",
}

NEG_MARKERS = {"not", "dont", "don't", "never", "no", "cannot", "cant", "stop"}
HELP_MARKERS = {"how", "what", "where", "why", "explain", "guide", "help",
                "show", "tell"}


def categorize(row: dict, families: dict, kinds: dict, train_counts: dict) -> str:
    text = str(row["text"]).lower()
    gold, pred = row["true_intent"], row["predicted_intent"]
    toks = set(text.replace("'", "").split())

    if row["suite"] == "ood":
        return "ood_false_acceptance"
    if row["suite"] == "stt":
        return "stt_error"
    if toks & NEG_MARKERS:
        return "negation"
    # `kind` is now measured from the data (action / howto / mixed), not read off
    # the label prefix — see policy P1b. Only an action<->howto swap is a genuine
    # result-vs-feature confusion; `mixed` intents legitimately absorb both
    # frames (Help_HeartRate answers "what is my heart rate" AND "how does it
    # work"), so counting those as confusions inflates this bucket.
    gk, pk = kinds.get(gold), kinds.get(pred)
    if {gk, pk} == {"action", "howto"}:
        return "result_request_vs_feature_question"
    if families.get(gold) and families.get(gold) == families.get(pred):
        return "same_family_opposite_or_sibling"
    if len(text.split()) >= 14:
        return "long_context"
    if train_counts.get(gold, 0) < 60:
        return "insufficient_training_coverage"
    if row["margin"] < 0.10:
        return "low_margin_uncertainty"
    if row["confidence"] >= 0.90:
        return "confident_and_wrong_calibration_issue"
    return "other_embedding_or_classifier_weakness"


def main(model_dir: str = "models/final", out_prefix: str = "final") -> None:
    model = IntentModel.load(ROOT / model_dir)
    cfg = yaml.safe_load((ROOT / "configs" / "intents.yaml").read_text())
    families = {k: v["family"] for k, v in cfg["intents"].items()}
    kinds = {k: v["kind"] for k, v in cfg["intents"].items()}
    train_counts = pd.read_csv(DATA / "train.csv")["intent"].value_counts().to_dict()

    rows = []
    for suite, fname in SUITES.items():
        df = pd.read_csv(DATA / fname)
        probs = model.probs(df["text"].tolist())
        order = np.argsort(-probs, axis=1)
        labels = np.array(model.labels)
        pred = labels[order[:, 0]]
        top2 = labels[order[:, 1]]
        conf = probs.max(1)
        marg = margin_of(probs)
        dec = model.decide(df["text"].tolist()) if model.gate else None
        for i in range(len(df)):
            wrong = pred[i] != df["intent"].iloc[i]
            if suite == "ood":
                wrong = bool(dec[i]["accepted"]) if dec else pred[i] != FALLBACK
            if not wrong:
                continue
            rows.append(dict(suite=suite, text=df["text"].iloc[i],
                             true_intent=df["intent"].iloc[i],
                             predicted_intent=pred[i], top2=top2[i],
                             confidence=round(float(conf[i]), 4),
                             margin=round(float(marg[i]), 4),
                             accepted=bool(dec[i]["accepted"]) if dec else None))

    err = pd.DataFrame(rows)
    if err.empty:
        print("no errors")
        return
    err["category"] = [categorize(r, families, kinds, train_counts)
                       for r in err.to_dict("records")]
    out_csv = ROOT / "reports" / f"errors_{out_prefix}.csv"
    err.to_csv(out_csv, index=False)

    summary = err.groupby(["category"]).size().sort_values(ascending=False)
    by_suite = err.groupby(["suite", "category"]).size().unstack(fill_value=0)
    conf_pairs = (err.groupby(["true_intent", "predicted_intent"]).size()
                  .sort_values(ascending=False).head(25))

    dangerous = err[(err["accepted"] == True) & (err["suite"] != "ood")]  # noqa: E712

    md = ["# Error Analysis\n",
          f"Total errors captured across {len(SUITES)} suites: **{len(err)}**\n",
          "## By cause\n", "| category | n | share |", "|---|---|---|"]
    for c, n in summary.items():
        md.append(f"| {c} | {n} | {100*n/len(err):.1f}% |")
    md += ["", "## By suite\n", df_to_markdown(by_suite, index=True), ""]
    md += ["## Most frequent confusions (true -> predicted)\n",
           "| true | predicted | n |", "|---|---|---|"]
    for (t, p), n in conf_pairs.items():
        md.append(f"| `{t}` | `{p}` | {n} |")
    md += ["", "## Errors that the safety gate ACCEPTED (false executions)\n",
           f"These are the dangerous ones: the model was wrong and the gate let "
           f"it through. Count: **{len(dangerous)}**\n"]
    if len(dangerous):
        md += ["| text | true | predicted | conf | margin |", "|---|---|---|---|---|"]
        for _, r in dangerous.sort_values("confidence", ascending=False).head(30).iterrows():
            md.append(f"| {r['text']} | `{r['true_intent']}` | "
                      f"`{r['predicted_intent']}` | {r['confidence']:.3f} | {r['margin']:.3f} |")
    md += ["", "## Next targeted data batch\n",
           "Ordered by how many errors each cause explains. Per Phase 23, the "
           "next batch addresses these and nothing else.\n"]
    ACTION = {
        "negation": "extend the P2/P3 negation templates with new openers and objects",
        "result_request_vs_feature_question": "add matched result-request / feature-question pairs for the affected topic (policy P1)",
        "same_family_opposite_or_sibling": "add minimal pairs inside that family",
        "insufficient_training_coverage": "collect or generate examples for the tail intents",
        "stt_error": "add the observed corruption pattern to the STT augmentation set",
        "long_context": "add long conversational forms for the affected intents",
        "ood_false_acceptance": "add near-OOD training examples for the accepting intent",
        "low_margin_uncertainty": "raise the margin threshold or add separating examples",
        "confident_and_wrong_calibration_issue": "re-check calibration; consider per-class temperature",
        "other_embedding_or_classifier_weakness": "candidate for a stronger encoder",
    }
    for c, n in summary.items():
        md.append(f"- **{c}** ({n}) — {ACTION.get(c, 'investigate')}")

    (ROOT / "reports" / f"error_analysis_{out_prefix}.md").write_text("\n".join(md))
    print(summary.to_string())
    print(f"\naccepted-but-wrong (false executions in-domain): {len(dangerous)}")
    print(f"-> reports/error_analysis_{out_prefix}.md")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
