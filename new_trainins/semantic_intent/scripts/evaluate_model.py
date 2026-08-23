"""Phase 24 — the full evaluation matrix, run against one fitted IntentModel.

Suites: standard test, contextual, minimal pairs, hard negatives, negation,
STT noise, OOD. Plus calibration and the accept/reject gate behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from calibration import ece, margin_of, mce, reliability, brier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FALLBACK = "Default Fallback Intent"


def _basic(model, df, calibrated=True) -> dict:
    probs = model.probs(df["text"].tolist(), calibrated=calibrated)
    pred = np.array(model.labels)[probs.argmax(1)]
    y = df["intent"].tolist()
    return dict(n=len(df),
                accuracy=round(float(accuracy_score(y, pred)), 4),
                macro_f1=round(float(f1_score(y, pred, average="macro",
                                              zero_division=0)), 4),
                probs=probs, pred=pred, y=y)


def eval_standard(model, df, calibrated=True) -> dict:
    r = _basic(model, df, calibrated)
    probs, y = r.pop("probs"), r["y"]
    yi = model.y_index(y)
    conf = probs.max(1)
    correct = (probs.argmax(1) == yi)
    r["ece"] = round(ece(conf, correct), 4)
    r["mce"] = round(mce(conf, correct), 4)
    r["brier"] = round(brier(probs, yi), 4)
    r["mean_conf"] = round(float(conf.mean()), 4)
    r["mean_margin"] = round(float(margin_of(probs).mean()), 4)
    r["reliability"] = reliability(conf, correct)
    r.pop("pred"); r.pop("y")
    return r


def eval_gated(model, df) -> dict:
    """Accept/reject behaviour on in-domain data."""
    dec = model.decide(df["text"].tolist())
    y = df["intent"].tolist()
    acc_mask = np.array([d["accepted"] for d in dec])
    pred = np.array([d["intent"] for d in dec])
    correct = pred == np.array(y)
    n_acc = int(acc_mask.sum())
    return dict(n=len(df), coverage=round(float(acc_mask.mean()), 4),
                accepted_precision=round(float(correct[acc_mask].mean()), 4) if n_acc else None,
                rejected_error_rate=round(float((~correct[~acc_mask]).mean()), 4) if (~acc_mask).sum() else None,
                false_rejection=round(float((correct & ~acc_mask).mean()), 4),
                false_execution=round(float((~correct & acc_mask).mean()), 4))


def eval_ood(model, df) -> dict:
    """OOD must be rejected: either classified as the reject label or gated out."""
    dec = model.decide(df["text"].tolist())
    accepted = np.array([d["accepted"] for d in dec])
    as_fallback = np.array([d["intent"] == FALLBACK for d in dec])
    out = dict(n=len(df),
               rejection_rate=round(float((~accepted).mean()), 4),
               false_acceptance=round(float(accepted.mean()), 4),
               classified_as_fallback=round(float(as_fallback.mean()), 4))
    if "ood_type" in df.columns:
        for t, sub in df.groupby("ood_type"):
            idx = df.index.get_indexer(sub.index)
            out[f"rejection_{t}"] = round(float((~accepted[idx]).mean()), 4)
    # A single averaged rejection rate hides which KIND of input gets through.
    # "0.87 overall" is compatible with rejecting every weather question and
    # accepting every request aimed at another device in the room — and only
    # one of those two failures reaches the hardware.
    if "family" in df.columns:
        per_family = {}
        for f, sub in df.groupby("family"):
            idx = df.index.get_indexer(sub.index)
            per_family[str(f)] = dict(n=len(sub),
                                      rejection=round(float((~accepted[idx]).mean()), 4))
        out["by_family"] = dict(sorted(per_family.items(),
                                       key=lambda kv: kv[1]["rejection"]))
    if accepted.any():
        out["false_accept_examples"] = [
            dict(text=t, intent=d["intent"], confidence=d["confidence"])
            for t, d, a in zip(df["text"], dec, accepted) if a
        ][:15]
    return out


def eval_minimal_pairs(model, df) -> dict:
    r = _basic(model, df)
    probs = r.pop("probs")
    pred = r["pred"]
    df = df.copy()
    df["pred"] = pred
    df["ok"] = df["pred"] == df["intent"]
    both = df.groupby("pair_id")["ok"].all()
    per_axis = df.groupby("axis")["ok"].mean().round(4).to_dict()
    r.pop("pred"); r.pop("y")
    r.update(_gated_stats(model, df, pred))
    r["pair_accuracy"] = round(float(both.mean()), 4)
    r["per_axis_accuracy"] = per_axis
    r["failed_pairs"] = [
        dict(pair_id=p,
             items=[dict(text=t, gold=g, pred=pr)
                    for t, g, pr in zip(sub["text"], sub["intent"], sub["pred"])])
        for p, sub in df[df["pair_id"].isin(both[~both].index)].groupby("pair_id")
    ][:20]
    return r


def _gated_stats(model, df, pred) -> dict:
    """What the GATE does on this suite, not just what the classifier says.

    The suite accuracies above are classification metrics — they ignore the
    gate entirely. A structural refusal cannot show up in them by construction,
    so without this the whole point of a refusal (fewer wrong ACTIONS) is
    invisible and would look like the fix did nothing.
    """
    if model.gate is None:
        return {}
    dec = model.decide(df["text"].tolist())
    acc = np.array([d["accepted"] for d in dec])
    correct = np.array(pred) == df["intent"].values
    return dict(
        gated_coverage=round(float(acc.mean()), 4),
        gated_false_execution=round(float((acc & ~correct).mean()), 4),
        gated_accepted_precision=(round(float(correct[acc].mean()), 4)
                                  if acc.any() else None),
        refused_as_corrective=round(
            float(np.mean([d.get("corrective", False) for d in dec])), 4),
    )


def eval_simple(model, df, group_col: str | None = None) -> dict:
    r = _basic(model, df)
    probs = r.pop("probs")
    pred, y = r.pop("pred"), r.pop("y")
    d = df.copy(); d["pred"] = pred; d["ok"] = d["pred"] == d["intent"]
    if group_col and group_col in d.columns:
        r["per_group_accuracy"] = d.groupby(group_col)["ok"].mean().round(4).to_dict()
    r.update(_gated_stats(model, df, pred))
    r["failures"] = [dict(text=t, gold=g, pred=p, confidence=round(float(c), 3))
                     for t, g, p, c in zip(d.loc[~d["ok"], "text"],
                                           d.loc[~d["ok"], "intent"],
                                           d.loc[~d["ok"], "pred"],
                                           probs[~d["ok"].values].max(1))][:25]
    return r


def run_all(model, include_gated: bool = True) -> dict:
    out = {}
    out["standard_test"] = eval_standard(model, pd.read_csv(DATA / "test.csv"))
    out["validation"] = eval_standard(model, pd.read_csv(DATA / "validation.csv"))
    out["contextual"] = eval_simple(model, pd.read_csv(DATA / "contextual_test.csv"))
    # Accessories are the product's first priority; without a line here they
    # were invisible to every report and only findable by grepping errors.csv.
    out["accessories"] = eval_simple(model, pd.read_csv(DATA / "accessories_test.csv"))
    out["minimal_pairs"] = eval_minimal_pairs(model, pd.read_csv(DATA / "minimal_pair_test.csv"))
    out["hard_negatives"] = eval_simple(model, pd.read_csv(DATA / "hard_negative_test.csv"), "reason")
    out["negation"] = eval_simple(model, pd.read_csv(DATA / "negation_test.csv"), "policy")
    out["stt"] = eval_simple(model, pd.read_csv(DATA / "stt_test.csv"), "ops")
    if include_gated and model.gate is not None:
        out["gated_test"] = eval_gated(model, pd.read_csv(DATA / "test.csv"))
        out["ood"] = eval_ood(model, pd.read_csv(DATA / "ood_test.csv"))
    return out


def headline(res: dict) -> str:
    s = res["standard_test"]
    parts = [f"test acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f} ECE={s['ece']:.4f}",
             f"ctx={res['contextual']['accuracy']:.3f}",
             f"mp_pair={res['minimal_pairs']['pair_accuracy']:.3f}",
             f"hardneg={res['hard_negatives']['accuracy']:.3f}",
             f"neg={res['negation']['accuracy']:.3f}",
             f"hn_falseexec={res['hard_negatives'].get('gated_false_execution', float('nan')):.3f}",
             f"stt={res['stt']['accuracy']:.3f}"]
    if "ood" in res:
        parts.append(f"ood_rej={res['ood']['rejection_rate']:.3f}")
    if "gated_test" in res:
        g = res["gated_test"]
        # accepted_precision is None when the gate accepted nothing at all —
        # a real outcome (thresholds pinned at the ceiling), not a bug to
        # crash on. Reporting it as "n/a (0 accepted)" is the whole point.
        ap_ = g.get("accepted_precision")
        ap_s = f"{ap_:.4f}" if ap_ is not None else "n/a(0 accepted)"
        parts.append(f"cov={g['coverage']:.3f} acc_prec={ap_s}")
    return " | ".join(parts)
