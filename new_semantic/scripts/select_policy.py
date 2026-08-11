#!/usr/bin/env python3
"""
Pick the Stage2+Stage3 combination policy HONESTLY.

Two problems this fixes:

1. Every threshold and policy decision so far was made on the same 565 stress
   and 403 OOD rows that were then reported. Seventeen policies were compared on
   one seed. That is selection on the test set, and with sets this small the
   winner is partly noise.

2. Model selection already burned us once: identical configs varied by 44 points
   on OOD across seeds. Policy selection has the same exposure.

So: eval sets are split ONCE into dev/test (stratified, fixed seed, saved to
disk). Policies are chosen on DEV, averaged over seeds. The chosen policy is
then reported on TEST — read exactly once, at the end.

Usage:
    python scripts/select_policy.py --tags unkaug_s1 unkaug_s42 unkaug_s7
    python scripts/select_policy.py --tags ... --reveal-test
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from scripts.common import encode, load_rows, load_vocab, token_key  # noqa: E402
from scripts.compare_policies import stage2_probs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "models" / "intent_classifier_weights.json"
S2_LABELS = REPO / "models" / "intent_labels.json"
SPLIT = config.DATA / "eval" / "dev_test_split.json"
SPLIT_SEED = 20260810


def build_split(force=False) -> dict:
    """Stratified 50/50 dev/test assignment, keyed on token_key so it survives
    re-generation of the CSVs. Written once and then never changed."""
    if SPLIT.exists() and not force:
        return json.loads(SPLIT.read_text(encoding="utf-8"))

    rng = np.random.default_rng(SPLIT_SEED)
    assign: dict[str, str] = {}
    for name, path in (
        ("stress", config.STRESS_TEST),
        ("locked", config.LOCKED_TEST),
        ("ood", config.OOD_TEST),
    ):
        if not path.exists():
            continue
        by_label = defaultdict(list)
        for t, g in load_rows(path):
            by_label[g].append(token_key(t))
        for _, keys in by_label.items():
            keys = sorted(set(keys))
            order = rng.permutation(len(keys))
            for rank, j in enumerate(order):
                assign[keys[j]] = "dev" if rank % 2 == 0 else "test"
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps({"seed": SPLIT_SEED, "assign": assign}, indent=2), encoding="utf-8")
    return {"seed": SPLIT_SEED, "assign": assign}


def student_temperature(tag: str, override: float | None = None) -> float:
    """The temperature the RUNTIME will apply to this student's logits.

    Why this exists: gates are thresholds on a confidence number, so a gate is
    only meaningful on the scale it was chosen on. `StudentSemantic` now divides
    logits by the fitted T before softmax, which sharpens confidence — pick the
    gate at T=1 and ship at T=0.68 and the shipped gate is effectively looser
    than the one that was measured. Selecting here on the uncalibrated scale
    while the runtime serves a calibrated one is the same class of mistake as
    reading Stage 2's decision_function as if it were probabilities.

    Falls back to 1.0 (identity) when no calibration has been fitted, so an
    uncalibrated tag behaves exactly as it did before.
    """
    if override is not None:
        return override
    rep = config.REPORTS / f"calibration_{tag}.json"
    if not rep.exists():
        return 1.0
    return float(json.loads(rep.read_text(encoding="utf-8")).get("temperature", 1.0))


def policy_scores(tag, split, part, student_T: float = 1.0):
    """All policy scores (end-to-end) for one model on one split part."""
    import torch

    from scripts.train_en import build_student

    meta_w = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    s2_T = float(meta_w.get("temperature", 1.0))
    s2_gate = float(meta_w.get("conf_threshold", 0.7))
    s2_labels = json.loads(S2_LABELS.read_text(encoding="utf-8"))

    vocab, tok_mode = load_vocab(config.MODELS / f"vocab_{tag}.json")
    meta = json.loads((config.REPORTS / f"train_{tag}_summary.json").read_text(encoding="utf-8"))
    max_len = meta.get("max_len", config.MAX_LEN)
    tok_mode = meta.get("tokenizer", tok_mode)
    labels = json.loads((config.MODELS / f"labels_{tag}.json").read_text(encoding="utf-8"))
    model = build_student(len(vocab), len(labels), dim=meta.get("embed_dim", config.EMBED_DIM))
    model.load_state_dict(torch.load(config.MODELS / f"student_{tag}.pt"))
    model.eval()

    perm = [{l: i for i, l in enumerate(s2_labels)}[l] for l in labels]
    fb = labels.index(config.FALLBACK_INTENT)
    out = {}

    for name, path, is_ood in (
        ("stress", config.STRESS_TEST, False),
        ("locked", config.LOCKED_TEST, False),
        ("ood", config.OOD_TEST, True),
    ):
        if not path.exists():
            continue
        rows = [
            (t, g)
            for t, g in load_rows(path)
            if split["assign"].get(token_key(t)) == part and (is_ood or g in labels)
        ]
        if not rows:
            continue
        texts = [t for t, _ in rows]
        gold = [g for _, g in rows]

        P2 = stage2_probs(texts, s2_T)[:, perm]
        c2, i2 = P2.max(1), P2.argmax(1)
        hand = c2 < s2_gate
        idx = np.where(hand)[0]
        if len(idx) == 0:
            continue

        X = np.array([encode(texts[i], vocab, max_len, tok_mode)[0] for i in idx], dtype=np.int64)
        M = X != config.PAD_ID
        with torch.no_grad():
            # divide by T BEFORE softmax — this must match StudentSemantic.classify
            PS = (
                torch.softmax(model(torch.tensor(X), torch.tensor(M)) / student_T, -1)
                .numpy()
                .astype(np.float64)
            )
        cs, isx = PS.max(1), PS.argmax(1)
        p2 = P2[idx]
        c2h = p2.max(1)
        mx = np.where(cs >= c2h, isx, p2.argmax(1))

        kept = np.where(~hand)[0]
        if is_ood:
            kept_ok = int(sum(1 for i in kept if i2[i] == fb))
        else:
            kept_ok = int(sum(1 for i in kept if labels[i2[i]] == gold[i]))

        def e2e(pred_idx, accept):
            if is_ood:
                v = float(np.mean([(p == fb) or (not a) for p, a in zip(pred_idx, accept)]))
            else:
                v = float(
                    np.mean([a and labels[p] == gold[i] for p, a, i in zip(pred_idx, accept, idx)])
                )
            return (kept_ok + v * len(idx)) / len(texts)

        pol = {}
        pol["s2_only"] = e2e(p2.argmax(1), [True] * len(idx))
        pol["replace"] = e2e(isx, cs >= 0.40)
        pol["max_conf"] = e2e(mx, [True] * len(idx))
        pol["s2_backstop"] = e2e(np.where(cs >= 0.40, isx, p2.argmax(1)), [True] * len(idx))
        for s3g in (0.40, 0.50, 0.60):
            for s2g in (0.30, 0.45, 0.60):
                pol[f"dual@s3={s3g:.2f},s2={s2g:.2f}"] = e2e(mx, (cs >= s3g) | (c2h >= s2g))
        avg = (PS + p2) / 2
        pol["avg"] = e2e(avg.argmax(1), avg.max(1) >= 0.40)
        out[name] = pol
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument(
        "--reveal-test",
        action="store_true",
        help="also print TEST numbers (do this once, at the end)",
    )
    ap.add_argument("--rebuild-split", action="store_true")
    ap.add_argument(
        "--student-temperature",
        type=float,
        default=None,
        help="override the fitted T; default reads reports/calibration_<tag>.json, "
        "falling back to 1.0. Use 1.0 to reproduce pre-calibration selections.",
    )
    args = ap.parse_args()

    split = build_split(force=args.rebuild_split)
    n_dev = sum(1 for v in split["assign"].values() if v == "dev")
    print(
        f"dev/test split: {n_dev} dev / {len(split['assign']) - n_dev} test "
        f"(seed {split['seed']}, saved at {SPLIT.name})\n"
    )

    agg = {
        "dev": defaultdict(lambda: defaultdict(list)),
        "test": defaultdict(lambda: defaultdict(list)),
    }
    temps = {t: student_temperature(t, args.student_temperature) for t in args.tags}
    print("student temperature per tag (the scale the gate is chosen on):")
    for t, T in temps.items():
        src = (
            "override"
            if args.student_temperature is not None
            else ("calibration_%s.json" % t if T != 1.0 else "uncalibrated -> 1.0")
        )
        print(f"   {t:<16} T={T:.4f}   ({src})")
    if len(set(temps.values())) > 1:
        print(
            "   ! tags disagree on T. Thresholds are not comparable across "
            "different scales; calibrate every tag or pass --student-temperature."
        )
    print()

    for tag in args.tags:
        for part in ("dev", "test"):
            for setname, pol in policy_scores(tag, split, part, temps[tag]).items():
                for k, v in pol.items():
                    agg[part][setname][k].append(v)

    def table(part):
        keys = list(agg[part]["stress"].keys())
        rows = []
        for k in keys:
            s = agg[part]["stress"][k]
            o = agg[part]["ood"][k]
            rows.append(
                (
                    k,
                    st.mean(s),
                    st.stdev(s) if len(s) > 1 else 0.0,
                    st.mean(o),
                    st.stdev(o) if len(o) > 1 else 0.0,
                )
            )
        return sorted(rows, key=lambda r: -r[1])

    print(f"=== DEV (n={len(args.tags)} seeds) — selection happens here ===")
    print(f"{'policy':<26}{'stress':>9}{'sd':>7}{'OOD':>9}{'sd':>7}{'harmonic':>10}")
    dev = table("dev")
    scored = []
    for k, sm, ss, om, os_ in dev:
        h = 0.0 if sm + om == 0 else 2 * sm * om / (sm + om)
        scored.append((k, h, sm, om))
        print(f"{k:<26}{sm:>9.4f}{ss:>7.4f}{om:>9.4f}{os_:>7.4f}{h:>10.4f}")

    best = max(scored, key=lambda x: x[1])
    print(f"\nSELECTED on dev (max harmonic of stress & OOD): {best[0]}")
    print(f"   dev stress {best[2]:.4f}   dev OOD {best[3]:.4f}")

    if args.reveal_test:
        t = {r[0]: r for r in table("test")}
        print("\n=== TEST (read once) ===")
        for k in ("s2_only", "replace", best[0]):
            if k in t:
                _, sm, ss, om, os_ = t[k]
                tagl = {"s2_only": " (bar)", "replace": " (current)"}.get(k, " <- SELECTED")
                print(f"  {k:<26} stress {sm:.4f} ±{ss:.4f}   OOD {om:.4f} ±{os_:.4f}{tagl}")
        gap = best[2] - t[best[0]][1] if best[0] in t else None
        if gap is not None:
            print(f"\n  dev->test drop for the selected policy: {gap:+.4f} on stress")
            print("  (a large drop means the dev choice was partly noise)")

    payload = {
        "tags": args.tags,
        "split_seed": split["seed"],
        "selected": best[0],
        "selection_criterion": "max harmonic(stress, OOD) on DEV, mean over seeds",
        "dev": {k: {"stress": sm, "ood": om} for k, _, sm, om in scored},
    }
    if args.reveal_test:
        t = {r[0]: r for r in table("test")}
        payload["test"] = {
            k: {"stress": r[1], "stress_sd": r[2], "ood": r[3], "ood_sd": r[4]}
            for k, r in t.items()
        }
        if best[0] in t:
            payload["dev_to_test_drop"] = {
                "stress": round(t[best[0]][1] - best[2], 4),
                "ood": round(t[best[0]][3] - best[3], 4),
            }
    (config.REPORTS / "policy_selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {config.REPORTS / 'policy_selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
