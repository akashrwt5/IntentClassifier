"""Phases 15-20 — calibration, calibration metrics, threshold + margin gate.

Nothing here is allowed to look at the final test set. Every parameter
(temperature, confidence threshold, margin threshold) is fitted on the
validation split only.
"""
from __future__ import annotations

import re

import numpy as np

# A sentence that rejects one option and asks for another. Measured accuracy on
# this shape is 0.48 on intent pairs the model was never taught it for and 0.74
# on the ones it was — both far below the 0.97 precision the gate promises, so
# the honest answer is to ask the user again rather than act on a coin flip.
#
# Deterministic rules are a last resort here (plan Section 23), and this one is
# scoped to earn that: it fires on a structural pattern, not on a keyword, and
# it was measured before being adopted — 0 of 1513 held-out rows, 0 of 1508
# validation rows and 0 of 1496 STT rows match it, so it costs nothing on real
# traffic while catching 75% of the corrective cases.
#
# Requiring BOTH halves is what keeps it honest: a rejection, a clause break,
# and then a request. "i don't want the stream anymore" is a plain command with
# a negated verb and no second option, and it must not match.
_NEG = r"(?:not|no|never|dont|do not|didnt|did not|skip|forget)"
CORRECTIVE_STRUCTURE = re.compile(
    rf"\b{_NEG}\b[^,\-—]*[,\-—]\s*\S+"
    rf"|\bi meant\b"
    rf"|\binstead\b"
    rf"|\bis not what i (?:wanted|asked)\b"
)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(logits / T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0,
                    iters: int = 60) -> float:
    """Golden-section search on NLL. Deterministic, no optimizer dependency."""
    phi = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(logits, y, c), nll(logits, y, d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(logits, y, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(logits, y, d)
    return float((a + b) / 2)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------
def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append(dict(bin_lo=lo, bin_hi=hi, n=0, avg_conf=np.nan,
                             accuracy=np.nan, gap=np.nan))
            continue
        rows.append(dict(bin_lo=float(lo), bin_hi=float(hi), n=int(m.sum()),
                         avg_conf=float(conf[m].mean()),
                         accuracy=float(correct[m].mean()),
                         gap=float(conf[m].mean() - correct[m].mean())))
    return rows


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    rows = reliability(conf, correct, n_bins)
    n = len(conf)
    return float(sum(r["n"] / n * abs(r["gap"]) for r in rows if r["n"]))


def mce(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    rows = reliability(conf, correct, n_bins)
    gaps = [abs(r["gap"]) for r in rows if r["n"] >= 10]
    return float(max(gaps)) if gaps else float("nan")


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# ---------------------------------------------------------------------------
# Operating point selection
# ---------------------------------------------------------------------------
def margin_of(probs: np.ndarray) -> np.ndarray:
    part = np.partition(probs, -2, axis=1)
    return part[:, -1] - part[:, -2]


def select_operating_point(probs: np.ndarray, y: np.ndarray,
                           target_precision: float = 0.97,
                           min_coverage: float = 0.50,
                           margin_grid=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)):
    """Smallest gate that reaches `target_precision` on accepted predictions
    while keeping coverage >= min_coverage. Precision-first, per Phase 17:
    a false execution on a hearing aid is worse than asking again."""
    pred = probs.argmax(1)
    conf = probs.max(1)
    marg = margin_of(probs)
    correct = (pred == y)

    best = None
    for m_thr in margin_grid:
        for c_thr in np.round(np.arange(0.20, 0.995, 0.005), 3):
            acc_mask = (conf >= c_thr) & (marg >= m_thr)
            n_acc = int(acc_mask.sum())
            if n_acc == 0:
                continue
            precision = float(correct[acc_mask].mean())
            coverage = n_acc / len(y)
            if precision >= target_precision and coverage >= min_coverage:
                cand = dict(conf_threshold=float(c_thr), margin_threshold=float(m_thr),
                            precision=precision, coverage=coverage, n_accepted=n_acc)
                if best is None or cand["coverage"] > best["coverage"]:
                    best = cand
    if best is None:  # target unreachable — report the best precision available
        rows = []
        for m_thr in margin_grid:
            for c_thr in np.round(np.arange(0.20, 0.995, 0.005), 3):
                acc_mask = (conf >= c_thr) & (marg >= m_thr)
                if acc_mask.sum() < max(20, 0.05 * len(y)):
                    continue
                rows.append(dict(conf_threshold=float(c_thr),
                                 margin_threshold=float(m_thr),
                                 precision=float(correct[acc_mask].mean()),
                                 coverage=float(acc_mask.mean()),
                                 n_accepted=int(acc_mask.sum())))
        rows.sort(key=lambda r: (-r["precision"], -r["coverage"]))
        best = rows[0] if rows else dict(conf_threshold=0.5, margin_threshold=0.0,
                                         precision=float("nan"), coverage=1.0,
                                         n_accepted=len(y))
        best["target_met"] = False
    else:
        best["target_met"] = True
    best["target_precision"] = target_precision
    return best


def coverage_precision_curve(probs, y, margin_threshold=0.0):
    pred, conf = probs.argmax(1), probs.max(1)
    marg = margin_of(probs)
    correct = (pred == y)
    out = []
    for c in np.round(np.arange(0.3, 1.0, 0.05), 2):
        m = (conf >= c) & (marg >= margin_threshold)
        if m.sum() == 0:
            continue
        out.append(dict(threshold=float(c), coverage=float(m.mean()),
                        precision=float(correct[m].mean()),
                        n=int(m.sum())))
    return out


# ---------------------------------------------------------------------------
# The runtime safety gate (Phase 20)
# ---------------------------------------------------------------------------
class SafetyGate:
    """Four independent reasons to refuse.

    The reject class, confidence and margin all come from the classifier head.
    `ood_threshold` is the only one computed before the classifier, in embedding
    space, and it is the only one that can catch an input that is unlike
    anything in training while still looking decisively like one known class.
    """

    def __init__(self, conf_threshold: float, margin_threshold: float,
                 labels: list[str], reject_label: str = "Default Fallback Intent",
                 temperature: float = 1.0, ood_threshold: float = float("inf"),
                 ood_percentile: float | None = None,
                 risk_of: dict | None = None, conf_by_risk: dict | None = None,
                 reject_corrective: bool = True):
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold
        self.labels = labels
        self.reject_label = reject_label
        self.reject_index = labels.index(reject_label) if reject_label in labels else -1
        self.temperature = temperature
        self.ood_threshold = ood_threshold
        self.ood_percentile = ood_percentile
        # Per-intent thresholds. Muting an aid and nudging the volume are not
        # the same bet: one the user can undo in a second, the other removes
        # the channel they would use to notice the mistake.
        self.risk_of = risk_of or {}
        self.conf_by_risk = conf_by_risk or {}
        self.reject_corrective = reject_corrective

    def threshold_for(self, intent: str) -> float:
        tier = self.risk_of.get(intent, "normal")
        return self.conf_by_risk.get(tier, self.conf_threshold)

    @staticmethod
    def is_corrective(text: str) -> bool:
        """Deliberately NOT run on normalize() output.

        normalize() strips punctuation, and the comma or dash IS the structural
        signal here — it is what separates the rejected option from the
        requested one. Normalising first silently disabled the main branch of
        this pattern and left only the explicit "i meant" / "instead" markers
        working, which is why the first version caught exactly the 3 of 4 test
        frames that carry such a marker.
        """
        import unicodedata
        t = unicodedata.normalize("NFKC", str(text)).lower()
        t = t.replace("\u2019", "'").replace("\u2018", "'")
        t = " ".join(t.split())
        return bool(CORRECTIVE_STRUCTURE.search(t))

    def decide(self, logits: np.ndarray,
               ood_scores: np.ndarray | None = None,
               texts: list | None = None) -> list[dict]:
        probs = softmax(logits / self.temperature)
        order = np.argsort(-probs, axis=1)
        results = []
        for i in range(len(probs)):
            i1, i2 = order[i, 0], order[i, 1]
            c1, c2 = float(probs[i, i1]), float(probs[i, i2])
            m = c1 - c2
            o = float(ood_scores[i]) if ood_scores is not None else 0.0
            intent = self.labels[i1]
            thr = self.threshold_for(intent)
            tier = self.risk_of.get(intent, "normal")
            corrective = bool(self.reject_corrective and texts is not None
                              and self.is_corrective(texts[i]))
            if i1 == self.reject_index:
                accepted, reason = False, "classified as unsupported"
            elif corrective:
                accepted, reason = False, (
                    "corrective phrasing — the model cannot reliably tell which "
                    "option was rejected")
            elif ood_scores is not None and o > self.ood_threshold:
                accepted, reason = False, "unlike anything in training (OOD score)"
            elif c1 < thr:
                accepted, reason = False, (
                    f"below calibrated confidence threshold for {tier}-risk intent"
                    if tier != "normal" else
                    "below calibrated confidence threshold")
            elif m < self.margin_threshold:
                accepted, reason = False, "top-1/top-2 margin too small"
            else:
                accepted, reason = True, "above calibrated threshold"
            results.append(dict(intent=intent, risk=tier,
                                threshold_applied=round(thr, 4),
                                confidence=round(c1, 4),
                                top2_intent=self.labels[i2], top2_score=round(c2, 4),
                                margin=round(m, 4), ood_score=round(o, 4),
                                corrective=corrective,
                                accepted=accepted, reason=reason))
        return results

    def to_dict(self) -> dict:
        return dict(conf_threshold=self.conf_threshold,
                    conf_by_risk=self.conf_by_risk,
                    risk_of={k: v for k, v in self.risk_of.items()
                             if v != "normal"},
                    margin_threshold=self.margin_threshold,
                    reject_corrective=self.reject_corrective,
                    corrective_pattern=CORRECTIVE_STRUCTURE.pattern,
                    ood_threshold=(None if self.ood_threshold == float("inf")
                                   else self.ood_threshold),
                    ood_percentile=self.ood_percentile,
                    temperature=self.temperature,
                    reject_label=self.reject_label,
                    labels=self.labels)


def select_operating_point_3d(probs: np.ndarray, y: np.ndarray,
                              ood: np.ndarray, ood_grid: list,
                              reject_index: int,
                              target_precision: float = 0.97,
                              min_coverage: float = 0.50,
                              margin_grid=(0.0, 0.05, 0.10, 0.20, 0.30),
                              max_fallback_leak: float = 0.02):
    """Pick (confidence, margin, OOD) together on validation.

    Coverage is measured on the rows that SHOULD be actioned, and the
    fallback-labelled validation rows are scored separately as the rows that
    should be refused. Optimising a single blended number hides the trade-off
    that matters: a gate can always buy precision by refusing everything.
    """
    pred, conf = probs.argmax(1), probs.max(1)
    marg = margin_of(probs)
    correct = pred == y
    is_reject_gold = y == reject_index
    actionable = ~is_reject_gold

    rows = []
    for o_thr in ood_grid:
        for m_thr in margin_grid:
            for c_thr in np.round(np.arange(0.20, 0.995, 0.005), 3):
                acc = ((pred != reject_index) & (ood <= o_thr)
                       & (conf >= c_thr) & (marg >= m_thr))
                n_acc = int(acc.sum())
                if n_acc < 20:
                    continue
                rows.append(dict(
                    conf_threshold=float(c_thr), margin_threshold=float(m_thr),
                    ood_threshold=float(o_thr),
                    precision=float(correct[acc].mean()),
                    coverage=float(acc[actionable].mean()),
                    fallback_leak=float(acc[is_reject_gold].mean())
                    if is_reject_gold.any() else 0.0,
                    n_accepted=n_acc))
    if not rows:
        return dict(conf_threshold=0.5, margin_threshold=0.0,
                    ood_threshold=float("inf"), precision=float("nan"),
                    coverage=1.0, fallback_leak=float("nan"),
                    n_accepted=len(y), target_met=False,
                    target_precision=target_precision)
    ok = [r for r in rows if r["precision"] >= target_precision
          and r["coverage"] >= min_coverage
          and r["fallback_leak"] <= max_fallback_leak]
    if ok:
        best = max(ok, key=lambda r: (r["coverage"], -r["fallback_leak"]))
        best["target_met"] = True
    else:
        # Constraints are relaxed in order of what they promise.
        #
        #   1. the leak cap    — a tuning preference
        #   2. min_coverage    — a wish; the product would LIKE this much
        #   3. nothing else    — target_precision is the promise and is kept
        #
        # min_coverage used to be relaxed LAST, which meant an infeasible
        # request fell straight through to "maximise precision". That branch
        # does exactly what this function's own docstring warns against: a gate
        # can always buy precision by refusing everything. Asking for
        # --min-coverage 0.80 when nothing on the grid reached 0.68 produced
        # precision 0.9955 at coverage 0.299 — the opposite of what was asked
        # for, silently, with only target_met=False to show for it.
        relaxed = [r for r in rows if r["precision"] >= target_precision
                   and r["coverage"] >= min_coverage]
        if relaxed:
            best = min(relaxed, key=lambda r: (r["fallback_leak"], -r["coverage"]))
            best["relaxed"] = "fallback_leak"
        else:
            # Drop min_coverage, keep the precision promise, take the most
            # coverage available. This is what a caller asking for impossible
            # coverage actually wants: as close as the model can get.
            keep_precision = [r for r in rows
                              if r["precision"] >= target_precision]
            if keep_precision:
                best = max(keep_precision,
                           key=lambda r: (r["coverage"], -r["fallback_leak"]))
                best["relaxed"] = "fallback_leak+min_coverage"
            else:
                # Nothing reaches the precision target at all. Report the
                # highest-coverage point that gets closest, and make the
                # shortfall loud — this is a model problem, not a tuning one.
                near = max(r["precision"] for r in rows)
                best = max([r for r in rows if r["precision"] >= near - 0.005],
                           key=lambda r: r["coverage"])
                best["relaxed"] = "everything"
                print(f"  WARNING: no operating point reaches precision "
                      f"{target_precision}. Best available is "
                      f"{best['precision']:.4f} at coverage "
                      f"{best['coverage']:.4f}. The gate cannot fix this; the "
                      f"model cannot separate these classes well enough.")
        best["target_met"] = False
        print(f"  note: constraints infeasible, relaxed {best['relaxed']} "
              f"-> coverage {best['coverage']:.4f} "
              f"precision {best['precision']:.4f}")
    best["target_precision"] = target_precision
    best["max_fallback_leak"] = max_fallback_leak
    return best


def ood_ablation(probs, y, ood, ood_grid, reject_index,
                 target_precision=0.97, margin_grid=(0.0,)):
    """What each OOD cut-off actually buys, at matched in-domain precision.

    Without this the OOD threshold looks free — it is not. It trades coverage
    (real commands the user must repeat) for fewer unsupported requests being
    executed, and only a person who knows the product can price that.
    """
    rows = []
    for o_thr in ood_grid:
        cand = select_operating_point_3d(probs, y, ood, [o_thr], reject_index,
                                         target_precision, min_coverage=0.0,
                                         margin_grid=margin_grid,
                                         max_fallback_leak=1.0)
        rows.append(dict(ood_threshold=("off" if not np.isfinite(o_thr)
                                        else round(float(o_thr), 3)),
                         conf=cand["conf_threshold"],
                         coverage=round(cand["coverage"], 4),
                         precision=round(cand["precision"], 4),
                         fallback_leak=round(cand["fallback_leak"], 4)))
    return rows


def fit_per_risk_thresholds(probs, y, risk_of: dict, labels: list,
                            base_threshold: float,
                            targets: dict | None = None) -> dict:
    """One confidence threshold per risk tier, fitted on validation.

    Fitted on the rows the model PREDICTS into that tier, because that is the
    population the threshold will actually gate at runtime. Gating on the gold
    tier would measure a set the runtime never sees.

    A tier's threshold is never allowed below the shared base: the tiers exist
    to be stricter where a mistake costs more, not to buy back coverage.
    """
    targets = targets or {"high": 0.99, "normal": 0.97}
    pred, conf = probs.argmax(1), probs.max(1)
    correct = pred == y
    pred_tier = np.array([risk_of.get(labels[i], "normal") for i in pred])

    def fit_one(tier: str, target: float, floor: float) -> float:
        m = pred_tier == tier
        if m.sum() < 20:
            return floor
        for c_thr in np.round(np.arange(floor, 0.999, 0.005), 3):
            acc = m & (conf >= c_thr)
            if acc.sum() < 10:
                break
            if correct[acc].mean() >= target:
                return float(c_thr)
        # No threshold in range reaches the target. Returning the ceiling is
        # not "strict" — it is the gate giving up, and at 0.995 it will accept
        # almost nothing. Silently returning it makes a broken model look like
        # a cautious one.
        print(f"  WARNING: no confidence threshold reaches {target:.2f} "
              f"precision for {tier}-risk intents. Pinning at 0.995, which "
              f"will refuse nearly everything. The model is not good enough "
              f"for this target, not merely uncalibrated.")
        return 0.995

    # `normal` is fitted first and becomes the FLOOR for `high`. Fitting them
    # independently can hand the high-risk tier a lower threshold than the
    # normal one — the model happens to be more accurate on those five intents,
    # so a looser cut-off still clears the stricter precision target. That is
    # arithmetically fine and exactly backwards as a safety policy: a mistake
    # the user cannot notice must never be easier to trigger than one they can.
    out = {"normal": fit_one("normal", targets.get("normal", 0.97),
                             base_threshold)}
    out["high"] = fit_one("high", targets.get("high", 0.99), out["normal"])
    return out
