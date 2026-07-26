#!/usr/bin/env python3
"""
Assemble a versioned, signed single-language `.nlu` for release.

Takes an unpacked `spec/bundle/3.0` bundle, refreshes it with freshly trained
artifacts, stamps a release version, and hands it to the compiler to package and
sign. The output is a Language Pack in the only sense this repo recognises: a
bundle declaring exactly one language (ADR-005 Part 11 — a packaging profile of
the same format, not a new one).

WHAT THIS IS NOT
----------------
This is NOT a content->bundle ingestion pipeline. Compiling `content/` into an
unpacked bundle tree is the full compiler's job and is still partly unbuilt
(EXECUTION_STATUS: "transforms/ingestion arrive with the full compiler"). Until
then the input is an existing unpacked bundle and this script refreshes the
parts a release actually changes: the model artifacts, the report card, the
version and the channel.

SIGNING
-------
`--key-id` and `--channel` are passed straight through to the compiler. They are
parameters precisely so the ND-8 cutover to production signing is a change to
the workflow inputs, not to any code. Today the dev key is the default and
production runtimes correctly refuse dev-signed artifacts — that refusal IS the
gate, not a bug to work around.

USAGE
    python scripts/ci/assemble_pack.py --src spec/examples/3.0/minimal \\
        --version 1.0.3 --out dist
    python scripts/ci/assemble_pack.py --src ... --version 1.0.3 --out dist \\
        --model models/intent_model.onnx --report dist/report_card.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUILDTIME = REPO / "packages" / "buildtime"

_SEMVER = re.compile(r"^\d+\.\d+\.\d+([-+].+)?$")


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def assemble(src: Path, version: str, out_dir: Path, *,
             language: str | None = None,
             model: Path | None = None,
             labels: Path | None = None,
             weights: Path | None = None,
             calibration: Path | None = None,
             coreml: Path | None = None,
             report: Path | None = None,
             key_id: str | None = None,
             channel: str = "dev") -> int:
    if not _SEMVER.match(version):
        return _fail(f"--version {version!r} is not semver (e.g. 1.0.3)")
    if not (src / "bundle.json").exists():
        return _fail(f"no bundle.json under {src}")

    out_dir.mkdir(parents=True, exist_ok=True)
    staged = out_dir / f"pack-{version}-staged"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(src, staged)

    manifest = json.loads((staged / "bundle.json").read_text(encoding="utf-8"))

    # A release pack declares exactly ONE language. Narrowing here rather than
    # assuming the source bundle is already single-language keeps the multi-
    # language golden bundles usable as input.
    languages = manifest.get("languages", {})
    if language:
        if language not in languages:
            return _fail(f"bundle declares {sorted(languages)}; {language!r} is not among them")
        manifest["languages"] = {language: languages[language]}
    elif len(languages) != 1:
        return _fail(f"bundle declares {len(languages)} languages "
                     f"({sorted(languages)}); pass --language to pick one")
    lang = language or next(iter(manifest["languages"]))

    # Refresh the artifacts a release actually changes. Each is optional so the
    # script is usable in a dry run without a trained model present.
    refreshed: list[str] = []
    intent_dir = staged / "models" / "intent" / lang
    for artifact, dest in ((model, "model.onnx"), (labels, "labels.pkl"),
                           (weights, "weights.json")):
        if artifact is None:
            continue
        if not artifact.exists():
            return _fail(f"artifact not found: {artifact}")
        intent_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(artifact, intent_dir / dest)
        refreshed.append(f"models/intent/{lang}/{dest}")

    # calibration.json travels WITH the model on purpose. Confidence is
    # softmax(logits / T), so a pack shipped without its T falls back to T = 1.0
    # (plain softmax) and mis-tunes the fire threshold, the confirm band and slot
    # acceptance all at once — blocker B8 in a new place.
    #
    # It is TRANSLATED, not copied. The build artifact written by
    # nlu_training.fit_calibration is deliberately richer (full fit provenance,
    # excluded eval sets, fitter identity); the bundle form is the lean on-device
    # contract in spec/bundle/3.0/calibration.schema.json, which forbids
    # additional properties and requires conf_threshold. Copying the build file
    # in raw fails stage-1 validation.
    if calibration is not None:
        if not calibration.exists():
            return _fail(f"artifact not found: {calibration}")
        fitted = json.loads(calibration.read_text(encoding="utf-8"))
        schema_path = REPO / "content" / "nlu_schema.json"
        conf_threshold = json.loads(
            schema_path.read_text(encoding="utf-8")).get("confidence_threshold", 0.70)
        payload = {
            "temperature": fitted["temperature"],
            # The fire threshold ships with the temperature it is expressed in:
            # a runtime that has one without the other cannot reproduce a gate.
            "conf_threshold": conf_threshold,
            "method": "temperature_scaling",
        }
        if "ece_uncalibrated" in fitted:
            payload["ece_raw"] = fitted["ece_uncalibrated"]
        if "ece" in fitted:
            payload["ece_calibrated"] = fitted["ece"]
        # `fitted_on` exists for the leakage audit — it is the hash of the data
        # the temperature was fit on, which is what makes a stale T detectable.
        src_hash = (fitted.get("provenance") or {}).get("source_sha256")
        if isinstance(src_hash, str) and len(src_hash) == 64:
            payload["fitted_on"] = src_hash
        intent_dir.mkdir(parents=True, exist_ok=True)
        (intent_dir / "calibration.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        refreshed.append(f"models/intent/{lang}/calibration.json")

    # CoreML is deliberately NOT packaged into the signed bundle. Two reasons,
    # both blocking:
    #
    # 1. spec/bundle/3.0 cannot express it. `models` is a closed set of STAGES
    #    (intent / embedder / semantic_head) and `modelLangMap` allows exactly one
    #    artifact per language, so a pack carries ONE intent-model format. Writing
    #    models.coreml.<lang> fails stage-1 validation ("Additional properties are
    #    not allowed ('coreml' was unexpected)"), and the files inside a
    #    .mlpackage DIRECTORY have no schema mapping either ("UNMAPPED_FILE
    #    models/coreml/en/IntentClassifier.mlpackage/Manifest.json"). `format`
    #    already lists "mlmodelc-ref", so the spec anticipates CoreML as a FORMAT
    #    of the intent stage — supporting both at once is a spec change (ADR-005),
    #    not something to force past the validator.
    #
    # 2. Even if it validated, the .mlpackage currently produced by
    #    nlu_export.export_coreml is derived from the repo-committed DEVICE
    #    weights, not from the ONNX model in this pack (see release-pack.yml).
    #    Shipping both inside one signed artifact would assert they correspond
    #    when they do not — worse than omitting it.
    #
    # The .mlpackage is still published as a workflow artifact for iOS to consume.
    if coreml is not None:
        return _fail(
            "--coreml cannot be packaged: spec/bundle/3.0 allows one intent-model "
            "artifact per language (models.<stage>.<lang>), so a second format is "
            "not expressible, and the current .mlpackage derives from stale device "
            "weights rather than this pack's ONNX. Publish it as a separate "
            "artifact instead.")

    if report is not None:
        if not report.exists():
            return _fail(f"report card not found: {report}")
        (staged / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy(report, staged / "meta" / "report_card.json")
        refreshed.append("meta/report_card.json")

    manifest["bundle_id"] = f"pack-{lang}-v{version}"
    manifest["channel"] = channel
    manifest["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (staged / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")

    # Package + sign via the compiler — the single path that produces a .nlu.
    nlu_out = out_dir / f"pack-{lang}-v{version}.nlu"
    cmd = [sys.executable, "-m", "nlu_compiler.build", str(staged),
           "--out", str(nlu_out), "--channel", channel]
    if key_id:
        cmd += ["--key-id", key_id]
    env = {"PYTHONPATH": str(BUILDTIME)}
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env={**__import__("os").environ, **env})
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, file=sys.stderr)
        return _fail("nlu_compiler.build failed")

    print(f"language     : {lang}")
    print(f"version      : {version}")
    print(f"channel      : {channel}")
    for r in refreshed:
        print(f"refreshed    : {r}")
    print(proc.stdout.strip())
    print(f"\nassembled: {nlu_out.relative_to(REPO) if nlu_out.is_relative_to(REPO) else nlu_out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="unpacked spec/bundle/3.0 directory to release from")
    ap.add_argument("--version", required=True, help="semver, e.g. 1.0.3")
    ap.add_argument("--out", type=Path, default=REPO / "dist")
    ap.add_argument("--language", default=None,
                    help="language to release (required if src has several)")
    ap.add_argument("--model", type=Path, default=None, help="trained intent ONNX")
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--calibration", type=Path, default=None,
                    help="fitted calibration.json — the temperature MUST ship "
                         "with the model it calibrates")
    ap.add_argument("--coreml", type=Path, default=None, help=".mlpackage for iOS")
    ap.add_argument("--report", type=Path, default=None, help="report_card.json")
    ap.add_argument("--key-id", default=None,
                    help="signing key id (default: the compiler's dev key)")
    ap.add_argument("--channel", default="dev",
                    choices=["dev", "beta", "production"])
    a = ap.parse_args(argv)
    return assemble(a.src, a.version, a.out, language=a.language, model=a.model,
                    labels=a.labels, weights=a.weights,
                    calibration=a.calibration, coreml=a.coreml,
                    report=a.report, key_id=a.key_id, channel=a.channel)


if __name__ == "__main__":
    sys.exit(main())
