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
             coreml_compiled: Path | None = None,
             coreml_full: Path | None = None,
             coreml_full_compiled: Path | None = None,
             tflite: Path | None = None,
             tflite_int8: Path | None = None,
             ios_weights: Path | None = None,
             ios_weights_full: Path | None = None,
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

    # The manifest's model_version otherwise stays whatever the SOURCE bundle
    # said — the published pack-en-v1.0.0 declared "golden-en-1", the golden
    # fixture's placeholder, so a consumer could not tell which model it held.
    if model is not None:
        entry = manifest.setdefault("models", {}).setdefault("intent", {}).setdefault(lang, {})
        entry["model_version"] = f"{lang}-{version}"

    # labels.json is DERIVED from labels.pkl, never carried over from the source
    # bundle. The golden `minimal` tree ships a 2-entry placeholder
    # (["audio.volume.mute", "audio.volume.set"] — still in the superseded
    # `audio.*` naming), and because only labels.pkl was refreshed, a published
    # pack declared 2 labels beside a 57-class ONNX graph. iOS reads labels.json
    # to map output indices, so every prediction would have been mislabelled.
    # Deriving it here makes the two physically incapable of disagreeing.
    if labels is not None:
        import joblib
        names = [str(x) for x in joblib.load(str(labels))]
        intent_dir.mkdir(parents=True, exist_ok=True)
        (intent_dir / "labels.json").write_text(
            json.dumps(names, indent=2) + "\n", encoding="utf-8")
        refreshed.append(f"models/intent/{lang}/labels.json")

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
        schema_path = REPO / "language_packs" / lang / "nlu_schema.json"
        conf_threshold = json.loads(
            schema_path.read_text(encoding="utf-8")).get("confidence_threshold", 0.70)
        payload = {
            "temperature": fitted["temperature"],
            # The fire threshold ships with the temperature it is expressed in:
            # a runtime that has one without the other cannot reproduce a gate.
            "conf_threshold": conf_threshold,
            "method": "temperature_scaling",
        }
        if "temperature_int8" in fitted:
            payload["temperature_int8"] = fitted["temperature_int8"]

        if ios_weights is not None and ios_weights.exists():
            ios_data = json.loads(ios_weights.read_text(encoding="utf-8"))
            if "temperature" in ios_data:
                payload["temperature_coreml"] = ios_data["temperature"]

        if ios_weights_full is not None and ios_weights_full.exists():
            ios_full_data = json.loads(ios_weights_full.read_text(encoding="utf-8"))
            if "temperature" in ios_full_data:
                payload["temperature_coreml_full"] = ios_full_data["temperature"]

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

    # CoreML rides INSIDE the signed bundle (the "Fat Bundle"). It is expressed
    # as models.intent.<lang>.coreml_artifact — a schema-legal SIBLING of the
    # ONNX `artifact`, NOT a second model stage. `models` stays the closed set
    # (intent / embedder / semantic_head), so this does not reopen the
    # models.coreml.<lang> path the schema rejects; it fills the `coreml_artifact`
    # slot the bundle schema already declares. The .mlpackage DIRECTORY is copied
    # into the staging tree here, so its files are checksummed and signed with the
    # rest of the pack rather than shipped as a loose side artifact.
    #
    # CAVEAT (see release-pack.yml): the .mlpackage that nlu_export.export_coreml
    # currently emits is derived from the repo-committed DEVICE weights, not from
    # the ONNX trained in this run. Until the exporter is retargeted, treat the
    # bundled CoreML model as an iOS convenience artifact, not proof of parity
    # with the ONNX graph beside it.
    # Two CoreML heads may ride in the bundle, both derived from the same trained
    # pipeline (see export_ios_weights.py): the default top-per-class PRUNED head
    # (small, on-device default) and the optional FULL-vocab head that matches the
    # ONNX/TFLite feature space. Each is a .mlpackage DIRECTORY (copytree), carried
    # as a sibling reference on the intent entry, never a separate model stage.
    for cml, dst_name, key in (
        (coreml, "IntentClassifier.mlpackage", "coreml_artifact"),
        (coreml_compiled, "IntentClassifier.mlmodelc", "coreml_compiled_artifact"),
        (coreml_full, "IntentClassifier_full.mlpackage", "coreml_full_artifact"),
        (coreml_full_compiled, "IntentClassifier_full.mlmodelc", "coreml_full_compiled_artifact"),
    ):
        if cml is None:
            continue
        if not cml.exists():
            return _fail(f"coreml artifact not found: {cml}")
        cml_dst = staged / "models" / "intent" / lang / "iOS" / dst_name
        cml_dst.parent.mkdir(parents=True, exist_ok=True)
        if cml_dst.exists():
            shutil.rmtree(cml_dst)
        shutil.copytree(cml, cml_dst)
        refreshed.append(f"models/intent/{lang}/iOS/{dst_name}")
        entry = manifest.setdefault("models", {}).setdefault("intent", {}).setdefault(lang, {})
        entry[key] = f"models/intent/{lang}/iOS/{dst_name}"

    # TFLite rides in the bundle the same way CoreML does: as a sibling
    # reference on the intent entry (tflite_artifact / tflite_int8_artifact), NOT
    # a new model stage. It is the HEAD only (float TF-IDF vector -> logits);
    # vectorisation stays native on-device, so this is a single flat file, not a
    # directory like the .mlpackage.
    for tfl, dest, key in ((tflite, "model.tflite", "tflite_artifact"),
                           (tflite_int8, "model_int8.tflite", "tflite_int8_artifact")):
        if tfl is None:
            continue
        if not tfl.exists():
            return _fail(f"tflite artifact not found: {tfl}")
        tfl_dst = intent_dir / "tflite" / dest
        tfl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tfl, tfl_dst)
        refreshed.append(f"models/intent/{lang}/tflite/{dest}")
        entry = manifest.setdefault("models", {}).setdefault("intent", {}).setdefault(lang, {})
        entry[key] = f"models/intent/{lang}/tflite/{dest}"

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
        # Stage 8 compares labels.json against the intent set compiled from the
        # source bundle's capabilities. A mismatch here is almost never a labels
        # problem — it means the SOURCE BUNDLE is not this product's content, and
        # the dump is 57 names long, so say what it means.
        if "LABEL_INTENT_MISMATCH" in (proc.stdout + proc.stderr):
            print(
                "\nHINT: the trained model's label space does not match the intent "
                "set compiled from --src. If --src is spec/examples/3.0/minimal or "
                "full, that is expected: those are GOLDEN TEST FIXTURES (1-2 "
                "capabilities, 2 intents), not this product's content, which has "
                "12 capabilities and 57 intents under content/capabilities/. "
                "Building a real pack needs a content->bundle compiler "
                "(content/capabilities/ -> spec/bundle/3.0 tree); none exists yet. "
                "Until it does, a pack assembled from a fixture proves the "
                "pipeline, not the product.", file=sys.stderr)
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
    ap.add_argument("--coreml", type=Path, default=None,
                    help="pruned-vocab .mlpackage for iOS (small on-device default)")
    ap.add_argument("--coreml-compiled", type=Path, default=None,
                    help="pruned-vocab .mlmodelc for iOS (compiled)")
    ap.add_argument("--coreml-full", type=Path, default=None,
                    help="full-vocab .mlpackage (matches ONNX/TFLite; optional, larger)")
    ap.add_argument("--coreml-full-compiled", type=Path, default=None,
                    help="full-vocab .mlmodelc (compiled)")
    ap.add_argument("--tflite", type=Path, default=None,
                    help="fp32 model.tflite head (float TF-IDF vector -> logits)")
    ap.add_argument("--tflite-int8", type=Path, default=None,
                    help="dynamic-range int8 model_int8.tflite head (optional)")
    ap.add_argument("--ios-weights", type=Path, default=None,
                    help="CoreML specific intent classifier weights JSON containing temperature_coreml")
    ap.add_argument("--ios-weights-full", type=Path, default=None,
                    help="CoreML full intent classifier weights JSON containing temperature_coreml_full")
    ap.add_argument("--report", type=Path, default=None, help="report_card.json")
    ap.add_argument("--key-id", default=None,
                    help="signing key id (default: the compiler's dev key)")
    ap.add_argument("--channel", default="dev",
                    choices=["dev", "beta", "production"])
    a = ap.parse_args(argv)
    return assemble(a.src, a.version, a.out, language=a.language, model=a.model,
                    labels=a.labels, weights=a.weights,
                    calibration=a.calibration, coreml=a.coreml,
                    coreml_compiled=a.coreml_compiled,
                    coreml_full=a.coreml_full,
                    coreml_full_compiled=a.coreml_full_compiled,
                    tflite=a.tflite, tflite_int8=a.tflite_int8,
                    ios_weights=a.ios_weights,
                    ios_weights_full=a.ios_weights_full,
                    report=a.report, key_id=a.key_id, channel=a.channel)


if __name__ == "__main__":
    sys.exit(main())
