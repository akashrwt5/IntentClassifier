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

        # A device head, its vocabulary and its temperature are ONE triple.
        # iOS builds the TF-IDF vector in Swift from the weights file, so the
        # pruned head (~1317 features) and the full head (~4718) each need their
        # own vocab AND their own fitted T. Reading a temperature out of a file
        # the pack does not then ship is how they drift apart, so the file
        # travels with the number it contributed.
        for src, dest, key in ((ios_weights, "intent_classifier_weights.json", "temperature_coreml"),
                               (ios_weights_full, "intent_classifier_weights_full.json",
                                "temperature_coreml_full")):
            if src is None or not src.exists():
                continue
            data = json.loads(src.read_text(encoding="utf-8"))
            if "temperature" in data:
                payload[key] = data["temperature"]
            intent_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, intent_dir / dest)
            refreshed.append(f"models/intent/{lang}/{dest}")

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
        cml_dst = staged / "models" / "intent" / lang / dst_name
        cml_dst.parent.mkdir(parents=True, exist_ok=True)
        if cml_dst.exists():
            shutil.rmtree(cml_dst)
        shutil.copytree(cml, cml_dst)
        refreshed.append(f"models/intent/{lang}/{dst_name}")
        entry = manifest.setdefault("models", {}).setdefault("intent", {}).setdefault(lang, {})
        entry[key] = f"models/intent/{lang}/{dst_name}"

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
        tfl_dst = intent_dir / dest
        tfl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tfl, tfl_dst)
        refreshed.append(f"models/intent/{lang}/{dest}")
        entry = manifest.setdefault("models", {}).setdefault("intent", {}).setdefault(lang, {})
        entry[key] = f"models/intent/{lang}/{dest}"

    if report is not None:
        if not report.exists():
            return _fail(f"report card not found: {report}")
        (staged / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy(report, staged / "meta" / "report_card.json")
        refreshed.append("meta/report_card.json")

    # bundle_id and version are ONE decision. This function rewrites bundle_id from
    # `--version` while the source bundle's `version` came from whatever compiled it;
    # leaving that alone ships a pack whose id says 1.2.3 and whose version says
    # something else. Nothing downstream can tell which one is real — iOS names its
    # storage directory from `version`, the OTA backend compares against `version`,
    # and telemetry keys on `bundle_id`.
    manifest["bundle_id"] = f"pack-{lang}-v{version}"
    manifest["version"] = version
    manifest["channel"] = channel
    manifest["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # A non-dev iOS pack carries ONE head, and it is the full one (see mod_ios).
    # If this build never produced it, the slice would ship with no head the
    # device can bind: `BundleDataLoader` defaults to `.full`, so the pack would
    # fail at load with `declaredArtifactMissing` — loud, but on a device instead
    # of in CI. Refuse here, where the person who can fix it is still watching.
    # Only meaningful for a build that HAS CoreML. A pack with no CoreML head at
    # all is a legitimate shape — an ONNX-only build, which is what a plain
    # `assemble_pack` run without --coreml-compiled produces — and the
    # pruned/full split does not exist there, so there is nothing for the channel
    # to choose between and nothing to refuse. Guarding on `channel` alone
    # refused exactly that build, which is what
    # test_key_id_and_channel_are_parameters_not_constants exists to catch: the
    # ND-8 cutover must be a settings change, and a guard that rejects
    # `--channel production` on its own makes it a code change.
    _intent = manifest.get("models", {}).get("intent", {}).get(lang, {})
    _has_coreml = any(_intent.get(k) for k in
                      ("coreml_artifact", "coreml_compiled_artifact",
                       "coreml_full_artifact", "coreml_full_compiled_artifact"))
    if channel != "dev" and _has_coreml:
        _missing = [k for k in ("coreml_full_compiled_artifact",)
                    if not _intent.get(k)]
        if not (staged / "models" / "intent" / lang
                / "intent_classifier_weights_full.json").exists():
            _missing.append("intent_classifier_weights_full.json")
        if _missing:
            return _fail(
                f"channel {channel!r} ships only the full head, and this CoreML "
                f"build is missing {', '.join(_missing)} — the slice would carry "
                f"no head the device can bind. Pass --coreml-compiled with the "
                f"full head and its weights, or build --channel dev.")
    (staged / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")

    def build_slice(suffix: str, mod_func) -> Path:
        staged_slice = out_dir / f"staged_tmp_{suffix}"
        if staged_slice.exists():
            shutil.rmtree(staged_slice)
        shutil.copytree(staged, staged_slice)

        slice_manifest = json.loads((staged_slice / "bundle.json").read_text(encoding="utf-8"))
        mod_func(staged_slice, slice_manifest)
        (staged_slice / "bundle.json").write_text(json.dumps(slice_manifest, indent=2) + "\n", encoding="utf-8")

        nlu_out = out_dir / f"pack-{lang}-v{version}-{suffix}.nlu"
        cmd = [sys.executable, "-m", "nlu_compiler.build", str(staged_slice),
               "--out", str(nlu_out), "--channel", channel]
        if key_id:
            cmd += ["--key-id", key_id]
        env = {"PYTHONPATH": str(BUILDTIME)}
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              env={**__import__("os").environ, **env})
        if proc.returncode != 0:
            print(proc.stdout, proc.stderr, file=sys.stderr)
            if "LABEL_INTENT_MISMATCH" in (proc.stdout + proc.stderr):
                print("\nHINT: LABEL_INTENT_MISMATCH detected.", file=sys.stderr)
            raise RuntimeError(f"nlu_compiler.build failed for {suffix}")

        shutil.rmtree(staged_slice)
        return nlu_out

    def mod_universal(s_dir, s_man):
        pass

    def mod_ios(s_dir, s_man):
        # Whatever this function deletes must also leave the manifest, or the pack
        # declares an artifact it does not ship. `mod_android` below already does
        # both halves; this one used to do only the first, so every iOS pack's
        # bundle.json pointed at a `models/intent/<lang>/model.onnx` that this very
        # function had just unlinked. Nothing caught it on either side: VoiceAIKit
        # has the machinery for the check (`ModelSpec.declaredPaths`,
        # `PackLoadPolicy.toleratedMissingArtifacts`) and never calls it.
        intent_dir = s_dir / "models" / "intent" / lang
        intent = s_man.get("models", {}).get("intent", {}).get(lang, {})

        for t in ["model.tflite", "model_int8.tflite"]:
            f = intent_dir / t
            if f.exists():
                f.unlink()
        for key in ["tflite_artifact", "tflite_int8_artifact"]:
            intent.pop(key, None)

        # The flattened `nlu_schema.json` / `nlu_entities.json` at the bundle root
        # have no iOS consumer. They are the REFERENCE ENGINE's input — Python
        # reads `language_packs/<lang>/nlu_schema.json`, and `compile_models`
        # copies them into the bundle to "expose schema and entities at the
        # bundle root (ADR-005)", which is an archive concern, not a device one.
        #
        # Swift reads the compiled tables instead: `runtime/policies.json`,
        # `runtime/cascade.json`, `keywords/<lang>.json`, `lexicons/<lang>.json`
        # and the per-capability directories. Every mention of these two files in
        # VoiceAIKit is a comment about what a type USED to read before the pack
        # format existed — `DialogSchema`, `PackSlotResolver` and
        # `PackEntityExtractor` all say so at their declarations. There is no live
        # read.
        #
        # Nothing enforces their presence either: neither is in the validator's
        # REQUIRED_FILES, neither is declared in `bundle.json`, and
        # `nlu_langpack`'s `_RUNTIME_TABLES` is (cascade, policies, plan_facts).
        # They stay in the universal slice, which is the archive.
        for flat in ("nlu_schema.json", "nlu_entities.json"):
            flat_path = s_dir / flat
            if flat_path.exists():
                flat_path.unlink()

        # A Python pickle has no iOS consumer. The labels iOS reads are
        # `labels.json`, which `compile_models` DERIVES from this file at compile
        # time precisely so the two cannot disagree. Shipping the source pickle to
        # a device puts a deserialization-primitive artifact in a signed medical
        # bundle for no runtime benefit.
        pkl = intent_dir / "labels.pkl"
        if pkl.exists():
            pkl.unlink()

        # WHICH HEADS RIDE ALONG IS THE CHANNEL'S DECISION.
        #
        # A pack carries two CoreML heads: the RFE-pruned one (~1317 features,
        # 88.57% on the honest holdout) and the full-vocabulary one (~4718,
        # 90.20%). `BundleDataLoader` defaults to `.full` and no production path
        # passes a variant, so the pruned head is loaded by nobody outside an A/B
        # experiment or a test.
        #
        # dev   — both, so a variant can be flipped without republishing.
        # beta,
        # production — the full head only. Production cannot be fatter than beta,
        #              so the same rule covers both.
        #
        # This rides on `channel` rather than a new field on purpose. `channel` is
        # a TRUST axis (ADR-005 Part 11: channel + signing-key id is what lets a
        # production runtime refuse dev-signed artifacts), so overloading it does
        # cost something: there is now no way to build "dev channel, production
        # contents" to isolate whether a field bug comes from the artifact set.
        # Accepted deliberately — the alternative is a second axis that every
        # reader has to keep straight, and `bundle.json` is
        # `additionalProperties: false`, so it is a spec change either way. If
        # that debug need ever arrives, it arrives as an explicit override flag.
        #
        # The pruned head is a TRIPLE, not a file: the `.mlmodelc`, its own
        # vocabulary/idf in `intent_classifier_weights.json`, and its own fitted
        # `temperature_coreml`. VoiceAIKit binds all three together or throws,
        # because "mixing legs produces a shape mismatch at best and
        # plausible-looking wrong confidences at worst". So all three go, or none.
        # Conditional on the FULL head existing, not just on the channel: with no
        # full head there is nothing to strip down to, and removing the pruned one
        # would leave the slice with no CoreML at all. The guard above already
        # refuses that combination for a CoreML build; this keeps a non-CoreML
        # build untouched.
        if channel != "dev" and intent.get("coreml_full_compiled_artifact"):
            for name in ("IntentClassifier.mlmodelc", "IntentClassifier.mlpackage"):
                path = intent_dir / name
                if path.exists():
                    shutil.rmtree(path)
            for key in ("coreml_compiled_artifact", "coreml_artifact"):
                intent.pop(key, None)
            pruned_weights = intent_dir / "intent_classifier_weights.json"
            if pruned_weights.exists():
                pruned_weights.unlink()
            # The third leg. Optional in calibration.schema.json, so removing it
            # is legal — and leaving a temperature behind for a head that is not
            # in the pack is exactly the drift this slice function exists to stop.
            cal_path = intent_dir / "calibration.json"
            if cal_path.exists():
                cal = json.loads(cal_path.read_text(encoding="utf-8"))
                if cal.pop("temperature_coreml", None) is not None:
                    cal_path.write_text(json.dumps(cal, sort_keys=True,
                                                   separators=(",", ":")) + "\n",
                                        encoding="utf-8")

        # `.mlpackage` is the SOURCE form. iOS never opens it when the compiled
        # form is present: `ModelSpec.iOSModel(_:)` returns
        # `coreml_compiled_artifact` first, and `MLModel(contentsOf:)` requires
        # `.mlmodelc` while `compileModel(at:)` REJECTS one — so the two keys are
        # not interchangeable and the packaged form is dead weight in a device
        # slice. Together they are ~1.7 MB of a 7.1 MB pack.
        #
        # Each is dropped only when ITS OWN compiled counterpart shipped, so a
        # slice built without --coreml-compiled keeps the only model it has.
        # `.mlpackage` stays in the universal slice, which is what model tooling
        # and OTA debug read.
        for pkg, pkg_key, compiled_key in (
            ("IntentClassifier.mlpackage", "coreml_artifact",
             "coreml_compiled_artifact"),
            ("IntentClassifier_full.mlpackage", "coreml_full_artifact",
             "coreml_full_compiled_artifact"),
        ):
            if not intent.get(compiled_key):
                continue
            pkg_path = intent_dir / pkg
            if pkg_path.exists():
                shutil.rmtree(pkg_path)
            intent.pop(pkg_key, None)

        # `artifact` and `format` are REQUIRED by bundle.schema.json and are
        # non-optional in VoiceAIKit's `ModelSpec`, so unlike the tflite keys they
        # cannot be dropped — they have to be made TRUE. The format enum already
        # admits `mlmodelc-ref`; it is what ADR-005's bundle layout means by
        # `model.{onnx|mlmodelc-ref}`.
        #
        # LAST, deliberately: everything above may remove a head, and this names
        # the one that survived. Ordering it before the strips is how an earlier
        # revision came to point `artifact` at a pruned `.mlmodelc` that the same
        # function had just deleted — the exact defect this slice was fixed for,
        # reintroduced one channel over.
        #
        # Prefers the FULL head because that is what the runtime binds:
        # `BundleDataLoader` defaults to `.full` and no production path passes a
        # variant. On a dev pack both heads are present and this still names the
        # one that actually gets loaded.
        primary = (intent.get("coreml_full_compiled_artifact")
                   or intent.get("coreml_compiled_artifact"))
        if primary:
            onnx_file = intent_dir / "model.onnx"
            if onnx_file.exists():
                onnx_file.unlink()
            intent["artifact"] = primary
            intent["format"] = "mlmodelc-ref"

    def mod_android(s_dir, s_man):
        intent_dir = s_dir / "models" / "intent" / lang
        intent = s_man.get("models", {}).get("intent", {}).get(lang, {})

        for cml in [
            "IntentClassifier.mlpackage",
            "IntentClassifier.mlmodelc",
            "IntentClassifier_full.mlpackage",
            "IntentClassifier_full.mlmodelc",
        ]:
            cml_path = intent_dir / cml
            if cml_path.exists():
                if cml_path.is_dir():
                    shutil.rmtree(cml_path)
                else:
                    cml_path.unlink()

        for key in ["coreml_artifact", "coreml_compiled_artifact",
                    "coreml_full_artifact", "coreml_full_compiled_artifact"]:
            intent.pop(key, None)

        # The CoreML temperatures live in `calibration.json`, NOT in the manifest
        # entry — `bundle.schema.json` has no `temperature_*` property, so the two
        # keys this used to pop off `intent` were never there and the pop did
        # nothing. The drift stayed in the file: an Android pack shipped
        # `temperature_coreml` and `temperature_coreml_full` for two CoreML heads
        # the same function had just deleted.
        #
        # Both are optional in calibration.schema.json, so removing them is legal.
        # `temperature` (ONNX) and `temperature_int8` (tflite int8) stay, and so
        # does `fitted_on` — the sha256 of the train.csv the fit was measured on,
        # which is the file's provenance and must survive every slice.
        cal_path = intent_dir / "calibration.json"
        if cal_path.exists():
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            dropped = [k for k in ("temperature_coreml", "temperature_coreml_full")
                       if cal.pop(k, None) is not None]
            if dropped:
                cal_path.write_text(json.dumps(cal, sort_keys=True,
                                               separators=(",", ":")) + "\n",
                                    encoding="utf-8")

        for w in ["intent_classifier_weights.json", "intent_classifier_weights_full.json"]:
            w_file = intent_dir / w
            if w_file.exists():
                w_file.unlink()

        # A Python pickle has no Android consumer either. `mod_ios` has removed it
        # since VIK-051 with a rationale that applies here verbatim — the labels
        # Android reads are `labels.json`, which `compile_models` DERIVES from this
        # pickle at compile time precisely so the two cannot disagree — and this
        # slice kept shipping it, so the fix was half done. A deserialization
        # primitive does not belong in a signed medical bundle for no runtime
        # benefit, on either platform.
        pkl = intent_dir / "labels.pkl"
        if pkl.exists():
            pkl.unlink()

    try:
        nlu_universal = build_slice("universal", mod_universal)
        nlu_ios = build_slice("ios", mod_ios)
        nlu_android = build_slice("android", mod_android)
    except RuntimeError:
        return _fail("nlu_compiler.build failed")

    print(f"language     : {lang}")
    print(f"version      : {version}")
    print(f"channel      : {channel}")
    for r in refreshed:
        print(f"refreshed    : {r}")
    print(f"\nassembled universal: {nlu_universal}")
    print(f"assembled ios:       {nlu_ios}")
    print(f"assembled android:   {nlu_android}")
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
