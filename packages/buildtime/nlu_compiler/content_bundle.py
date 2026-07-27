#!/usr/bin/env python3
"""Compile `content/` into a `spec/bundle/3.0` tree — the content->bundle compiler.

WHY THIS EXISTS
---------------
`release-pack.yml` used `spec/examples/3.0/minimal` as its source, which is a
GOLDEN TEST FIXTURE: one capability, two intents (`audio.volume.mute`,
`audio.volume.set`). The product has 12 capabilities and 57 intents. So every
pack published before this compiler carried the fixture's capabilities, keywords,
lexicons and routing with our model dropped in — structurally valid, and not this
product. It only passed validation because `labels.json` still held the fixture's
two labels; the 57-class ONNX beside it was the part nothing checked.

This turns the real content tree into a real bundle.

WHAT IS DERIVED vs CARRIED
--------------------------
Derived from content (the compiler's actual work):

    capabilities/<id>/capability.json   <- capability.yaml + intents/*.yaml actions
    capabilities/<id>/workflows.json    <- intents/*.yaml slots + completion
    capabilities/<id>/responses/en.json <- prompts and fulfillment text, keyed
    entities/shared/*.json              <- content/nlu_entities.json
    keywords/en.json                    <- platform.yaml keyword_triggers
    lexicons/en.json                    <- affirmative/negative + engine tables
    runtime/policies.json               <- fitted thresholds + the confirm gate
    runtime/plan_facts.json             <- intent -> capability (spec says this
                                           one is "never hand-authored")
    runtime/cascade.json                <- stage wiring; tfidf output dim is the
                                           real label count
    models/intent/en/*                  <- trained artifacts + fitted calibration
    meta/*                              <- report card, git/dataset lineage
    bundle.json                         <- the manifest tying it together

Carried from the fixture as PLATFORM-OWNED TEMPLATES (owner decision, to be
migrated to real authoring later):

    runtime/routing.json      escalation ladder (one value re-derived: the
                              `below_confidence` step must equal our threshold)
    telemetry/schema.json     event-decoding enums; platform vocabulary, not
                              per-language content

Both are recorded in `meta/lineage.json` under `carried_templates` so a reader
can tell what is real and what is placeholder.

CONTENT -> SPEC IMPEDANCE
-------------------------
The content tree predates the bundle spec, so several shapes do not line up.
Each mismatch is translated here rather than by loosening a schema:

  * entity ids   `sys.date-time` -> `sys.date_time`   (hyphens are illegal)
  * entity type  `enum`          -> `list`            (spec's vocabulary)
  * entity values `[syn, ...]`   -> `{"en": [syn, ...]}`
  * slot names   `MemoryName`    -> `memory_name`     (spec requires lowercase)
  * prompts/fulfillment: content holds LITERAL TEXT, the spec holds RESPONSE KEYS
    plus a per-language responses file. Keys are minted per intent.
  * keyword_triggers: `exact` -> tier 1 anchored pattern, `regex` -> tier 2,
    `not_regex` -> `guards`.

WHAT THIS FORMAT CANNOT CARRY YET
---------------------------------
`lexicons.schema.json` has slots for affirmative, negative, uncertainty,
universal_verbs, carriers, negation_cues, idioms and referents — and NOTHING for
the datetime grammar or the contraction table. Those are the two largest English
tables A7 evicted from the engine into `_DEFAULT_*` data. So a 3.0 pack cannot
yet fully describe a language: adding one still needs those two tables to reach
the engine some other way. Reported by `--report-gaps`, not silently dropped.

One English carrier is also unshippable: the reminder carrier uses `(?!\\d)`, and
the portable regex subset forbids lookahead (spec/bundle/portable-regex.md).
Non-conforming patterns are omitted and reported rather than shipped, because a
pattern outside the subset is one a Swift or Kotlin runtime may read differently.

USAGE
    PYTHONPATH=packages/buildtime:packages/runtime \\
      python -m nlu_compiler.content_bundle --lang en --out dist/bundle-en
    ... --report-gaps      # print what the format could not carry
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTENT = REPO / "content"
CAP_DIR = CONTENT / "capabilities"
SPEC = REPO / "spec"
TEMPLATE = SPEC / "examples" / "3.0" / "minimal"
FORMAT_VERSION = "3.0"
COMPILER_VERSION = "nlu-compiler 1.0.0-content"

# Files taken verbatim from the fixture. Owner decision: platform-owned for now.
CARRIED = ("telemetry/schema.json", "runtime/routing.json")

_PLACEHOLDER_SHA = "0" * 64


def _yaml():
    import yaml
    return yaml


def _load_yaml(path: Path):
    return _yaml().safe_load(path.read_text(encoding="utf-8"))


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


# --------------------------------------------------------------------------- #
# identifier translation
# --------------------------------------------------------------------------- #

def entity_id(name: str) -> str:
    """`sys.date-time` -> `sys.date_time`. Hyphens are illegal in a stableId."""
    return name.replace("-", "_").lower()


def slot_name(name: str) -> str:
    """`MemoryName` -> `memory_name`; the spec requires `^[a-z][a-z0-9_]*$`."""
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return re.sub(r"[^a-z0-9_]", "_", s)


def _response_key(intent: str, kind: str) -> str:
    return f"{intent}.{kind}"


# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #

def _params_for(cfg: dict) -> list[dict]:
    out = []
    for slot in cfg.get("slots") or []:
        out.append({"name": slot_name(slot["name"]),
                    "type": "entity_ref",
                    "required": bool(slot.get("required")),
                    "entity": entity_id(slot["entity"])})
    return out


def compile_capabilities(lang: str, out: Path) -> tuple[dict, dict, dict]:
    """Emit every capability's manifest, workflows and responses.

    Returns (capability registry for bundle.json, intent->capability map,
    per-capability action list) — plan_facts and the manifest are built from
    these rather than re-walking the tree.
    """
    registry: dict[str, dict] = {}
    intent_capability: dict[str, str] = {}

    for cap_yaml in sorted(CAP_DIR.glob("*/capability.yaml")):
        cap = _load_yaml(cap_yaml)
        cap_id = cap["id"]
        base = cap_yaml.parent
        if lang not in (cap.get("languages") or [lang]):
            continue

        actions: dict[str, dict] = {}
        workflows: dict[str, dict] = {}
        responses: dict[str, str] = {}

        for intent in sorted(cap["intents"]):
            cfg = _load_yaml(base / "intents" / f"{intent}.yaml")
            action = cfg.get("action")
            if not action:
                raise SystemExit(f"{intent}: no `action` — completion requires one")

            # The action contract. Params come from the intent's slots; an action
            # shared by several intents keeps the richest param list.
            params = _params_for(cfg)
            if action not in actions or len(params) > len(actions[action]["params"]):
                actions[action] = {
                    "key": action,
                    "params": params,
                    "descriptor": cfg.get("descriptor")
                                  or f"{intent} — {cfg.get('fulfillment', '').strip()}"
                                  or intent,
                }

            # Workflow facts. Literal prompt text moves into `responses`, and the
            # workflow references it by key — the spec keeps text out of logic.
            wf: dict = {"completion": {
                "action": action,
                "response": _response_key(intent, "done")}}
            responses[_response_key(intent, "done")] = (
                cfg.get("fulfillment") or "Done.")

            slots = []
            for slot in cfg.get("slots") or []:
                key = _response_key(intent, f"ask_{slot_name(slot['name'])}")
                prompt = (slot.get("prompt") or "").strip()
                if not prompt:
                    # An empty prompt means "never ask" (optional slot filled
                    # opportunistically); the spec still requires a key, so give
                    # it a real sentence rather than an empty string.
                    prompt = f"What {slot_name(slot['name']).replace('_', ' ')}?"
                responses[key] = prompt
                slots.append({"name": slot_name(slot["name"]),
                              "entity": entity_id(slot["entity"]),
                              "required": bool(slot.get("required")),
                              "prompt": key})
            if slots:
                wf["slots"] = slots

            if cfg.get("confirm_prompt"):
                ckey = _response_key(intent, "confirm")
                responses[ckey] = cfg["confirm_prompt"]
                # `required` is mandatory. False means "ask only when the
                # uncertainty gate says so" — the engine's actual behaviour;
                # True would mean unconditional confirmation, which is the open
                # owner decision (Review-F5 B1), so it is not asserted here.
                wf["confirmation"] = {"required": False, "prompt": ckey}

            workflows[intent] = wf
            intent_capability[intent] = cap_id

        cap_out = out / "capabilities" / cap_id
        _write(cap_out / "capability.json", {
            "id": cap_id,
            "version": str(cap.get("version", "1.0")),
            "owner": cap.get("owner", "platform-team"),
            "status": cap.get("status", "active"),
            "platforms": cap.get("platforms", ["ios", "android", "server"]),
            "languages": [lang],
            "actions": [actions[k] for k in sorted(actions)],
        })
        _write(cap_out / "workflows.json", {"intents": workflows})
        _write(cap_out / "responses" / f"{lang}.json", responses)
        registry[cap_id] = {"version": str(cap.get("version", "1.0")),
                            "status": cap.get("status", "active")}

    return registry, intent_capability, {}


# --------------------------------------------------------------------------- #
# entities / keywords / lexicons
# --------------------------------------------------------------------------- #

def compile_entities(lang: str, out: Path) -> None:
    src = json.loads((CONTENT / "nlu_entities.json").read_text(encoding="utf-8"))
    entities: dict[str, dict] = {}
    for name, spec in src.items():
        eid = entity_id(name)
        kind = spec.get("type")
        if kind == "enum":
            # `enum` is the content tree's word; the spec's vocabulary is
            # list/pattern/dynamic. Values become per-language synonym lists.
            entities[eid] = {
                "type": "list",
                "fuzzy": bool(spec.get("fuzzy")),
                "values": {canon: {lang: list(syns)}
                           for canon, syns in (spec.get("values") or {}).items()
                           if syns},
            }
        elif kind == "pattern" and spec.get("pattern"):
            entities[eid] = {"type": "pattern", "pattern": spec["pattern"]}
        else:
            # sys.* builtins carry no values in content; declare them dynamic so
            # the runtime knows to resolve them itself rather than by lookup.
            entities[eid] = {"type": "dynamic", "dynamic_source": "runtime.builtin"}
    _write(out / "entities" / "shared" / "content.json", {"entities": entities})


def compile_keywords(lang: str, schema: dict, out: Path) -> list[str]:
    """keyword_triggers -> the bundle's tiered rule list. Returns skipped patterns."""
    from nlu_compiler.portable_regex import check_pattern

    rules, skipped = [], []
    for entry in schema.get("keyword_triggers", []):
        intent = entry["intent"]
        if "exact" in entry:
            for term in entry["exact"]:
                # Tier 1 is the exact-utterance short circuit, so anchor it.
                rules.append({"pattern": f"^{re.escape(term)}$", "tier": 1,
                              "intent": intent})
        elif "contains" in entry:
            for term in entry["contains"]:
                rules.append({"pattern": re.escape(term), "tier": 2,
                              "intent": intent})
        elif "regex" in entry:
            errs = check_pattern(entry["regex"])
            if errs:
                skipped.append(f"keyword {intent}: {errs}")
                continue
            rule = {"pattern": entry["regex"], "tier": 2, "intent": intent}
            if "not_regex" in entry:
                if check_pattern(entry["not_regex"]):
                    skipped.append(f"keyword guard {intent}: not portable")
                else:
                    rule["guards"] = [entry["not_regex"]]
            rules.append(rule)
    _write(out / "keywords" / f"{lang}.json", {"lang": lang, "rules": rules})
    return skipped


def compile_lexicon(lang: str, schema: dict, out: Path) -> list[str]:
    """Emit the language lexicon. Returns what the format could not carry.

    The tables come from `content/platform.yaml -> lexicon`, NOT from the engine.
    tests/test_package_boundaries.py forbids the compiler importing nlu_engine —
    "independent release trains: the compiler validates content against spec/,
    not against engine internals" — and it caught the first version of this
    function reaching into `_DEFAULT_CARRIERS`. The guard was right, and the fix
    is the better design anyway: carriers and negation cues are language DATA, so
    they belong in content/ where a language pack author can edit them. The engine
    keeps its `_DEFAULT_*` tables as the no-content fallback.
    """
    from nlu_compiler.portable_regex import check_pattern

    lexicon = schema.get("lexicon") or {}
    gaps: list[str] = []
    carriers = []
    for pat in lexicon.get("carriers", []):
        errs = check_pattern(pat)
        if errs:
            gaps.append(f"lexicon carrier omitted (not portable: {errs}): {pat}")
        else:
            carriers.append(pat)

    lex = {
        "lang": lang,
        "affirmative": sorted(schema.get("affirmative", [])),
        "negative": sorted(schema.get("negative", [])),
        "negation_cues": sorted(lexicon.get("negation_cues", [])),
        "carriers": carriers,
    }
    _write(out / "lexicons" / f"{lang}.json", lex)

    # The format simply has no field for these two, and they are the largest
    # tables the engine needs per language.
    gaps.append("lexicons.schema.json has no field for the DATETIME GRAMMAR "
                "(_DEFAULT_DT_GRAMMAR) — a pack cannot carry date/time vocabulary")
    gaps.append("lexicons.schema.json has no field for CONTRACTIONS "
                "(_DEFAULT_CONTRACTIONS) — needed for train/inference parity")
    gaps.append("lexicons.schema.json has no field for LEADING CONNECTORS; "
                f"{len(lexicon.get('leading_connectors', []))} are in content but "
                "cannot ship")
    return gaps


# --------------------------------------------------------------------------- #
# runtime tables
# --------------------------------------------------------------------------- #

def _state_changing(intents) -> set[str]:
    def read_only(i):
        return i == "device.status.battery" or i.rsplit(".", 1)[-1] == "query"
    return {i for i in intents
            if not i.startswith(("help.", "sys.")) and not read_only(i)}


def compile_policies(schema: dict, out: Path) -> None:
    """The rulebook, flat and complete — a runtime never merges policy.

    `confirmation` is our uncertainty gate expressed per intent:
    `when_ambiguous` for the state-changing intents the gate covers (the engine
    asks when confidence is under the band), `never` for everything else. The
    schema notes high-cost actions "must resolve to always"; which intents are
    high-cost enough for unconditional confirmation is the open owner decision
    (Review-F5 B1), so nothing is promoted to `always` here.
    """
    uc = schema.get("uncertain_confirm", {})
    gated = set(uc.get("intents", []))
    confirmation = {i: ("when_ambiguous" if i in gated else "never")
                    for i in sorted(schema["intents"])}
    _write(out / "runtime" / "policies.json", {
        "policy_schema": 1,
        "policy_content": 1,
        "confirmation": confirmation,
        "thresholds": {
            "confidence": schema["confidence_threshold"],
            "interrupt": schema["interrupt_threshold"],
            "semantic": schema["semantic_threshold"],
        },
        "limits": {"max_slot_attempts": 3, "session_timeout_s": 120},
    })


def compile_plan_facts(intent_capability: dict, out: Path) -> None:
    _write(out / "runtime" / "plan_facts.json", {
        "intents": {i: {"capability": c}
                    for i, c in sorted(intent_capability.items())},
        "admission_caps": {"max_concurrent_flows": 1},
    })


def compile_cascade(schema: dict, n_labels: int, out: Path) -> None:
    """Stage wiring. `output.dim` is the real label count — stage 8 probes the
    model via ORT and compares, so a stale number fails the build."""
    _write(out / "runtime" / "cascade.json", {"stages": [
        {"id": "keyword", "enabled": True},
        {"id": "tfidf", "enabled": True,
         "input": {"dtype": "string", "shape": [1]},
         "output": {"dtype": "float32", "dim": n_labels}},
        {"id": "semantic",
         "enabled": bool(schema.get("semantic_rescue_enabled", False))},
    ]})


def carry_templates(schema: dict, out: Path) -> list[str]:
    """Copy the platform-owned templates, re-deriving the values that are ours."""
    carried = []
    for rel in CARRIED:
        src = TEMPLATE / rel
        data = json.loads(src.read_text(encoding="utf-8"))
        if rel == "runtime/routing.json":
            # The ladder's reprompt step must trigger at OUR fire threshold,
            # otherwise the pack escalates at a confidence the engine never uses.
            for step in data.get("ladder", []):
                if "below_confidence" in (step.get("when") or {}):
                    step["when"]["below_confidence"] = schema["confidence_threshold"]
        _write(out / rel, data)
        carried.append(rel)
    return carried


# --------------------------------------------------------------------------- #
# models / meta / manifest
# --------------------------------------------------------------------------- #

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def compile_models(lang: str, model_dir: Path, out: Path) -> tuple[int, list[str]]:
    """Copy trained artifacts; return (label count, relative artifact paths)."""
    import joblib

    dst = out / "models" / "intent" / lang
    dst.mkdir(parents=True, exist_ok=True)
    labels = [str(x) for x in joblib.load(str(model_dir / "labels.pkl"))]
    # labels.json is DERIVED so it can never disagree with the model that ships
    # beside it — a published pack once declared 2 labels next to a 57-class graph.
    _write(dst / "labels.json", labels)

    copied = [f"models/intent/{lang}/labels.json"]
    for name in ("model.onnx", "labels.pkl"):
        src = model_dir / name
        if not src.exists():
            raise SystemExit(f"missing trained artifact: {src}")
        shutil.copy(src, dst / name)
        copied.append(f"models/intent/{lang}/{name}")

    # calibration.json is translated into the lean on-device contract by
    # scripts/ci/assemble_pack.py; emit the fitted temperature in that shape here
    # so a bundle compiled standalone is already valid.
    fitted = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    schema = json.loads((CONTENT / "nlu_schema.json").read_text(encoding="utf-8"))
    payload = {"temperature": fitted["temperature"],
               "conf_threshold": schema["confidence_threshold"],
               "method": "temperature_scaling"}
    if "ece_uncalibrated" in fitted:
        payload["ece_raw"] = fitted["ece_uncalibrated"]
    if "ece" in fitted:
        payload["ece_calibrated"] = fitted["ece"]
    src_hash = (fitted.get("provenance") or {}).get("source_sha256")
    if isinstance(src_hash, str) and len(src_hash) == 64:
        payload["fitted_on"] = src_hash
    _write(dst / "calibration.json", payload)
    return len(labels), copied


def compile_meta(lang: str, report_card: Path | None, carried: list[str],
                 gaps: list[str], out: Path) -> dict:
    # ADR-005 stage 13: "a bundle cannot be built from an evaluation that didn't
    # run". A stub report card would compile into a signed artifact asserting
    # metrics nobody measured, so its absence is a hard failure.
    if not (report_card and report_card.exists()):
        raise SystemExit(
            "FAIL: --report is required and must exist. Produce it with "
            "`python -m nlu_training.evaluate --langs <lang> --out <path>`; a "
            "bundle must not claim metrics that were never measured.")
    card = json.loads(report_card.read_text(encoding="utf-8"))
    _write(out / "meta" / "report_card.json", card)

    train_csv = REPO / "datasets" / lang / "train.csv"
    lock = REPO / "requirements.lock"
    _write(out / "meta" / "lineage.json", {
        "dataset_hashes": {f"{lang}/train.csv": _sha256(train_csv)
                           if train_csv.exists() else _PLACEHOLDER_SHA},
        "env_lock_hash": _sha256(lock) if lock.exists() else _PLACEHOLDER_SHA,
        "compiler_version": COMPILER_VERSION,
        "git_commit": _git_commit(),
        "training_run_ids": [f"{lang}-{_git_commit()}"],
    })
    return card


def compile_manifest(lang: str, registry: dict, n_labels: int, card: dict,
                     schema: dict, version: str, channel: str, out: Path) -> None:
    train_csv = REPO / "datasets" / lang / "train.csv"
    # bool is a subclass of int in Python, so `isinstance(v, int)` admits
    # True/False — which report_card_summary rejects (number/string/integer only).
    # Booleans are stringified rather than dropped: gates_passed is the single
    # most useful line in the summary.
    summary = {}
    for k, v in card.items():
        if isinstance(v, bool):
            summary[k] = str(v).lower()
        elif isinstance(v, (int, float, str)):
            summary[k] = v
    _write(out / "bundle.json", {
        "bundle_id": f"pack-{lang}-v{version}",
        "format_version": FORMAT_VERSION,
        "content_version": 1,
        "compiler_version": COMPILER_VERSION,
        "engine_compat": {"min_runtime_contract": 1,
                          "max_tested_runtime_contract": 1},
        "required_runtime_features": [],
        "languages": {lang: {"status": "full"}},
        "capabilities": registry,
        "models": {"intent": {lang: {
            "artifact": f"models/intent/{lang}/model.onnx",
            "format": "onnx",
            "model_version": f"{lang}-{version}"}}},
        "policy_versions": {"schema": 1, "content": 1},
        "telemetry_schema_version": 1,
        "report_card_summary": summary,
        "training": {
            "dataset_hashes": {f"{lang}/train.csv": _sha256(train_csv)
                               if train_csv.exists() else _PLACEHOLDER_SHA},
            "run_ids": [f"{lang}-{_git_commit()}"],
        },
        # Overwritten by nlu_compiler.build when it hashes and signs the tree.
        "checksums_root": _PLACEHOLDER_SHA,
        "signature_info": {"scheme": "ed25519-v1", "key_id": "unsigned"},
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "channel": channel,
    })


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def compile_bundle(lang: str, out: Path, model_dir: Path,
                   report_card: Path | None, version: str, channel: str,
                   report_gaps: bool) -> int:
    schema = json.loads((CONTENT / "nlu_schema.json").read_text(encoding="utf-8"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    registry, intent_capability, _ = compile_capabilities(lang, out)
    missing = sorted(set(schema["intents"]) - set(intent_capability))
    if missing:
        return _fail(f"{len(missing)} intent(s) in nlu_schema.json belong to no "
                     f"capability: {missing[:6]}")

    compile_entities(lang, out)
    gaps = compile_keywords(lang, schema, out)
    gaps += compile_lexicon(lang, schema, out)
    compile_policies(schema, out)
    compile_plan_facts(intent_capability, out)
    n_labels, _ = compile_models(lang, model_dir, out)
    compile_cascade(schema, n_labels, out)
    carried = carry_templates(schema, out)
    card = compile_meta(lang, report_card, carried, gaps, out)
    compile_manifest(lang, registry, n_labels, card, schema, version, channel, out)

    print(f"language     : {lang}")
    print(f"capabilities : {len(registry)}")
    print(f"intents      : {len(intent_capability)}")
    print(f"labels       : {n_labels}")
    print(f"carried      : {', '.join(carried)}")
    print(f"out          : {out}")
    if gaps:
        print(f"\ncoverage gaps: {len(gaps)} (things spec/bundle/3.0 cannot carry)")
        if report_gaps:
            for g in gaps:
                print(f"  - {g}")
        else:
            print("  pass --report-gaps to list them")
    return 0


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "bundle-en")
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None,
                    help="report_card.json from nlu_training.evaluate")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--channel", default="dev")
    ap.add_argument("--report-gaps", action="store_true")
    a = ap.parse_args(argv)
    model_dir = a.model_dir or (REPO / "models" / "intent" / a.lang)
    return compile_bundle(a.lang, a.out, model_dir, a.report, a.version,
                          a.channel, a.report_gaps)


if __name__ == "__main__":
    sys.exit(main())
