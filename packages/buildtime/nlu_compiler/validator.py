"""Shared validator library — compiler stages 1–10 (v0) per ADR-005 Part 5.

Validates an UNPACKED bundle directory (the golden-bundle layout). This is
the single validation path: the compiler, CI, and the Training Studio import
these functions; nothing reimplements them.

Stage coverage in v0 (validation-only stages; transforms come with the full
compiler):
  1  schema validation (JSON Schema, spec/bundle/{format})
  2  vocabulary check (closed enums — enforced by the schemas in 3.0)
  3  reference resolution (slots→entities, prompts→responses, actions,
     keyword intents, plan_facts, superseded_by)
  4  dependency rules (no cross-capability references; orphan shared entities)
  5  policy completeness (confirmation matrix ≡ intent set; high-cost ⇒ always)
  8  cross-artifact parity (labels ≡ intents; head labels/embedder_id;
     calibration presence)  [model-file probing joins with stage 7 ingestion]
  9  localization completeness (responses/lexicons/keywords per language)
 10  portable regex subset
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .diagnostics import Diagnostic, Severity
from .portable_regex import check_pattern

SPEC_ROOT = Path(__file__).resolve().parents[3] / "spec"

FILE_SCHEMA_MAP = [
    (r"^bundle\.json$", "bundle.schema.json"),
    (r"^models/intent/[a-z]{2}/labels\.json$", "labels.schema.json"),
    (r"^models/intent/[a-z]{2}/calibration\.json$", "calibration.schema.json"),
    (r"^models/semantic_head/[a-z]{2}/head\.json$", "semantic_head.schema.json"),
    (r"^runtime/cascade\.json$", "cascade.schema.json"),
    (r"^runtime/policies\.json$", "policies.schema.json"),
    (r"^runtime/routing\.json$", "routing.schema.json"),
    (r"^runtime/plan_facts\.json$", "plan_facts.schema.json"),
    (r"^capabilities/[^/]+/capability\.json$", "capability.schema.json"),
    (r"^capabilities/[^/]+/workflows\.json$", "workflows.schema.json"),
    (r"^capabilities/[^/]+/entities\.json$", "entities.schema.json"),
    (r"^capabilities/[^/]+/responses/[a-z]{2}\.json$", "responses.schema.json"),
    (r"^entities/shared/[^/]+\.json$", "entities.schema.json"),
    (r"^lexicons/[a-z]{2}\.json$", "lexicons.schema.json"),
    (r"^keywords/[a-z]{2}\.json$", "keywords.schema.json"),
    (r"^telemetry/schema\.json$", "telemetry.schema.json"),
    (r"^meta/report_card\.json$", "report_card.schema.json"),
    (r"^meta/lineage\.json$", "lineage.schema.json"),
]

REQUIRED_FILES = ["bundle.json", "runtime/policies.json", "runtime/plan_facts.json",
                  "telemetry/schema.json"]


def _load(bundle: Path, rel: str):
    return json.loads((bundle / rel).read_text(encoding="utf-8"))


def _bundle_files(bundle: Path) -> list[str]:
    return sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*.json"))


# --------------------------------------------------------------------------
# Stage 1+2 — schema validation (closed vocabularies live in the schemas)
# --------------------------------------------------------------------------

def _schema_registry(fmt: str):
    from referencing import Registry, Resource

    resources = []
    for f in (SPEC_ROOT / "bundle" / fmt).glob("*.schema.json"):
        schema = json.loads(f.read_text())
        resources.append((schema["$id"], Resource.from_contents(schema)))
        resources.append((f.name, Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def stage_1_schemas(bundle: Path, fmt: str = "3.0") -> list[Diagnostic]:
    import jsonschema

    registry = _schema_registry(fmt)
    diags: list[Diagnostic] = []
    for rel in REQUIRED_FILES:
        if not (bundle / rel).exists():
            diags.append(Diagnostic(1, "FILE_MISSING", rel, "required bundle file absent"))
    for rel in _bundle_files(bundle):
        matches = [s for pat, s in FILE_SCHEMA_MAP if re.match(pat, rel)]
        if len(matches) != 1:
            diags.append(Diagnostic(1, "UNMAPPED_FILE", rel,
                                    "file has no (unique) schema mapping — unknown files do not compile"))
            continue
        schema = json.loads((SPEC_ROOT / "bundle" / fmt / matches[0]).read_text())
        validator = jsonschema.Draft202012Validator(schema, registry=registry)
        try:
            instance = _load(bundle, rel)
        except json.JSONDecodeError as exc:
            diags.append(Diagnostic(1, "MALFORMED_JSON", rel, str(exc)))
            continue
        for err in validator.iter_errors(instance):
            diags.append(Diagnostic(1, "SCHEMA_INVALID", rel, err.message))
    return diags


# --------------------------------------------------------------------------
# Stage 3+4 — reference resolution and dependency rules
# --------------------------------------------------------------------------

def stage_3_references(bundle: Path) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    manifest = _load(bundle, "bundle.json")

    shared_entities: set[str] = set()
    for f in bundle.glob("entities/shared/*.json"):
        shared_entities |= set(_load(bundle, f.relative_to(bundle).as_posix())["entities"])

    all_intents: set[str] = set()
    caps: dict[str, dict] = {}
    used_shared: set[str] = set()

    for cap_dir in sorted(bundle.glob("capabilities/*")):
        cid = cap_dir.name
        cap = _load(bundle, f"capabilities/{cid}/capability.json")
        wf = _load(bundle, f"capabilities/{cid}/workflows.json")
        local_entities = set()
        ef = cap_dir / "entities.json"
        if ef.exists():
            local_entities = set(_load(bundle, f"capabilities/{cid}/entities.json")["entities"])
        caps[cid] = cap
        action_keys = {a["key"] for a in cap["actions"]}
        all_intents |= set(wf["intents"])

        if cap["id"] != cid:
            diags.append(Diagnostic(3, "CAPABILITY_ID_MISMATCH",
                                    f"capabilities/{cid}/capability.json",
                                    f"id '{cap['id']}' != directory '{cid}'"))
        for intent_id, intent in wf["intents"].items():
            loc = f"capabilities/{cid}/workflows.json"
            action = intent["completion"]["action"]
            if action not in action_keys:
                diags.append(Diagnostic(3, "DANGLING_ACTION", loc,
                                        f"{intent_id}: completion action '{action}' not in capability.json"))
            for slot in intent.get("slots", []):
                ent = slot["entity"]
                if ent in shared_entities:
                    used_shared.add(ent)
                elif ent not in local_entities:
                    # entity is neither local nor shared: dangling or cross-capability
                    owner = _entity_owner(bundle, ent, exclude=cid)
                    if owner:
                        diags.append(Diagnostic(4, "CROSS_CAPABILITY_REF", loc,
                                                f"{intent_id}.{slot['name']}: entity '{ent}' belongs to "
                                                f"capability '{owner}' — capabilities may not depend on "
                                                "each other (ADR-002 A9)"))
                    else:
                        diags.append(Diagnostic(3, "DANGLING_ENTITY", loc,
                                                f"{intent_id}.{slot['name']}: entity '{ent}' not found"))
            sup = intent.get("superseded_by")
            if sup:
                diags.append(Diagnostic(3, "SUPERSEDED_FORWARD_REF", loc,
                                        f"{intent_id}: superseded_by '{sup}' — resolved after "
                                        "all capabilities load", Severity.WARNING)
                             ) if sup not in all_intents else None

    # keyword rules → intents
    for f in sorted(bundle.glob("keywords/*.json")):
        rel = f.relative_to(bundle).as_posix()
        for rule in _load(bundle, rel)["rules"]:
            if rule["intent"] not in all_intents:
                diags.append(Diagnostic(3, "KEYWORD_DANGLING_INTENT", rel,
                                        f"rule '{rule['pattern']}' targets unknown intent '{rule['intent']}'"))

    # plan_facts → intents + capabilities
    plan = _load(bundle, "runtime/plan_facts.json")
    if set(plan["intents"]) != all_intents:
        missing = all_intents - set(plan["intents"])
        extra = set(plan["intents"]) - all_intents
        diags.append(Diagnostic(3, "PLAN_FACTS_MISMATCH", "runtime/plan_facts.json",
                                f"missing={sorted(missing)} extra={sorted(extra)}"))
    for intent_id, fact in plan["intents"].items():
        if fact["capability"] not in caps:
            diags.append(Diagnostic(3, "PLAN_FACTS_DANGLING_CAPABILITY", "runtime/plan_facts.json",
                                    f"{intent_id} → unknown capability '{fact['capability']}'"))

    # manifest capability registry ≡ capability dirs
    if set(manifest["capabilities"]) != set(caps):
        diags.append(Diagnostic(3, "MANIFEST_CAPABILITY_MISMATCH", "bundle.json",
                                f"manifest={sorted(manifest['capabilities'])} dirs={sorted(caps)}"))

    # orphan shared entities (stage 4/6 flag)
    for ent in sorted(shared_entities - used_shared):
        diags.append(Diagnostic(4, "ORPHAN_SHARED_ENTITY", "entities/shared/",
                                f"'{ent}' referenced by nothing", Severity.WARNING))
    return diags


def _entity_owner(bundle: Path, entity: str, exclude: str) -> str | None:
    for f in bundle.glob("capabilities/*/entities.json"):
        cid = f.parent.name
        if cid == exclude:
            continue
        if entity in json.loads(f.read_text())["entities"]:
            return cid
    return None


# --------------------------------------------------------------------------
# Stage 5 — policy completeness (validation half; the merge algebra is a
# compile transform and arrives with the full compiler)
# --------------------------------------------------------------------------

def stage_5_policies(bundle: Path) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    policies = _load(bundle, "runtime/policies.json")
    all_intents: set[str] = set()
    high_cost_intents: set[str] = set()
    for cap_dir in bundle.glob("capabilities/*"):
        cid = cap_dir.name
        wf = _load(bundle, f"capabilities/{cid}/workflows.json")
        cap = _load(bundle, f"capabilities/{cid}/capability.json")
        hc_actions = {a["key"] for a in cap["actions"] if a.get("high_cost")}
        all_intents |= set(wf["intents"])
        for intent_id, intent in wf["intents"].items():
            if intent["completion"]["action"] in hc_actions:
                high_cost_intents.add(intent_id)

    matrix = policies["confirmation"]
    missing = all_intents - set(matrix)
    if missing:
        diags.append(Diagnostic(5, "POLICY_MATRIX_INCOMPLETE", "runtime/policies.json",
                                f"no confirmation policy for {sorted(missing)}"))
    for intent_id in sorted(high_cost_intents):
        if matrix.get(intent_id) != "always":
            diags.append(Diagnostic(5, "HIGH_COST_UNCONFIRMED", "runtime/policies.json",
                                    f"'{intent_id}' completes a high_cost action but confirmation is "
                                    f"'{matrix.get(intent_id)}' — must be 'always' (wrong-action budget)"))
    return diags


# --------------------------------------------------------------------------
# Stage 8 — cross-artifact parity (metadata half)
# --------------------------------------------------------------------------

def stage_8_parity(bundle: Path) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    manifest = _load(bundle, "bundle.json")
    all_intents: set[str] = set()
    for wf in bundle.glob("capabilities/*/workflows.json"):
        all_intents |= set(json.loads(wf.read_text())["intents"])

    for lang in manifest.get("models", {}).get("intent", {}):
        rel = f"models/intent/{lang}/labels.json"
        if not (bundle / rel).exists():
            diags.append(Diagnostic(8, "LABELS_MISSING", rel, "declared model has no labels.json"))
            continue
        labels = _load(bundle, rel)
        if set(labels) != all_intents:
            diags.append(Diagnostic(8, "LABEL_INTENT_MISMATCH", rel,
                                    f"labels ≠ compiled intent set: only_labels="
                                    f"{sorted(set(labels) - all_intents)} only_intents="
                                    f"{sorted(all_intents - set(labels))}"))
        cal = f"models/intent/{lang}/calibration.json"
        if not (bundle / cal).exists():
            diags.append(Diagnostic(8, "CALIBRATION_MISSING", cal,
                                    "calibration must exist for every shipped model"))

    embedders = {v["model_version"]
                 for v in manifest.get("models", {}).get("embedder", {}).values()}
    for lang, entry in manifest.get("models", {}).get("semantic_head", {}).items():
        rel = f"models/semantic_head/{lang}/head.json"
        declared = entry.get("embedder_id")
        if declared and embedders and declared not in embedders:
            diags.append(Diagnostic(8, "EMBEDDER_ID_MISMATCH", "bundle.json",
                                    f"semantic_head[{lang}].embedder_id '{declared}' matches no "
                                    f"shipped embedder {sorted(embedders)} — wrong-vector-space bug class"))
        if (bundle / rel).exists():
            head = _load(bundle, rel)
            if head["embedder_id"] != declared:
                diags.append(Diagnostic(8, "EMBEDDER_ID_MISMATCH", rel,
                                        f"head.json embedder_id '{head['embedder_id']}' != manifest '{declared}'"))
            expected = all_intents | {head.get("oos_label", "sys.oos")}
            if set(head["labels"]) != expected:
                diags.append(Diagnostic(8, "HEAD_LABEL_MISMATCH", rel,
                                        "semantic-head labels must be the intent set plus OOS"))
            if len(head["weights"]) != len(head["labels"]) or len(head["bias"]) != len(head["labels"]):
                diags.append(Diagnostic(8, "HEAD_SHAPE_MISMATCH", rel,
                                        "weights/bias rows must equal label count"))

    if manifest["telemetry_schema_version"] != _load(bundle, "telemetry/schema.json")["telemetry_schema_version"]:
        diags.append(Diagnostic(8, "TELEMETRY_VERSION_MISMATCH", "telemetry/schema.json",
                                "manifest and telemetry schema versions disagree"))
    return diags


# --------------------------------------------------------------------------
# Stage 9 — localization completeness
# --------------------------------------------------------------------------

def stage_9_localization(bundle: Path) -> list[Diagnostic]:
    diags: list[Diagnostic] = []
    manifest = _load(bundle, "bundle.json")
    langs = set(manifest["languages"])
    production = manifest.get("channel") == "production"

    for lang in sorted(langs):
        for rel, code in ((f"lexicons/{lang}.json", "LEXICON_MISSING"),
                          (f"keywords/{lang}.json", "KEYWORDS_MISSING")):
            if not (bundle / rel).exists():
                sev = Severity.ERROR if production else Severity.WARNING
                diags.append(Diagnostic(9, code, rel, f"language '{lang}' declared but file absent", sev))

    for cap_dir in sorted(bundle.glob("capabilities/*")):
        cid = cap_dir.name
        cap = _load(bundle, f"capabilities/{cid}/capability.json")
        wf = _load(bundle, f"capabilities/{cid}/workflows.json")
        referenced: set[str] = set()
        for intent in wf["intents"].values():
            for slot in intent.get("slots", []):
                referenced.add(slot["prompt"])
                if "reprompt" in slot:
                    referenced.add(slot["reprompt"])
                referenced.update(v["failure_prompt"] for v in slot.get("validation", [])
                                  if "failure_prompt" in v)
            if "confirmation" in intent:
                referenced.add(intent["confirmation"]["prompt"])
            referenced.add(intent["completion"]["response"])
            for k, v in (intent.get("followup") or {}).items():
                if k != "on_yes":
                    referenced.add(v)
        ur = cap.get("availability", {}).get("unavailable_response")
        if ur:
            referenced.add(ur)
        for lang in sorted(set(cap["languages"]) & langs):
            rel = f"capabilities/{cid}/responses/{lang}.json"
            if not (bundle / rel).exists():
                diags.append(Diagnostic(9, "RESPONSES_MISSING", rel, "declared language has no responses file"))
                continue
            keys = set(_load(bundle, rel))
            for key in sorted(referenced - keys):
                diags.append(Diagnostic(9, "RESPONSE_KEY_MISSING", rel, f"referenced key '{key}' absent"))
    return diags


# --------------------------------------------------------------------------
# Stage 10 — portable regex subset
# --------------------------------------------------------------------------

def stage_10_portability(bundle: Path) -> list[Diagnostic]:
    diags: list[Diagnostic] = []

    def check(rel: str, pattern: str, where: str):
        for problem in check_pattern(pattern):
            diags.append(Diagnostic(10, "REGEX_NOT_PORTABLE", rel, f"{where}: {problem} in {pattern!r}"))

    for f in sorted(bundle.glob("keywords/*.json")):
        rel = f.relative_to(bundle).as_posix()
        for i, rule in enumerate(_load(bundle, rel)["rules"]):
            check(rel, rule["pattern"], f"rules[{i}].pattern")
            for j, g in enumerate(rule.get("guards", [])):
                check(rel, g, f"rules[{i}].guards[{j}]")
    for f in sorted(bundle.rglob("entities*.json")):
        rel = f.relative_to(bundle).as_posix()
        data = json.loads(f.read_text())
        if "entities" not in data:
            continue
        for name, ent in data["entities"].items():
            if ent.get("type") == "pattern":
                check(rel, ent["pattern"], f"entities.{name}.pattern")
    for f in sorted(bundle.glob("lexicons/*.json")):
        rel = f.relative_to(bundle).as_posix()
        for i, carrier in enumerate(_load(bundle, rel).get("carriers", [])):
            check(rel, carrier, f"carriers[{i}]")
    return diags


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def validate_bundle_dir(bundle: Path | str, fmt: str = "3.0") -> list[Diagnostic]:
    """Run all v0 validation stages in compiler order; returns all diagnostics.

    Schema failures (stage 1) short-circuit the semantic stages — semantic
    checks on unvalidated shapes would produce noise, not signal.
    """
    bundle = Path(bundle)
    diags = stage_1_schemas(bundle, fmt)
    if any(d.severity is Severity.ERROR for d in diags):
        return diags
    for stage in (stage_3_references, stage_5_policies, stage_8_parity,
                  stage_9_localization, stage_10_portability):
        diags.extend(stage(bundle))
    return diags
