# Restructure Move Plan — packages/ + apps/ + spec/ + content/ (ND-2 proposal)

Status: **PROPOSAL — awaiting owner approval.** No file moves happen until
this plan is approved. Per roadmap §13 / ADR-005 Part 13 / ADR-003 Part 13.

## Principle

Mechanical and behavior-preserving. Every phase ends green (`make check` +
parity replay) and is a single revertable commit. Imports change; **logic
does not**. The frozen-oracle rule applies: after each phase, the replay
corpus (holdout + datetime parity CSVs + NLU golden fixtures) must produce
byte-identical outputs vs. the pre-move baseline.

## Target layout (end state)

```
intent-platform/
├── spec/                          # ✅ exists already (format 3.0 schemas + examples)
├── content/                       # conversational content (CODEOWNERS: content team)
│   ├── capabilities/<id>/         # from data/nlu_schema.json + data/localization/ (split)
│   ├── entities/shared/           # from data/nlu_entities.json (split)
│   ├── lexicons/                  # from data/localization/ yes/no/carriers
│   └── policies.yaml routing.yaml # extracted from schema thresholds + engine constants
├── datasets/                      # DVC-tracked: from data/*.csv + multilingual/data/
├── packages/
│   ├── buildtime/nlu_compiler/    # NEW (validator library first — ADR-005 AI#3)
│   ├── buildtime/nlu_training/    # from scripts/train*.py, multilingual/train_*.py,
│   │                              #   calibrate_languages.py, build_* data scripts
│   ├── buildtime/nlu_export/      # from export_*.py, compare_coreml_quant.py
│   └── runtime/nlu_engine/        # from scripts/nlu/ (engine, classifier, entities,
│                                  #   context, semantic, manifest) + multilingual predict
├── apps/cli/                      # from scripts/nlu_cli*.py, predict.py
├── bundles/                       # build output (gitignored)
├── tests/                         # stays; grows unit/component/golden/parity/perf split
└── docs/                          # stays
```

## Move map (mechanical, per phase)

| Phase | Move | Import fix | Risk |
|---|---|---|---|
| M0 | Baseline capture: record `make check` output, replay-corpus outputs, holdout metrics as the oracle | none | none |
| M1 | `scripts/nlu/` → `packages/runtime/nlu_engine/` (git mv, package `__init__` + back-compat shim `scripts/nlu` → re-export with DeprecationWarning) | `from nlu import …` → `from nlu_engine import …`; shim keeps old imports working one release | LOW — pure move; shim guarantees zero breakage |
| M2 | Trainers/calibration → `packages/buildtime/nlu_training/`; exports → `packages/buildtime/nlu_export/`; CLIs + predict.py → `apps/cli/` | path constants (BASE_DIR climbs) centralized in one `paths.py` per package | MED — scripts compute repo-relative paths; centralizing is the point |
| M3 | `data/*.csv`, `multilingual/data/` → `datasets/` under DVC; `data/nlu_schema.json`, `nlu_entities.json`, `data/localization/` → `content/` (initial 1:1 copy split later by the capability repartition) | loader paths in `paths.py` only | MED — DVC init is additive; CSVs move wholesale |
| M4 | `Makefile` targets + CI workflows + docs/memory updated to new paths; delete back-compat shims after one green cycle | — | LOW |

Not moved: `multilingual/models/da/` (tracked exception), `models/`
(gitignored artifacts), `docs/`, `.claude/`.

## Parity strategy (the proof obligation)

1. **Freeze the oracle before M1**: run the full replay corpus (holdout
   `--strict`, datetime parity CSVs en/fr/de/da, NLU golden fixtures, ONNX↔iOS
   conformance vectors) on the pre-move commit; store outputs as
   `tests/parity/oracle_pre_restructure/` (committed).
2. After every phase: re-run the corpus through the moved code; **diff must be
   empty**. Any non-empty diff aborts the phase (revert the commit).
3. Model artifacts are NOT retrained during the restructure — the same .onnx/
   .pkl files are loaded from new paths, so parity failures can only come from
   path/import mistakes, which is exactly what the replay detects.
4. iOS is untouched (artifacts + fixtures keep their published names via the
   export step); the ONNX↔iOS conformance test guards this.

## What this plan does NOT include (separate approvals)

- Label-space/taxonomy changes (ND-3 — separate plan + baseline diff).
- The capability repartition of content/ (follows the restructure; ADR-002 A3
  map) — M3 lands content/ as a 1:1 move, repartition is its own reviewed change.
- The bundle compiler implementation (starts after M1 in
  `packages/buildtime/nlu_compiler/`, consuming `spec/`).

## Estimated shape

4 commits (one per phase), each individually green and revertable; no
behavior change anywhere; total diff is ~renames + import lines + one
`paths.py` per package.

**Decision requested:** approve M0–M4 as scoped above? (Any phase can also be
approved individually.)
