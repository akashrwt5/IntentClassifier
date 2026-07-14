# spec/ — the normative contracts (ADR-005)

Machine-readable source of truth for the NLU Bundle format. **Where this
directory and ADR-005 prose disagree, these schemas win** (ADR-005 risk R1).

- `bundle/3.0/*.schema.json` — JSON Schemas (draft 2020-12) for every JSON
  file in a format-3.0 bundle. `defs.schema.json` holds shared definitions
  (stable-id grammar, language codes, closed feature flags).
- `bundle/portable-regex.md` — the portable regex subset + normative
  conformance corpus (compiler stage 10; runtimes' shims must agree).
- `examples/3.0/minimal/` — smallest valid bundle: one language, one
  capability, two intents.
- `examples/3.0/full/` — full-featured: two languages (en/fr), two
  capabilities, semantic head + embedder pairing, high-cost confirmation
  workflow, shared entities, followups, keyword guards, experiment stamp.

Conformance: `tests/test_bundle_spec.py` proves schemas ↔ examples agree and
seeds the compiler's cross-artifact parity (stage 8) and localization
completeness (stage 9) checks. iOS/Android runtimes join by loading the same
examples in their CI (ADR-005 AI#6).

Deliberate omissions at this stage (tracked in EXECUTION_STATUS):

- `integrity/` files and `.nlu` packaging — compiler stages 11–15; signing is
  approval-gated (ND-8). Golden bundles are unpacked directory trees.
- Model binaries — `bundle.json` model entries reference artifact paths, but
  golden bundles ship no `.onnx` (generated artifacts are never committed);
  file-presence + tensor contracts are compiler stage-8 duties, not schema ones.
- `entities/system/datetime/{lang}.json` grammar tables — schema lands when the
  datetime grammar is extracted from `scripts/nlu/entities.py` (its shape is
  not yet specified by any ADR; inventing one here would be spec-by-accident).

Change rules: edits here follow ADR-005 Part 8 (format versioning law),
require platform-team review, and must update `spec/CHANGELOG.md`.
