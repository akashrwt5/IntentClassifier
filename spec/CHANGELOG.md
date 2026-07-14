# spec/ changelog

## 2026-07-14 — format 3.0 initial authoring (ADR-005 AI#2)

- 16 JSON Schemas for format 3.0 under `bundle/3.0/` (manifest, models
  metadata, runtime tables, capability subtree, entities, lexicons, keywords,
  telemetry, meta).
- `portable-regex.md`: allowed/forbidden constructs, fixed semantics,
  8-row normative conformance corpus.
- Golden bundles `examples/3.0/{minimal,full}`.
- Conformance tests: `tests/test_bundle_spec.py`.
- Format `3.0` is hereby the initial versioned format; pre-existing ad-hoc
  artifacts are retroactively "format 2" (ADR-005 AI#1, ratified 2026-07-14).
