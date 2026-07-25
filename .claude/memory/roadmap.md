# Memory: Roadmap

> Single responsibility: what is done, what is next, what is gated. Re-derived
> from this branch — **not** the `feature/production-work` roadmap. Live detail:
> `docs/Review-F5/IMPLEMENTATION-PROGRESS.md` (newest checkpoint at the top);
> the plan itself: `docs/Review-F5/IMPLEMENTATION-PLAN.md`.

## Where this branch stands

The Review-F5 **Language Pack** plan (§9) is **complete — all 6 steps**:

| Step | State |
|---|---|
| 1. Lock the `LanguagePack` interface + loader | done |
| 2. Evict language data from the engine into an `en` pack | done — zero `if language` matches in `scripts/nlu/` |
| 3. Move keyword rules + datetime grammar into pack tables | done — zero English datetime words in regex literals |
| 4. Semantic stage pack-declared + off by default | done |
| 5. `pr.yml` CI (neutrality guard, hostile-pack tests) | done |
| 6. `train-and-gate.yml` + `release-pack.yml` automated release | done (ONNX + CoreML/ANE) |

Behaviour was preserved byte-for-byte through the eviction — classifier 37/37,
English datetime corpus 77/77 on both paths, strip 20/20, fr/de/da parity
unchanged, suite at the same 8 pre-existing failures. Baselines: `langpack.md`.

## Next checkpoint — REQUIRES EXPLICIT GO-AHEAD

The first step that edits/moves existing engine code. Do not start it
unprompted:

1. Big-bang move `scripts/nlu` → `packages/nlu_engine` (wholesale; update every
   importer and test). No dual-run, no shim (ADR-009).
2. A `RUNTIME_CONTRACT_VERSION`-aware engine constructor that takes a
   `LanguagePack` and builds its interpreters from pack resources.
3. Evict any residual `engine.py` language constants (carriers, yes/no, idioms,
   connectors) into `packs/en/lexicons.json` — behavior-preserving, parity-tested
   against the frozen current engine.
4. Wire the semantic stage fully as a pack-declared, off-by-default plugin.

## Open / carried work

- **2 tracked test failures remain** (was 8; six fixed 2026-07-25). Both are
  French clock idioms that need `packs/fr/` — not engine bugs. See
  `known-issues.md`.
- **Second language pack — now the critical path.** The architecture's whole
  claim is that adding a language is authoring `packs/<lang>/` plus training
  data, with no engine edits. Only `en` exists as a real pack (plus the hostile
  `zz` fixture), and the two remaining test failures are exactly what that gap
  costs. CI now enforces the contract from the engine side (no language
  branches, no embedded vocabulary), so the next real proof is a second pack.
- **Danish quality.** `da` is the weakest language (macro-F1 0.7448, ECE 0.0352
  vs `en` 0.9018 / 0.0184) and is below the 0.80 accuracy gate posture.
- **macOS-only verification.** Tier-B Core ML runtime parity, live ANE
  compute-plan placement, and the iOS XCTest suite cannot run on Linux; the iOS
  workflow in `akashrwt5/STT` still needs a one-time `INTENTCLASSIFIER_PAT`
  secret.
- **Memory-optimization track.** Distillation/compression options (E5-small vs
  MiniLM teacher, Model2Vec static embeddings):
  `docs/on-device-memory-optimization-plan.md`. Not started.

## Deliberately not on this branch

- The 59 → 57 `domain.object.action` label migration, the `datasets/` +
  `content/` split, `apps/`, `spec/`, DVC/MLflow, the capability repartition and
  bundle/compiler work — all of that lives on `feature/production-work`. This
  branch is **2 ahead / 57 behind** it. Do not assume any of it exists here.
- A Rust core (ADR-010).

## Related memory

Language Pack -> `langpack.md` · Decisions -> `decisions.md` · Known issues ->
`known-issues.md` · Training gate -> `training.md`.
