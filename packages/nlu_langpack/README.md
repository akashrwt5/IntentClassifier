# nlu_langpack

The **Language Pack contract** for the on-device NLU platform. This package is the
locked boundary between the language-agnostic engine and everything
language-specific.

- `interfaces.py` — the component Protocols the engine depends on (the contract).
- `version.py` — the runtime-contract version + the compatibility gate.
- `manifest.py` — `pack.json` parsing + validation.
- `pack.py` — the `LanguagePack` container returned to the engine.
- `loader.py` — `load_pack()`: validate → compat gate → load config → resolve
  resources/models → gate the semantic stage off-by-default.

```python
from nlu_langpack import load_pack
pack = load_pack("packs/en")                  # semantic OFF by default
pack = load_pack("packs/en", enable_semantic=True)
print(pack.language, pack.stages, pack.semantic_available)
```

**Status:** plan steps 1–6 complete. The engine **does** consume packs: a bare
`NLUEngine()` loads `packs/en`, and every language-specific input — intents,
model, lexicons (incl. negation cues), keyword rules, entities, the full
datetime grammar, and policy — comes from the pack. `packs/en` is fully
populated, not a skeleton.

Adding a language is authoring `packs/<lang>/` plus training data; no engine
change is required, and CI enforces that
(`scripts/ci/check_language_neutral.py` rejects both language branches and
hardcoded match vocabulary).

Still pending: the big-bang move of `scripts/nlu` → `packages/nlu_engine`, which
is mechanical relocation and gated on explicit approval. See
`docs/Review-F5/IMPLEMENTATION-PROGRESS.md`.
