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

**Status:** Phase-1 step 1 (interface + loader). The engine does not yet consume
packs — that is the next checkpoint (big-bang move of `scripts/nlu` →
`packages/nlu_engine`, then eviction of language data into packs). See
`docs/Review-F5/IMPLEMENTATION-PROGRESS.md`.
