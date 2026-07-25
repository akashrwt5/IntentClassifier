---
description: Add a new language by authoring a Language Pack (no engine edits)
---

Read `.claude/memory/langpack.md` first, then
`docs/Review-F5/LOCAL-TESTING-AND-NEW-LANGUAGE-GUIDE.md` and
`docs/adding-a-new-language.md`.

**The whole point of this architecture: adding a language is authoring data, not
editing engine code.** If you find yourself modifying `scripts/nlu/`, stop — the
gap belongs in the pack contract, and that is an architectural change (invoke the
**architect** agent).

Given a language code `<lang>`:

1. Author `packs/<lang>/` mirroring `packs/en/`:
   `pack.json` (id, language, `engine_compat`, components, models),
   `config.json` (thresholds, stages, policy), `schema.json` (intents +
   keyword_triggers + affirmative/negative), `keywords.json`, `lexicons.json`,
   `normalizer.json`, `entities/enums.json`, `datetime/grammar.json`.
2. Verify it loads before training anything (`packages/` must be on the path):
   ```bash
   PYTHONPATH=packages python -c "from nlu_langpack import load_pack; \
     p=load_pack('packs/<lang>'); print(p.language, p.stages, p.semantic_available)"
   ```
   `packs/en` prints `en ['keyword', 'intent_model'] False`.
3. Train the intent model: `python multilingual/train_multilingual.py`
   (or `--model <lang>`), then populate `packs/<lang>/intent_model/`.
4. Calibrate: `python scripts/calibrate_languages.py` — adds the `<lang>` block
   to `config/calibration.json`.
5. Add evaluation assets: a per-language holdout under `multilingual/test/` and a
   datetime parity fixture in `tests/datetime_parity/`.
6. Confirm neutrality still holds: `python scripts/ci/check_language_neutral.py`
   and `pytest tests/test_neutrality.py`.
7. Update `langpack.md`, `datasets.md`, `roadmap.md`, and `known-issues.md`.

Leave semantic **off by default** unless the pack genuinely declares and ships a
semantic head.
