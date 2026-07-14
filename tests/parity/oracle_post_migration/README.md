# Frozen baseline — post label-migration (ND-3, 2026-07-14)

Supersedes `../oracle_pre_restructure/` (which proved the ND-2 restructure
and is kept for history). Captured immediately after the approved ND-3
label migration + full retrain + recalibration on the 57-label
domain.object.action space.

vs the pre-migration baseline (accuracy / macro-F1):
  en .899/.893 → .895/.880 · fr .852/.840 → .845/.829
  de .833/.821 → .823/.791 · da .760/.728 → .765/.735 (flag-gated)
  OOS recall (min) .510 → .590  ← improved, as the migration predicted

Explained deviations: accuracy held within ±0.010 everywhere; macro-F1
shifts (de worst at −0.030) come from (a) removing the two easy dialogue-act
classes from the macro average and (b) the multinomial LR re-optimizing all
boundaries over 57 instead of 59 classes. Deterministic (fixed seeds).
Tracked in known-issues.md (de weak classes = small-support help.* topics).
