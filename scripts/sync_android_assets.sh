#!/usr/bin/env bash
#
# Replace the Android app's bundled NLU pack with the current build.
#
# WHY THIS IS A SCRIPT AND NOT A COPY-PASTE
# -----------------------------------------
# Hand-copying put a pack from the `domain.object.action` label era
# (a6cbb81c, pre-0ad05a7e) into the app, where it sat next to a `Cmd.*` model.
# It crashed on a missing `agreement` threshold, which was lucky: the same
# mismatch in plan_facts/workflows produces no error at all, just every intent
# silently falling through to GenAI.
#
# So this script REFUSES rather than copies when either side looks wrong, and
# verifies what actually landed. `dist/` is gitignored and regenerated, so the
# copy has to be reproducible or it will drift again.
#
# Usage:
#   scripts/sync_android_assets.sh [ANDROID_REPO]
#
# Defaults to ~/StudioProjects/Starkey.Mobile.Android.Engage
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/dist/android-assets/nlu_pack"
ANDROID="${1:-$HOME/StudioProjects/Starkey.Mobile.Android.Engage}"
DEST_MODULE="$ANDROID/device/features/voiceaikit/src/main/assets"

die() { printf '\033[31mFAIL\033[0m  %s\n' "$*" >&2; exit 1; }
ok()  { printf '\033[32mok\033[0m    %s\n' "$*"; }

# ---------------------------------------------------------------- verify source

[ -d "$SRC" ] || die "no pack at $SRC
      build it first:
        PYTHONPATH=packages/buildtime python -m nlu_compiler.content_bundle --lang en --out dist/bundle-en
        scripts/build_android_assets.sh"

verify() {  # verify <pack_dir> <label>
  python3 - "$1" "$2" <<'PY'
import json, sys, pathlib
pack, label = pathlib.Path(sys.argv[1]), sys.argv[2]

pol = json.loads((pack / "runtime/policies.json").read_text())
th, conf = pol["thresholds"], pol["confirmation"]
plan = json.loads((pack / "runtime/plan_facts.json").read_text())["intents"]
labels = json.loads((pack / "models/intent/en/labels.json").read_text())

problems = []

# The threshold that crashed. Its absence dates the pack to before
# 0ad05a7e "one fire threshold, no confidence-driven confirmation".
for t in ("confidence", "agreement", "interrupt"):
    if t not in th:
        problems.append(f"thresholds missing {t!r}")

# The removed confidence band. A pack carrying it wants a mechanism that was
# deleted for turning 103 correct predictions into questions.
for t in th:
    if t.startswith("uncertain_confirm"):
        problems.append(f"thresholds carries removed band {t!r}")

# Label space. Confirmation keyed by action id means the a6cbb81c label space,
# which no longer matches anything the model emits.
bad_conf = [k for k in conf if not (k.startswith("Cmd.") or k.startswith("Help_")
                                    or k.startswith("reminders.") or k == "Default Fallback Intent")]
if bad_conf:
    problems.append(f"confirmation keyed by action id, e.g. {bad_conf[0]!r}")

vals = set(conf.values()) - {"always", "never"}
if vals:
    problems.append(f"confirmation has removed values {sorted(vals)}")

# The three that must agree, because nothing downstream can detect it if they
# do not: the model's classes, the routing table, and the workflow set.
missing = [l for l in labels if l not in plan and l != "Default Fallback Intent"]
if missing:
    problems.append(f"{len(missing)} model labels absent from plan_facts, e.g. {missing[0]!r}")

if problems:
    print(f"  {label}: NOT USABLE")
    for p in problems:
        print(f"    - {p}")
    sys.exit(1)

print(f"  {label}: {len(labels)} labels, {len(plan)} routed, thresholds {sorted(th)}")
PY
}

echo "source pack:"
verify "$SRC" "$(basename "$SRC")" || die "the pack in dist/ is itself stale — rebuild it, do not copy it"
ok "source pack is current"

# ------------------------------------------------------------------- replace

[ -d "$ANDROID" ] || die "no Android repo at $ANDROID"

# Every copy, not just the one we know about. Two modules each shipping
# assets/nlu_pack/ merge into one APK and the loser is invisible.
mapfile -t EXISTING < <(find "$ANDROID" -type d -name nlu_pack -not -path "*/build/*" 2>/dev/null || true)
if [ ${#EXISTING[@]} -gt 0 ]; then
  echo "removing ${#EXISTING[@]} existing pack(s):"
  for d in "${EXISTING[@]}"; do
    echo "  - ${d#$ANDROID/}"
    rm -rf "$d"
  done
fi

mkdir -p "$DEST_MODULE"
cp -R "$SRC" "$DEST_MODULE/"
ok "copied to ${DEST_MODULE#$ANDROID/}/nlu_pack"

# ------------------------------------------------------------------- confirm

echo "installed pack:"
verify "$DEST_MODULE/nlu_pack" "installed" || die "copy landed wrong"

REMAINING=$(find "$ANDROID" -type d -name nlu_pack -not -path "*/build/*" | wc -l | tr -d ' ')
[ "$REMAINING" = "1" ] || die "$REMAINING copies of nlu_pack remain — they will fight at asset-merge time"
ok "exactly one nlu_pack in the tree"

cat <<EOF

Assets are bundled INTO the apk, and an incremental install does not refresh
them. Uninstall first or the device keeps serving the old pack:

  adb uninstall com.starkey.mystarkey.integration
  ./gradlew :app:engage:clean :app:engage:installMystarkeyWorldwideIntegration
EOF
