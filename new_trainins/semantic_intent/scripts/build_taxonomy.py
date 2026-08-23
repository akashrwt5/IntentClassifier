"""Phase 2 — Intent taxonomy / policy audit.

Writes configs/intents.yaml: for each of the 57 intents, a description, the
action class (command vs help vs reject), its confusable siblings, and the
policy decisions that were ambiguous. Positive examples are pulled from the
real data so the config stays honest about what the label actually contains.
"""

from __future__ import annotations

import sys
from pathlib import Path

import re

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# family -> members. Members inside a family are the ones a classifier is most
# likely to swap. Every family is a minimal-pair / hard-negative generator.
FAMILIES = {
    "volume": [
        "Cmd.VolumeIncrease",
        "Cmd.VolumeDecrease",
        "Cmd.VolumeMute",
        "Cmd.VolumeUnmute",
        "Help_Volume",
    ],
    "memory": ["Cmd.MemoryChange", "Help_ChangingMemories", "Help_MemoryOptions", "Help_Customize"],
    "streaming": ["Cmd.StreamingStart", "Cmd.StreamingStop", "Help_Accessories"],
    "transcribe": ["Cmd.TranscribeStart", "Help_Transcribe"],
    "translate": ["Cmd.TranslationStart", "Help_Translate"],
    "find": ["Cmd.FindMyPhone", "Help_FindMyHearingAids"],
    "battery": ["Cmd.BatteryLevel", "Help_Battery"],
    "reminders": ["reminders.add", "reminders.complete", "Help_Reminder"],
    "messages": ["Cmd.SendMessage", "Cmd.ListenMessage"],
    "activity": [
        "Cmd.ActivityWalk",
        "Cmd.ActivityRun",
        "Cmd.ActivityCycle",
        "Cmd.ActivityStep",
        "Cmd.ActivityStand",
        "Cmd.ActivityExercise",
        "Cmd.ActivityAerobics",
        "Cmd.ActivityCalories",
    ],
    "health": ["Help_Health", "Help_HeartRate", "Help_HeartRateRecovery", "Help_ThriveScore"],
    "care": ["Help_CleanCare", "Help_InsertDevice", "Help_SelfCheck", "Help_DeviceSettings"],
    "remote": ["Help_RemoteProgramming", "Help_HearingCareAnywhereConnect", "Help_HearShare"],
    "features": [
        "Help_EdgeMode",
        "Help_MaskMode",
        "Help_IntelliVoice",
        "Help_Tinnitus",
        "Help_WiCROS",
        "Help_DemoMode",
    ],
    "app": [
        "Help_Home",
        "Help_AppSettings",
        "Help_WhatsNew",
        "Help_VoiceAssistant",
        "Help_Pairing",
    ],
    "safety": ["Help_FallAlert"],
    "reject": ["Default Fallback Intent"],
}

# How much damage does executing this wrongly do, and can the user undo it?
# This is the axis the gate needs, and it is NOT the same as Cmd vs Help.
#   high   — the user may not be able to notice or reverse it, or it reaches
#            someone else. Needs a stricter confidence threshold.
#   normal — audible, obvious and trivially reversible, or read-only.
RISK = {
    # Muting is the worst case on a hearing aid: the user loses the channel
    # they would use to notice the mistake and correct it.
    "Cmd.VolumeMute": "high",
    # Sends something to another person — an external side effect.
    "Cmd.SendMessage": "high",
    # Clears a reminder the user may be relying on (medication, appointments).
    "reminders.complete": "high",
    # Cuts audio the user was actively listening to.
    "Cmd.StreamingStop": "high",
    # Silently changes the sound profile; disorienting and not obvious why.
    "Cmd.MemoryChange": "high",
    # Unmute is the RECOVERY action, not a risk — firing it wrongly is loud
    # but immediately obvious and immediately reversible.
}

DESCRIPTIONS = {
    "Cmd.VolumeIncrease": "User asks the aid to raise volume now.",
    "Cmd.VolumeDecrease": "User asks the aid to lower volume now.",
    "Cmd.VolumeMute": "User asks to silence the aids entirely.",
    "Cmd.VolumeUnmute": "User asks to restore sound after muting.",
    "Help_Volume": "User asks HOW volume works or how to adjust it, not to change it now.",
    "Cmd.MemoryChange": "Switch to a named or numbered program/memory now.",
    "Help_ChangingMemories": "How-to for switching programs.",
    "Help_MemoryOptions": "How to create/customize/geotag programs.",
    "Help_Customize": "How to personalize sound (equalizer, treble, noise).",
    "Cmd.StreamingStart": "Begin audio streaming to the aids now.",
    "Cmd.StreamingStop": "End audio streaming now.",
    "Help_Accessories": "How-to for TV streamers, remote mics and other accessories.",
    "Cmd.TranscribeStart": "Start live transcription now.",
    "Help_Transcribe": "How the transcribe feature works.",
    "Cmd.TranslationStart": "Start translation now, or translate a given phrase.",
    "Help_Translate": "How the translate feature works.",
    "Cmd.FindMyPhone": "Locate the user's phone.",
    "Help_FindMyHearingAids": "Locate a lost hearing aid / use the aid finder.",
    "Cmd.BatteryLevel": "Report current battery charge now.",
    "Help_Battery": "How-to about batteries: size, type, replacing, charging.",
    "reminders.add": "Create a reminder.",
    "reminders.complete": "Mark an existing reminder done.",
    "Help_Reminder": "How the reminder feature works.",
    "Cmd.SendMessage": "Record/compose and send a voice message.",
    "Cmd.ListenMessage": "Play back received messages.",
    "Cmd.ActivityWalk": "Log or query walking activity.",
    "Cmd.ActivityRun": "Log or query running activity.",
    "Cmd.ActivityCycle": "Log or query cycling activity.",
    "Cmd.ActivityStep": "Log or query step count / step goal.",
    "Cmd.ActivityStand": "Log or query standing time / stand goal.",
    "Cmd.ActivityExercise": "Log or query generic exercise/workout.",
    "Cmd.ActivityAerobics": "Log or query aerobics / cardio-dance activity.",
    "Cmd.ActivityCalories": "Query calories burned.",
    "Help_Health": "How-to for activity/health summaries and goals.",
    "Help_HeartRate": "How-to for the heart-rate feature.",
    "Help_HeartRateRecovery": "How-to for the heart-rate-recovery metric.",
    "Help_ThriveScore": "How-to for the engagement/brain/body score.",
    "Help_CleanCare": "Cleaning, wax guards, moisture, waterproofing.",
    "Help_InsertDevice": "How to physically insert/wear the aid.",
    "Help_SelfCheck": "Run or interpret a self-check / diagnose a faulty aid.",
    "Help_DeviceSettings": "Device-level settings and hardware troubleshooting.",
    "Help_RemoteProgramming": "Remote adjustments made by the audiologist.",
    "Help_HearingCareAnywhereConnect": "Connecting for remote care / cloud restore.",
    "Help_HearShare": "Sharing hearing data with a family member/caregiver.",
    "Help_EdgeMode": "The on-demand environmental-optimization mode.",
    "Help_MaskMode": "The mode that compensates for face masks.",
    "Help_IntelliVoice": "DNN speech-enhancement feature.",
    "Help_Tinnitus": "Tinnitus masker / stimulus feature.",
    "Help_WiCROS": "CROS/WiCROS single-sided-hearing system.",
    "Help_DemoMode": "App demo mode.",
    "Help_Home": "Generic 'how does this work' / home screen / getting started.",
    "Help_AppSettings": "App-level settings, caregiver and alert settings.",
    "Help_WhatsNew": "What changed in the new app version.",
    "Help_VoiceAssistant": "Third-party voice assistants.",
    "Help_Pairing": "Pairing aids with phone/app, Bluetooth.",
    "Help_FallAlert": "Fall detection and alert messages.",
    "Default Fallback Intent": "Anything outside the supported command space.",
}

# The ambiguous cases the plan (Section 4 / Phase 6) says must be decided
# BEFORE training. These are product-policy calls, written down so evaluation
# can be consistent.
POLICIES = [
    dict(
        id="P1-result-request-vs-feature-question",
        question="Is 'how do I turn up the volume' a command or a how-to? And "
        "why is 'how much battery is left' a command when it also "
        "starts with 'how'?",
        decision="The question word does NOT decide it. Ask instead: does the "
        "app have a function for this, and is the user asking for its "
        "RESULT, or asking how the function works?\n"
        "  result request  -> Cmd.*   ('how much battery is left', "
        "'where is my phone', 'what is my step count')\n"
        "  feature question -> Help_* ('how do i check the battery', "
        "'what battery size do i need')\n"
        "Both frames can start with how/what/where. The noun and the "
        "thing being asked for decide, not the opener.",
        rationale="A rule keyed on the question word is 89.7% right on this "
        "corpus, but the wrong 10.3% is not random: it is 240 rows "
        "covering Cmd.BatteryLevel (54), all eight Cmd.Activity* "
        "intents (154) and Cmd.FindMyPhone (17) — i.e. every intent "
        "whose job is to report the user's own data. Those are real "
        "app functions, so asking for them is a command. Executing a "
        "setting change when the user only asked how it works is the "
        "opposite error and is the more harmful one, which is why "
        "the boundary is written down rather than left to the model.",
    ),
    dict(
        id="P1b-help-prefix-is-not-a-kind",
        question="Does the `Help_` prefix mean the intent is a how-to intent?",
        decision="No. Treat the prefix as a name, not a type. Three Help_* "
        "intents are majority action requests: Help_FindMyHearingAids "
        "(66% action), Help_Pairing (49%), Help_InsertDevice (47%). "
        "Help_HeartRate absorbs BOTH 'how does heart rate work' and "
        "'what is my heart rate' because the app has no separate "
        "result intent for it.",
        rationale="Measured from en.csv. Any generated data, evaluation bucket "
        "or runtime rule that assumes Cmd.=action / Help_=question "
        "will be wrong on those intents. The taxonomy therefore "
        "records a measured `action_share` per intent instead of "
        "inferring a kind from the label string.",
    ),
    dict(
        id="P2-bare-negation",
        question="What does 'don't make it louder' mean with no alternative given?",
        decision="Default Fallback Intent (no action).",
        rationale="A negated command with no stated alternative does not "
        "identify an action. Mapping it to the opposite intent is "
        "the exact shortcut the plan forbids in Section 8.",
    ),
    dict(
        id="P3-corrective-negation",
        question="What about 'not quieter, louder' / 'I said turn it down, not up'?",
        decision="The affirmed alternative wins (Cmd.VolumeIncrease / "
        "Cmd.VolumeDecrease respectively).",
        rationale="A correction states a target action explicitly.",
        superseded_at_runtime="P3 still defines the correct LABEL, but the "
        "gate no longer ACTS on it. Measured accuracy on this sentence "
        "shape is 0.48 for intent pairs the model was not explicitly "
        "taught and 0.74 for ones it was — both far below the 0.97 "
        "precision the gate promises. The classifier's answer is recorded "
        "and the user is asked to repeat. Cost measured before adopting: "
        "0 of 1513 held-out rows, 0 of 1508 validation rows and 0 of 1496 "
        "STT rows carry this structure, so real traffic is unaffected.",
    ),
    dict(
        id="P4-symptom-reports",
        question="Is 'I can barely hear' a volume increase?",
        decision="Cmd.VolumeIncrease.",
        rationale="Already present in the training data as such; it is the "
        "dominant user phrasing and the action is low-risk and "
        "trivially reversible.",
    ),
    dict(
        id="P4b-symptom-vs-requested-action",
        question="'It is too loud in here, can you make it quieter' — does "
        "'loud' or 'quieter' decide the intent?",
        decision="Read what the person is ASKING FOR, not what they are "
        "describing.\n"
        "  symptom + an explicit request -> the REQUEST decides.\n"
        "      'it is too loud ... make it quieter'  -> Decrease\n"
        "      'it is too quiet ... make it louder'  -> Increase\n"
        "  symptom with NO request -> the symptom implies the action "
        "(this is P4).\n"
        "      'i can barely hear'                   -> Increase\n"
        "The state clause describes the problem; the request clause "
        "is the command. A sentence containing both is not ambiguous "
        "and must never be treated as such.",
        rationale="The model predicted Cmd.VolumeIncrease for 'it is too loud "
        "in here, can you make it quieter' — no negation, no "
        "correction, the action spelled out, and it still read the "
        "adjective instead of the verb. The corpus explains it: only "
        "54 rows use symptom phrasing at all and they run 31 "
        "Increase to 14 Decrease, so 'a complaint about sound' had "
        "been learned as evidence for turning it up. F12 supplies "
        "the missing half in equal numbers so the adjective stops "
        "carrying the label.",
    ),
    dict(
        id="P5-other-device-volume",
        question="'Turn the TV down' / 'make the phone louder'?",
        decision="Near-OOD -> Default Fallback Intent, EXCEPT where the aid "
        "controls the TV streamer, which the data labels Help_Accessories.",
        rationale="The aid must not silently reinterpret another device's " "volume as its own.",
    ),
    dict(
        id="P6-activity-query-vs-log",
        question="'How far have I walked' vs 'start a walk' — one intent or two?",
        decision="One intent per activity type (the shipped taxonomy merges "
        "log and query). Do not split.",
        rationale="Matches the existing label set; splitting would invalidate "
        "the baseline comparison.",
    ),
    dict(
        id="P6b-generic-activity",
        question="What is 'show me my activity goal' — there is no generic "
        "activity intent among the eight Cmd.Activity* labels.",
        decision="FOLLOW THE DATASET. 'activity goal' (and target/daily goal) "
        "is Cmd.ActivityAerobics. 'activity summary / overview / "
        "stats / history' is Help_Health. The boundary is the NOUN, "
        "not the verb.",
        rationale="All 8 'activity goal' rows in en.csv are labelled "
        "Cmd.ActivityAerobics and all 'activity summary' rows are "
        "Help_Health. The split is not semantically obvious, so the "
        "model cannot infer it from 8 examples — F6 in "
        "build_targeted_training.py teaches it explicitly with "
        "matched openers so only the noun carries the label. "
        "Re-labelling was considered and rejected: the app's "
        "behaviour is defined by the shipped label, not by what the "
        "phrase sounds like.",
    ),
    dict(
        id="P6c-walking-vs-steps",
        question="Is 'did I reach my walking goal' ActivityWalk or ActivityStep?",
        decision="Cmd.ActivityWalk. 'walking' selects Walk; 'step/steps' "
        "selects Step; 'stand' selects Stand.",
        rationale="Dataset convention: 'walking goal' -> ActivityWalk (5 rows), "
        "'step goal' -> ActivityStep (19 rows). An earlier draft of "
        "the minimal-pair suite got this backwards and was measuring "
        "the suite's own error as a model error.",
    ),
    dict(
        id="P7-fallback-is-not-ood-eval",
        question="Does the Default Fallback Intent class cover OOD?",
        decision="No. It is a supervised reject class for anticipated "
        "unsupported phrasings. OOD is measured separately on a "
        "held-out suite that includes near-OOD.",
        rationale="Section 11 of the plan.",
    ),
]


HOWTO_RE = re.compile(
    r"^(how|what|where|why|when|which|can i|do i|is there|does|explain|guide)\b"
    r"|\bhow do i\b|\bhelp with\b|\buser guide\b|\bexplain\b"
)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "en.csv")
    counts = df["intent"].value_counts().to_dict()
    # Measured, not inferred from the label string — see policy P1b.
    df["_howto"] = df["text"].str.lower().str.strip().str.contains(HOWTO_RE)
    action_share = (1 - df.groupby("intent")["_howto"].mean()).round(3).to_dict()

    fam_of = {}
    for fam, members in FAMILIES.items():
        for m in members:
            fam_of[m] = fam

    intents = {}
    for intent in sorted(counts):
        fam = fam_of.get(intent, "other")
        siblings = [m for m in FAMILIES.get(fam, []) if m != intent]
        sub = df[df["intent"] == intent]["text"]
        # `kind` is measured from how users actually phrase this intent, not
        # read off the label prefix (policy P1b).
        share = action_share.get(intent, 0.0)
        kind = (
            "reject"
            if intent == "Default Fallback Intent"
            else "action" if share >= 0.65 else "howto" if share <= 0.35 else "mixed"
        )
        intents[intent] = dict(
            description=DESCRIPTIONS.get(intent, ""),
            kind=kind,
            family=fam,
            n_train_rows=int(counts[intent]),
            action_share=float(share),
            risk=RISK.get(intent, "normal"),
            label_prefix_says=(
                "command"
                if intent.startswith(("Cmd.", "reminders."))
                else "reject" if intent == "Default Fallback Intent" else "help"
            ),
            confusable_with=siblings,
            positive_examples=sub.sample(min(5, len(sub)), random_state=7).tolist(),
        )

    cfg = dict(
        n_intents=len(intents),
        risk_tiers={k: [i for i in intents if intents[i]["risk"] == k] for k in ("high", "normal")},
        families={k: v for k, v in FAMILIES.items()},
        policies=POLICIES,
        intents=intents,
    )
    out = ROOT / "configs" / "intents.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=100))
    missing = [i for i in counts if i not in DESCRIPTIONS]
    print(
        f"wrote {out} — {len(intents)} intents, "
        f"{len(FAMILIES)} families, {len(POLICIES)} policies; "
        f"missing descriptions: {missing}"
    )


if __name__ == "__main__":
    main()
