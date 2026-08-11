"""
Evaluation probes that the training corpus does not cover.

These exist because held-out accuracy on a template-generated corpus is close
to meaningless — the model is being asked about phrasings it has effectively
already seen. The two sets below are the ones that actually move.

HARD_PARAPHRASES — real-world rewordings using vocabulary absent from training,
including the antonym trap that started this work ("it's too quiet, make it
louder"), plus typos.

OUT_OF_SCOPE — utterances the model must *refuse*, not classify. A linear head
always names a class, so rejection quality is measured separately via the
prototype OOD score.

Add to these lists whenever a real user utterance is misclassified. They are
the regression suite.
"""

from typing import List, Tuple

HARD_PARAPHRASES: List[Tuple[str, str]] = [
    # --- the antonym trap: state word and action word point opposite ways ---
    ("it's too loudy here can you make it quiter", "device.volume.decrease"),
    ("it's too quiter here can you make it louder", "device.volume.increase"),
    ("this restaurant is deafening tone it down a notch", "device.volume.decrease"),
    ("i'm struggling to follow the conversation give me more", "device.volume.increase"),
    ("my ears are ringing from how strong this is", "device.volume.decrease"),
    ("everything is a whisper i need more of it", "device.volume.increase"),
    ("everything sounds muffled and faint turn it up", "device.volume.increase"),
    ("the tv is way too harsh right now soften it", "device.volume.decrease"),
    ("i cannot make out a single word she is saying", "device.volume.increase"),
    ("this is blasting way beyond comfortable", "device.volume.decrease"),
    ("make it less aggressive on my ears", "device.volume.decrease"),
    # --- terse / idiomatic ---
    ("crank em", "device.volume.increase"),
    ("dial it back please", "device.volume.decrease"),
    ("boost it a couple notches", "device.volume.increase"),
    # --- mute / unmute ---
    ("i need total silence for a moment", "device.volume.mute"),
    ("kill the audio completely", "device.volume.mute"),
    ("i want peace and quiet switch the mics off", "device.volume.mute"),
    ("bring the sound back i want to hear again", "device.volume.unmute"),
    ("stop being silent i want audio again", "device.volume.unmute"),
    ("turn my microphones back on i muted them earlier", "device.volume.unmute"),
    # --- program / memory ---
    ("i am going to a noisy cafe adjust my setup for that", "device.memory.change"),
    ("flip me over to the outdoor configuration", "device.memory.change"),
    ("set me up for the concert hall tonight", "device.memory.change"),
    # --- reminders ---
    ("don't let me forget to take my pills at nine", "reminders.task.create"),
    ("nudge me about the dentist tomorrow morning", "reminders.task.create"),
    ("i already took my medication cross that off", "reminders.task.complete"),
    ("that errand is done tick it off the list", "reminders.task.complete"),
    # NOTE: help.reminder.show is help *about the reminder feature*
    # ("how do i set a reminder"), NOT "list my reminders". The corpus has no
    # list-my-reminders intent at all — see KNOWN_GAPS below.
    ("i have no idea how to make an alarm repeat every week", "help.reminder.show"),
    ("show me the steps to delete a scheduled alert", "help.reminder.show"),
    ("is there a way to snooze these notifications", "help.reminder.show"),
    # --- streaming ---
    ("pipe my music straight into my ears", "streaming.session.start"),
    ("start sending the podcast to my aids", "streaming.session.start"),
    ("cut the bluetooth feed to my aids", "streaming.session.stop"),
    ("disconnect the audio stream now", "streaming.session.stop"),
    # --- find phone ---
    ("i misplaced my handset help me track it", "find.phone.locate"),
    ("where did i leave my mobile", "find.phone.locate"),
    ("ping my phone i lost it in the couch", "find.phone.locate"),
]

# Antonym pairs stated explicitly: same words, opposite intent. If any of these
# flip, the head has regressed to bag-of-words behaviour.
ANTONYM_PAIRS: List[Tuple[str, str, str, str]] = [
    (
        "it's too loud here can you make it quieter",
        "device.volume.decrease",
        "it's too quiet here can you make it louder",
        "device.volume.increase",
    ),
    (
        "too noisy make it softer",
        "device.volume.decrease",
        "too weak make it stronger",
        "device.volume.increase",
    ),
    (
        "the sound is loud lower it",
        "device.volume.decrease",
        "the sound is low raise it",
        "device.volume.increase",
    ),
    (
        "it is silent turn the sound on",
        "device.volume.unmute",
        "it is loud turn the sound off",
        "device.volume.mute",
    ),
]

OUT_OF_SCOPE: List[str] = [
    "what is the weather in paris tomorrow",
    "who won the football match last night",
    "tell me a joke about penguins",
    "how do i cook basmati rice",
    "what is the capital of denmark",
    "book me a flight to berlin",
    "my battery is about to die",
    "translate this into german",
    "how old is the president",
    "send an email to my boss",
    "what time does the pharmacy close",
    "play chess with me",
    "convert 20 euros to dollars",
    "what is my step count today",
    "order me a pizza",
    "how tall is mount everest",
    "read me the news headlines",
    "set a timer for ten minutes",
    "call my daughter",
    "take a photo",
    "what's on tv tonight",
    "define serendipity",
    "spell accommodation",
    "how far is the airport",
    "open the calculator",
    "my knee hurts",
    "what year did the war end",
    "recommend a good book",
    "how do i get to the station",
    "turn on the kitchen lights",
    "is it going to rain",
    "what's the exchange rate",
    "start a workout",
    "scan this document",
    "what is two plus two",
    "who wrote hamlet",
    "tell me about your company",
    "i love you",
    "asdfgh qwerty",
    "",
    # These *feel* in-domain but the corpus has no intent for them. They must
    # be rejected, not forced into reminders.task.complete. See KNOWN_GAPS.
    "what have i got lined up for today",
    "read out my pending to dos",
    "run through everything on my agenda",
]

# Capabilities users ask for that the corpus cannot express. Until a training
# intent exists, the only correct behaviour is rejection via the OOD gate.
KNOWN_GAPS = {
    "reminders.task.list": [
        "what have i got lined up for today",
        "read out my pending to dos",
        "run through everything on my agenda",
    ],
}
# Previously listed here and since fixed by semantic_intent.augment:
#   state-phrased unmute ("it is silent turn the sound on") — was predicted
#   device.volume.mute before contrastive augmentation, now correct at 0.99.
