# Recording the data for signal 6 (recognizer confidence)

About an hour of recording. It is the only remaining piece of the gate that
cannot be produced from the dataset, because the number it needs comes from
your recognizer on your hardware, not from text.

## Why it cannot be skipped or guessed

The gate has five signals and all five read text. The failure they cannot see
looks like this:

> The user is telling a story: *"…and then the director said push it down for
> dramatics, and everyone laughed."* The recognizer, listening continuously,
> emits **"and push it down for dramatics"**.

As text this is unremarkable English that genuinely sits near a volume command.
It is not out of distribution — measured OOD AUROC on this kind of input is
**0.70**, against 0.92 on real OOD. Confidence and margin are computed over 57
classes that all assume the person was addressing the device, so a confident
answer is exactly what you get. Nothing in the text says *"this was not aimed
at you"*.

The recognizer usually knows. It scored that utterance lower than it scores a
deliberate command, because the audio was further from the microphone, ran into
other speech, and was not preceded by a pause. That number is currently
discarded between the ASR and this model.

## Before recording: check whether you need this at all

**If the product can use push-to-talk — a button, a tap, a double-tap on the
aid — stop here.** It removes this failure completely and costs no model work.
Signal 6 exists for always-listening designs. Do not spend an hour recording
for a problem a button already solves.

## What to record

**120 utterances minimum, roughly 60/40.** The script refuses to fit on fewer
than 25 of either class, and a threshold fitted near that floor will move with
the next handful you collect.

### Class 1 — `is_command = 1` (about 70 utterances)

Real commands, spoken the way people actually speak to the device. Vary what
the recognizer finds hard, not what the classifier finds hard:

| vary | examples |
|---|---|
| distance | at the phone, across the room, from another chair |
| background | quiet room, TV on, tap running, someone else talking |
| speed | normal, rushed, trailing off at the end |
| speaker | at least 3 people, including one older voice |

Use commands from across the taxonomy — some volume, some memory, some help
questions, a couple of reminders. The *content* barely matters here; the
recording conditions are the variable.

### Class 2 — `is_command = 0` (about 50 utterances)

Speech the recognizer picked up that was **not addressed to the device**. This
is the class people get wrong, so be strict about what belongs in it:

- one side of a phone call
- two people talking in the room, device not addressed
- the television or radio
- someone reading aloud, or talking to a pet
- the user thinking out loud — *"where did I put that"*
- **sentences that read like commands but were not aimed at the device** —
  "turn it up, I can't hear the telly", said to a spouse

That last row is the point of the whole exercise. A `is_command = 0` set made
only of weather chat will fit a threshold that separates nothing useful,
because the other five signals already reject weather chat.

> `is_command` records **whether the person was addressing the device**, not
> whether the words look like a command. "and push it down for dramatics" is
> `is_command = 0`.

## Getting the confidence number out

Read the **per-utterance** score, not a per-word average — per-word averages
are usually flat and will produce AUROC near 0.5.

| recognizer | field |
|---|---|
| Android `SpeechRecognizer` | `RESULTS_CONFIDENCE_SCORES[0]` from the results bundle |
| Whisper / whisper.cpp | `avg_logprob` for the segment; `no_speech_prob` is often more informative — try both as separate columns |
| Vosk | `result[].conf` averaged is per-word — prefer the endpointer's score if exposed |
| Picovoice / Porcupine | the endpoint score, not the wake-word score |

If your API gives more than one candidate signal, record each as its own
column and fit each separately. `--data` takes whichever column you name
`asr_confidence`, so run the fitter once per candidate and keep the one with
the higher AUROC.

## The file

`data/asr_samples.csv`, three columns:

```csv
text,asr_confidence,is_command
turn the volume up please,0.91,1
and push it down for dramatics,0.44,0
what is the weather doing,0.88,0
switch me to the restaurant setting,0.79,1
```

`text` is what the recognizer produced — including its mistakes. Do not correct
it. A garbled transcript with a low score is the most valuable row in the file.

## Fitting

```bash
python scripts/fit_asr_threshold.py --data data/asr_samples.csv
```

Read the output before applying it. Three things can happen:

**AUROC ≥ 0.60 and a threshold is found.** Apply it:

```bash
python scripts/fit_asr_threshold.py --data data/asr_samples.csv \
       --apply models/final_student_256/onnx
```

This writes an `asr` block into `runtime_config.json`. Nothing else changes —
the model file is untouched, because this is a threshold, not a weight.

**AUROC < 0.60.** The script stops and does not write a threshold. The score
you exported is not informative. Check you are reading per-utterance confidence,
try the no-speech / endpointing score instead, and if neither separates: use
push-to-talk. This is a real answer, not a failure.

**No threshold loses ≤ 2% of commands.** The distributions overlap. Raising
`--max-command-loss` is a product decision — a lost command is a request that
silently did nothing, and the user will simply say it again, louder.

## After fitting

`--max-command-loss` defaults to 2% because a false rejection is recoverable
(the user repeats themselves) and a false execution may not be (the volume
jumps in a quiet room). If your five high-risk intents make that trade look
wrong, change the flag, not the code.

Refit when the recognizer, its model version, or the microphone changes. The
threshold is a property of that stack, not of this classifier — which is why it
lives in `runtime_config.json` beside the other thresholds and not in the graph
with the temperature.
