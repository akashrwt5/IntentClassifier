# Where this model stands — a plain summary

For anyone picking this up: what was built, what is inside it, what is wrong
with it, and how to check for yourself. No ML background assumed.

---

## 1. What the model does

It listens to a sentence and decides which of **57 things** the user is asking
for — turn the volume up, switch to the restaurant setting, how much battery is
left, how do I clean these, and so on. It runs **entirely on the device**. There
is no internet call anywhere in the path.

It also decides when *not* to act. If it is not sure, it refuses and asks the
user to repeat, rather than guessing. That refusal is the safety design: a
hearing aid that guesses wrong can turn itself down in a room where the user
already cannot hear.

---

## 2. What is inside it

### The model file

| | |
|---|---|
| shipped file | `models/final_student_256/onnx/intent_int8.onnx` |
| size | **4.75 MB** |
| speed | about **1 millisecond** per sentence |
| intents it knows | **57** |
| **vocabulary** | **3,267 words/word-pieces** |
| **transformer layers** | **4** |
| width (hidden size) | 256 |
| attention heads | 8 |
| longest input | 64 word-pieces |

### Where it came from

It was not trained from nothing. It was shrunk down from a bigger public model:

```
BAAI/bge-small-en-v1.5        ← the starting point, downloaded
  12 transformer layers
  384 wide
  30,522 word vocabulary
  35 MB
          |
          |  vocabulary pruning — the corpus only ever uses 3,267 of those
          |  30,522 words, so the other 27,255 were dropped
          |
          |  distillation — a smaller model is trained to copy the big one
          v
student-h256-l4               ← what ships
  4 transformer layers
  256 wide
  3,267 word vocabulary
  4.75 MB
```

Two other public encoders were tried and lost: `e5-small-v2` and
`all-MiniLM-L6-v2`.

### What else is in the shipped package

- `runtime_config.json` — the 57 intent names, all the safety thresholds, and
  57 reference points used to spot unfamiliar input
- `tokenizer/` — how sentences get chopped into word-pieces

### The safety gate — six reasons the aid will refuse

1. it recognises the request as something the product does not support
2. the input is unlike anything it was trained on
3. it is not confident enough (a stricter bar for the five riskiest actions)
4. its top two answers are too close together to call
5. the sentence corrects itself — "not X, I meant Y" — which it is bad at
6. the speech recogniser itself was unsure *(built, switched off — needs
   recordings from the real hardware to switch on)*

**Only the five riskiest intents get the strict bar:** mute, send a message,
mark a reminder done, stop streaming, change program. A mistake in those is one
the user may not notice or be able to undo. Unmute is deliberately *not* on that
list — it is the recovery action, and firing it wrongly is loud and instantly
obvious.

---

## 3. What was built to train it

The dataset is `data/raw/en.csv` — **9,826 sentences** across the 57 intents.

Two things about it shaped everything else:

- After removing near-duplicates, those 9,826 sentences are really only
  **5,173 genuinely different ones**. About half the file is the same sentence
  written twice.
- The classes are wildly uneven: the biggest has **1,884** examples, the
  smallest has **53**.

On top of that, **16 batches of extra training sentences** were generated, each
one written because a specific test was failing — negation, direction words,
question-vs-command, long sentences, accessories, and so on. Training ends up at
about 16,900 sentences.

There are also **seven test suites** that the model never trains on, covering
minimal pairs, hard negatives, negation, long sentences, speech-recogniser
errors, out-of-scope requests, and accessories.

---

## 4. The main problem, in one paragraph

**The model reads words, not sentences.**

An earlier draft of this file said "give it a short command and it does well".
That was wrong, and a two-word test found it:

```
"it's quiet"   ->  turn it DOWN   (91% sure)     wrong
"it's faint"   ->  turn it UP     (86% sure)     right
```

Both mean the same thing — the room is quiet, turn it up. Two words each, no
sentence to misread. **It is not a length problem.** The word's training
association wins whether the sentence is two words or twenty. Short input is not
safer, it just has fewer places to go wrong.

The clearest example is still the long one, because there the mistake is
obvious:

> "it's a bit quieter here, can you make it louder"

The person is saying the room is quiet, so please turn it **up**. The model sees
the word "quieter" and turns it **down**.

Why: in the training data the word "quieter" nearly always appears as a
*request* ("make it quieter"), so the model learnt that the word itself means
turn-down — regardless of what the rest of the sentence says.

Two measurements back this up:

- Scramble the words of a test sentence into a random order, and the model still
  gets **99%** of its normal score. A model that read sentences would fall apart;
  this one barely notices.
- Take the *same* request in the *same* shape and only change the describing
  word: with "faint" it is right **100%** of the time, with "quieter" it drops to
  **39%**.

### Is it dangerous?

Mostly no, and that is by design. In most of these cases the model is not
confident enough, so the gate refuses and the user repeats themselves. Annoying,
not harmful.

But not always. In one measured group, more than **one in four** of these
sentences got acted on wrongly — the aid turned the volume the opposite way and
the gate did not stop it. That is the part that matters.

### Can it be fixed?

Yes — it already has been, on a bigger model. Training the *encoder* (rather than
just the classifier on top of it) closed the gap completely: the score above went
from 100%-vs-39% to **100%-vs-100%**, and long sentences went from 0.49 to
**0.86**.

The catch is size. That fix worked on the 12-layer, 35 MB model. Every attempt to
get the same behaviour into the 4-layer, 4.75 MB model failed. Whether that is a
hard limit or just a training problem is the first question in
`ROADMAP_both_short_and_long.md`.

### The other honest caveats

- **New accessory names are not recognised.** It handles the accessories it was
  trained on — remote mic, TV streamer. Give it one it has not seen, like a neck
  loop, and it fails. `auracast` appears **zero times** in the training data.
- **The rarest intents will stay weak.** `Help_DemoMode` has 53 examples. No
  amount of clever engineering fixes a thin tail.
- **Coverage.** At the safety bar the product asks for, the model acts on about
  two-thirds of requests and asks the user to repeat the rest.

---

## 5. How to test it yourself

### The quick check — anyone can run this

```bash
python scripts/acceptance_test.py
```

Runs 46 sentences and tells you, in plain English, what the aid would have done
with each. Three outcomes:

| | meaning |
|---|---|
| **OK** | did the right thing |
| **ASKS** | did nothing, asked the user to repeat — annoying, safe |
| **WRONG** | acted, and acted wrongly — **report every one of these** |

Read the WRONG count first. Then look at the group table — a failure that is all
in one group ("full sentences") tells you something; a percentage does not.

### Test your own sentences

Make a CSV with two columns:

```csv
text,expected
turn my aids up,Cmd.VolumeIncrease
it is loud in here so turn it down,Cmd.VolumeDecrease
turn the television down,reject
what is the weather,reject
```

`expected` is the intent name from `configs/intents.yaml`, or the word `reject`
if the aid should refuse. Then:

```bash
python scripts/acceptance_test.py --file my_tests.csv
```

### Try one sentence at a time

```bash
python scripts/predict.py
```

Type a sentence, get the full decision — what it thought, how sure it was, what
its second guess was, and why it did or did not act.

### The technical suites

```bash
python scripts/evaluate_onnx.py --model models/final_student_256 \
       --onnx models/final_student_256/onnx/intent_int8.onnx
python scripts/structure_probe.py
```

The first runs all seven held-out suites. The second measures whether the model
reads sentences or words — the problem in §4.

---

## 6. Where to look next

| file | what it holds |
|---|---|
| `HANDOVER.md` | the full technical picture — every file, threshold, and gotcha |
| `ROADMAP_both_short_and_long.md` | the plan to fix §4, phase by phase, with costs |
| `PLAN_sentence_understanding.md` | why §4 happens, and what was tried |
| `reports/android_integration.md` | what the app has to do with all this |
| `README.md` | the numbers, and what is solved versus not |

---

## 7. One thing to check before anything else

`ROADMAP` phase 6b: record **100 real user utterances** and count how many are
two-clause — a situation described, then a request.

If most real speech is short commands, the problem in §4 barely touches real
users and this model is the right product today. If a lot of it is full
sentences, the size budget has to move.

Nothing in `en.csv` can answer that. It is what somebody *wrote*, not how people
*speak*.
