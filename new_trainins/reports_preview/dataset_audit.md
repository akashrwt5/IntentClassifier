# Dataset Audit — en.csv

Phase 1 of the robustness plan. No model involved; pure data inspection.

## 1. Scale and balance

| metric | value |
|---|---|
| rows (after empty-text drop) | 9826 |
| intents | 57 |
| largest class | 1884 |
| median class | 80 |
| smallest class | 53 |
| imbalance ratio (max/min) | 35.55x |
| classes with <60 examples | 13 |

Classes under 60 examples (these carry the most risk of a weak, over-confident decision boundary):

```text
Help_Health                              59
Cmd.TranscribeStart                      59
Help_Reminder                            56
Help_MaskMode                            56
Help_Battery                             55
Help_HearingCareAnywhereConnect          55
Help_WiCROS                              55
Cmd.ActivityCycle                        55
Cmd.ActivityAerobics                     54
Cmd.ActivityCalories                     54
Help_AppSettings                         54
Help_HeartRateRecovery                   54
Help_DemoMode                            53
```

### Full distribution

| intent | n | share |
|---|---|---|
| `Cmd.MemoryChange` | 1884 | 19.2% |
| `Default Fallback Intent` | 1308 | 13.3% |
| `reminders.add` | 832 | 8.5% |
| `Cmd.VolumeDecrease` | 301 | 3.1% |
| `Help_FindMyHearingAids` | 264 | 2.7% |
| `Help_Pairing` | 263 | 2.7% |
| `Cmd.VolumeIncrease` | 253 | 2.6% |
| `Help_Tinnitus` | 225 | 2.3% |
| `Cmd.StreamingStart` | 209 | 2.1% |
| `Cmd.SendMessage` | 181 | 1.8% |
| `Help_Volume` | 170 | 1.7% |
| `Help_ChangingMemories` | 158 | 1.6% |
| `Help_Accessories` | 152 | 1.5% |
| `Help_DeviceSettings` | 151 | 1.5% |
| `Help_RemoteProgramming` | 149 | 1.5% |
| `Cmd.BatteryLevel` | 135 | 1.4% |
| `Help_FallAlert` | 133 | 1.4% |
| `Help_SelfCheck` | 130 | 1.3% |
| `Cmd.VolumeUnmute` | 130 | 1.3% |
| `Help_Home` | 128 | 1.3% |
| `Help_ThriveScore` | 124 | 1.3% |
| `Cmd.FindMyPhone` | 122 | 1.2% |
| `Cmd.VolumeMute` | 118 | 1.2% |
| `Help_IntelliVoice` | 107 | 1.1% |
| `Help_MemoryOptions` | 98 | 1.0% |
| `Cmd.ListenMessage` | 98 | 1.0% |
| `Help_HearShare` | 85 | 0.9% |
| `Help_Transcribe` | 82 | 0.8% |
| `Help_WhatsNew` | 80 | 0.8% |
| `Help_Translate` | 79 | 0.8% |
| `reminders.complete` | 78 | 0.8% |
| `Cmd.TranslationStart` | 74 | 0.8% |
| `Cmd.ActivityStep` | 71 | 0.7% |
| `Cmd.ActivityRun` | 71 | 0.7% |
| `Cmd.StreamingStop` | 71 | 0.7% |
| `Cmd.ActivityStand` | 70 | 0.7% |
| `Help_InsertDevice` | 70 | 0.7% |
| `Cmd.ActivityWalk` | 67 | 0.7% |
| `Help_Customize` | 67 | 0.7% |
| `Help_CleanCare` | 67 | 0.7% |
| `Help_VoiceAssistant` | 66 | 0.7% |
| `Help_EdgeMode` | 65 | 0.7% |
| `Cmd.ActivityExercise` | 61 | 0.6% |
| `Help_HeartRate` | 60 | 0.6% |
| `Help_Health` | 59 | 0.6% |
| `Cmd.TranscribeStart` | 59 | 0.6% |
| `Help_Reminder` | 56 | 0.6% |
| `Help_MaskMode` | 56 | 0.6% |
| `Help_Battery` | 55 | 0.6% |
| `Help_HearingCareAnywhereConnect` | 55 | 0.6% |
| `Help_WiCROS` | 55 | 0.6% |
| `Cmd.ActivityCycle` | 55 | 0.6% |
| `Cmd.ActivityAerobics` | 54 | 0.5% |
| `Cmd.ActivityCalories` | 54 | 0.5% |
| `Help_AppSettings` | 54 | 0.5% |
| `Help_HeartRateRecovery` | 54 | 0.5% |
| `Help_DemoMode` | 53 | 0.5% |

## 2. Duplicates

| check | value |
|---|---|
| exact duplicate rows | 0 |
| exact duplicate groups | 0 |
| normalized duplicate rows | 205 |
| normalized duplicate groups | 102 |
| fuzzy near-dupe pairs within an intent (>=92 token_sort) | 1579 |

## 3. Label consistency

- Normalized texts carrying more than one label: **0**
- Leakage-key collisions across different intents: **2**

Examples of cross-intent collisions (same content words, different label):

```text
labels: Cmd.MemoryChange, Default Fallback Intent
  - change to work
  - please change to work
  - can you change to work
  - that works to change it
labels: Default Fallback Intent, Help_Home
  - how do i use
  - how do i use this
```

## 4. Vocabulary shortcut risk

Tokens covering >=60% of one intent's rows while appearing in at most 2 intents overall. These are exactly the shortcuts the plan warns about (Section 5 / Section 23): the model can learn the token instead of the meaning, and then negation or context flips break it.

| intent | token | coverage | intents containing |
|---|---|---|---|
| `Help_HeartRateRecovery` | `recovery` | 98% | 1 |
| `Help_EdgeMode` | `edge` | 88% | 2 |
| `Help_HeartRateRecovery` | `rate` | 78% | 2 |
| `Help_HeartRate` | `rate` | 77% | 2 |
| `Cmd.ActivityAerobics` | `aerobics` | 70% | 1 |
| `Help_Tinnitus` | `tinnitus` | 64% | 2 |

### Most distinctive tokens per intent

```text
Cmd.ActivityAerobics                 aerobics(2.845), goal(0.547), aerobic(0.374), activity(0.35), reached(0.27), doing(0.246)
Cmd.ActivityCalories                 calorie(1.423), calories(1.164), burned(0.721), today(0.664), burn(0.406), used(0.334)
Cmd.ActivityCycle                    biking(1.157), cycling(1.103), ride(0.735), goal(0.436), bike(0.426), biked(0.305)
Cmd.ActivityExercise                 exercise(0.928), goal(0.484), workout(0.369), calories(0.29), exercising(0.265), working(0.257)
Cmd.ActivityRun                      running(0.96), jogging(0.797), run(0.686), goal(0.39), calories(0.249), jog(0.228)
Cmd.ActivityStand                    standing(1.502), stand(1.178), stood(0.866), goal(0.659), long(0.376), have(0.214)
Cmd.ActivityStep                     step(1.659), steps(1.123), goal(0.598), taken(0.566), count(0.443), many(0.336)
Cmd.ActivityWalk                     walking(0.981), walk(0.952), stroll(0.483), walked(0.317), log(0.205), goal(0.193)
Cmd.BatteryLevel                     battery(1.259), power(0.539), charge(0.433), level(0.295), remaining(0.27), batteries(0.262)
Cmd.FindMyPhone                      phone(0.868), android(0.741), iphone(0.555), detect(0.298), locate(0.29), find(0.281)
Cmd.ListenMessage                    message(1.247), play(1.217), last(0.574), messages(0.495), texts(0.289), read(0.276)
Cmd.MemoryChange                     program(0.87), memory(0.858), switch(0.535), change(0.366), set(0.211), number(0.19)
Cmd.SendMessage                      message(1.497), send(0.682), text(0.386), someone(0.313), contact(0.235), a(0.223)
Cmd.StreamingStart                   stream(0.64), tv(0.507), streaming(0.399), audio(0.358), remote(0.302), listen(0.28)
Cmd.StreamingStop                    streaming(0.983), stop(0.664), tv(0.553), stream(0.514), end(0.425), disconnect(0.283)
Cmd.TranscribeStart                  transcribe(0.681), recording(0.405), conversation(0.405), writing(0.343), start(0.328), text(0.267)
Cmd.TranslationStart                 translation(1.086), translate(1.074), owe(0.71), español(0.492), spanish(0.438), translating(0.328)
Cmd.VolumeDecrease                   volume(0.644), down(0.496), decrease(0.254), lower(0.242), quieter(0.211), too(0.195)
Cmd.VolumeIncrease                   volume(0.824), up(0.289), increase(0.26), too(0.216), turn(0.205), raise(0.176)
Cmd.VolumeMute                       volume(0.516), mute(0.515), off(0.488), turn(0.251), silent(0.24), sound(0.234)
Cmd.VolumeUnmute                     unmute(0.515), volume(0.483), turn(0.317), sound(0.266), mute(0.26), again(0.226)
Default Fallback Intent              your(0.173), that(0.133), it(0.094), and(0.069), but(0.061), so(0.058)
Help_Accessories                     accessories(1.011), accessory(0.825), mic(0.639), streamer(0.419), remote(0.4), tv(0.349)
Help_AppSettings                     app(0.823), advanced(0.674), basic(0.674), settings(0.624), version(0.327), change(0.269)
Help_Battery                         battery(1.546), charge(0.676), batteries(0.321), aid(0.255), change(0.198), ric(0.183)
Help_ChangingMemories                changing(1.193), programs(0.807), memories(0.764), memory(0.499), change(0.429), program(0.356)
Help_CleanCare                       clean(1.187), wax(0.845), tools(0.302), aids(0.272), buildup(0.241), build(0.241)
Help_Customize                       customize(0.872), treble(0.362), equalizer(0.302), customizing(0.241), settings(0.21), sound(0.206)
Help_DemoMode                        demo(1.444), without(1.138), mode(0.871), app(0.551), connected(0.351), features(0.184)
Help_DeviceSettings                  double(0.532), tap(0.487), sensitivity(0.295), model(0.241), serial(0.214), sound(0.206)
Help_EdgeMode                        edge(2.938), mode(1.647), wind(0.362), noise(0.258), reduce(0.163), setting(0.134)
Help_FallAlert                       alert(1.866), fall(1.107), manual(1.033), message(0.799), text(0.711), would(0.471)
Help_FindMyHearingAids               other(0.508), find(0.423), locate(0.368), aid(0.353), misplaced(0.305), cannot(0.303)
Help_Health                          goals(1.439), health(1.249), weekly(0.511), see(0.428), daily(0.343), dashboard(0.274)
Help_HearShare                       hearshare(1.427), share(1.143), person(0.434), caregiver(0.394), invitation(0.381), information(0.375)
Help_HearingCareAnywhereConnect      tele(0.515), care(0.398), remote(0.368), cloud(0.294), telehear(0.294), backup(0.294)
Help_HeartRate                       rate(2.568), heart(2.356), bpm(0.539), function(0.245), pulse(0.202), measure(0.167)
Help_HeartRateRecovery               recovery(3.968), rate(2.605), heart(2.236), measured(0.186), mean(0.164), number(0.125)
Help_Home                            screen(0.575), main(0.537), home(0.299), app(0.238), this(0.134), here(0.123)
Help_InsertDevice                    insert(0.574), aid(0.383), ear(0.323), put(0.312), t(0.304), keeps(0.294)
Help_IntelliVoice                    intellivoice(1.171), intelligent(1.134), enhancement(1.096), dnn(1.02), voice(0.968), speech(0.518)
Help_MaskMode                        mask(2.562), mode(1.516), face(0.538), masks(0.505), issues(0.361), fix(0.237)
Help_MemoryOptions                   memory(1.263), program(0.482), create(0.472), location(0.407), custom(0.371), geotag(0.289)
Help_Pairing                         pairing(1.553), phone(0.61), pair(0.369), bluetooth(0.277), connected(0.242), app(0.227)
Help_Reminder                        reminder(1.525), reminders(1.436), daily(0.241), event(0.179), set(0.178), create(0.13)
Help_RemoteProgramming               audiologist(0.967), specialist(0.652), remotely(0.652), adjusted(0.63), professional(0.517), made(0.495)
Help_SelfCheck                       self(1.337), check(0.503), problem(0.283), not(0.227), test(0.226), aid(0.187)
Help_ThriveScore                     score(2.517), wellness(0.404), ipro(0.378), engagement(0.351), information(0.343), body(0.332)
Help_Tinnitus                        tinnitus(2.159), masker(1.114), noise(0.792), stimulus(0.611), adjustment(0.53), adjustments(0.508)
Help_Transcribe                      transcribe(1.92), transcription(0.327), text(0.22), user(0.203), speech(0.203), feature(0.183)
Help_Translate                       translate(1.752), translation(1.272), languages(0.563), feature(0.263), language(0.235), user(0.117)
Help_VoiceAssistant                  assistant(2.186), voice(1.126), thrive(0.496), virtual(0.245), thive(0.184), control(0.184)
Help_Volume                          volume(0.966), loudness(0.802), just(0.347), changes(0.242), aid(0.229), adjust(0.173)
Help_WhatsNew                        overview(0.81), new(0.609), app(0.508), quick(0.442), engage(0.404), show(0.293)
Help_WiCROS                          cros(2.279), balance(1.124), transmitter(0.809), control(0.619), receiver(0.426), wicros(0.368)
reminders.add                        remind(1.719), m(1.215), p(0.865), reminder(0.853), at(0.63), 5(0.462)
reminders.complete                   reminder(2.529), complete(1.203), that(0.693), mark(0.601), latest(0.443), finish(0.429)
```

## 5. Length profile

p10=3 · p50=6 · p90=10 · p99=15 · max=31 words

Rows shorter than 3 words: 403

## 6. Fallback / OOD class already present

`Default Fallback Intent` has 1308 rows. This is a supervised reject class, which is useful, but it is not a substitute for OOD evaluation: it only covers unsupported phrasings someone thought of in advance. Phase 9 still needs a held-out OOD suite, including near-OOD.
