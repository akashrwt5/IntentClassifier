# Mic bug — root cause aur VoiceAssistantService ke exact changes

`SpeechManager.kt` poora replace ho chuka hai (`android/speechrecognition/SpeechManager.kt`).
Uska public surface bilkul wahi hai, sirf `startListening()` ab `suspend` hai —
aur dono call sites pehle se hi `launch(Dispatchers.Main)` ke andar hain, to
wahan kuch nahi badalta.

`VoiceAssistantService` mein **paanch** change chahiye. Baaki file ko haath mat lagao.

---

## Root cause — do lines

**1. Har turn par recognizer destroy + recreate.**

```kotlin
recognizerCall(CALL_DESTROY) { speechRecognizer.destroy() }
speechRecognizer = createRecognizer()
recognizerCall(CALL_START) { speechRecognizer.startListening(intent) }
```

`destroy()` recognition service ko **async** unbind karta hai. Naya instance usi
unbind ke beech bind karne jaata hai → `ERROR_SERVER_DISCONNECTED (11)` ~90ms par.
Har log mein yahi tha:

```
18.082  [S5]  startListening(intent) returned gen=3
18.176  [S16] onError code=11 gen=3          <- 94ms
```

**2. VA code 11 ko `FATAL` maanta hai.**

```kotlin
ERROR_CLIENT, RECOGNIZER_BUSY, ERROR_SERVER -> RECOGNISER_FAILED
else -> FATAL                                  // <- 11 yahan girta hai
```

`FATAL` → `endDialogAfterError()` → session khatam, koi retry nahi, mic wapas nahi aata.

**Cause 1 error paida karta hai, Cause 2 usse laa-ilaaj bana deta hai.** Bug yahi hai.

Teen aur cheezein jo isi ko pakka karti thin (sab SpeechManager mein theek):

| Symptom | Kya tha |
|---|---|
| `[S30] main=false` | `stopMicStream()` off-main se `stopListening()` bulata tha → call throw karti thi, `runCatching` nigal jaata tha, recognizer **actually ruka hi nahi**, par state "stopped" ho gaya. Agla start ek **zinda** recognizer ko destroy karta tha. |
| `[S21] subs=0` | `receiveSpeechAndClassify()` har turn collector cancel karke naya launch karta tha. Us gap mein aaya `Error` hamesha ke liye gum. |
| `[S58] mode 3→0` | `resetAudioInputRoute()` mode ko stale `originalAudioMode` (=0) par restore karta tha, aur usse **playback path se bhi** bulaya jaata tha. Mode 0 par comm-device mic audio deliver nahi karta → RMS 1–4 dB → `NO_MATCH`. |

---

## Change 1 — collector ek hi baar subscribe ho (sabse zaroori VA change)

`receiveSpeechAndClassify()` ko **poora replace** karo:

```kotlin
// PURANA: har turn par cancel + relaunch. Beech ka gap = subs=0 = event gum.
// NAYA: collector process-life ke liye ek hi baar chalta hai. Turn ke saath
// sirf ye badalta hai ki mic HA ka hai ya phone ka.
@Volatile private var activeMicIsHA = false

private fun receiveSpeechAndClassify(isHAMIC: Boolean) {
    activeMicIsHA = isHAMIC
}
```

`jobSpeechResponse` field **delete** kar do.

`init { ... }` block ke andar, sabse pehli line ke roop mein:

```kotlin
init {
    appCoroutineScope.launch { collectSpeechEventsForever() }
    // ...baaki init waisa hi...
}
```

Aur ye do method add karo:

```kotlin
/**
 * Kabhi cancel nahi hota. Agar collect kisi wajah se gira, dobara subscribe
 * karo -- kyunki subscriber ke bina SharedFlow(replay=0) par har event chup-chaap
 * gir jaata hai, aur khoya hua Error ka matlab hai mic hamesha ke liye band.
 */
private suspend fun collectSpeechEventsForever() {
    while (true) {
        runCatching {
            speechManager.events
                .onSubscription { Timber.d(V12, activeMicIsHA) }
                .collect { handleSpeechEvent(it) }
        }.onFailure { Timber.w(it, "[VA] speech collector restarting") }
    }
}

private suspend fun handleSpeechEvent(event: SpeechManager.SpeechEvent) {
    Timber.d(
        V30,
        event::class.simpleName,
        _streamingState.value::class.simpleName,
        pendingSlotFill != null,
        repeatingSessionID.value,
    )
    when (event) {
        is SpeechManager.SpeechEvent.Ready -> {
            if (micRetryUsed) Timber.d(V46)
            micRetryUsed = false
        }

        is SpeechManager.SpeechEvent.Processing -> Unit

        is SpeechManager.SpeechEvent.PartialResult -> {
            _streamingState.value = StreamingState.TRANSCRIPT(event.text)
        }

        is SpeechManager.SpeechEvent.Result -> {
            micRetryUsed = false
            updateStreamingStateAsStop()
            if (activeMicIsHA) stopMicStream() else stopPhoneMicStream()

            val pending = pendingSlotFill
            if (pending != null) {
                handleSlotFillFollowUp(pending, event.text)
            } else {
                onPVAResponse(offlineNluService.classifyIntent(event.text))
            }
        }

        is SpeechManager.SpeechEvent.Error -> {
            onSpeechRecognitionError(activeMicIsHA, event.code)
        }
    }
}
```

> Classification yahan **inline** hai, jaan-boojh kar. Us waqt mic ruk chuka hai,
> to koi naya event aa hi nahi sakta — collector ko block karne se kuch nahi bigadta,
> aur ordering guarantee bani rehti hai.

---

## Change 2 — code 11 ko survivable banao

`VoiceAssistantService.classifySpeechError()` mein:

```kotlin
private fun classifySpeechError(code: Int): SpeechFailure = when (code) {
    android.speech.SpeechRecognizer.ERROR_NO_MATCH,
    android.speech.SpeechRecognizer.ERROR_SPEECH_TIMEOUT,
    SpeechManager.ERROR_CODE_TIMEOUT,
        -> SpeechFailure.USER_SILENT

    android.speech.SpeechRecognizer.ERROR_CLIENT,
    android.speech.SpeechRecognizer.ERROR_RECOGNIZER_BUSY,
    android.speech.SpeechRecognizer.ERROR_SERVER,
    android.speech.SpeechRecognizer.ERROR_SERVER_DISCONNECTED,   // <-- NAYA
        -> SpeechFailure.RECOGNISER_FAILED

    else -> SpeechFailure.FATAL
}
```

Persistent recognizer ke baad 11 aana band ho jaana chahiye. Ye line backstop hai:
agar kabhi phir aaya, ek retry milega, dialog nahi marega.

---

## Change 3 — `startListening()` ab suspend hai

Dono call sites pehle se sahi hain, koi edit nahi chahiye:

```kotlin
appCoroutineScope.launch(Dispatchers.Main) {
    Timber.d(V14)
    speechManager.startListening()      // ab suspend -- compile ho jaayega
}
```

---

## Change 4 — `startPhoneMicStreaming` ka log tag galat hai

```kotlin
override fun startPhoneMicStreaming(isRepeatable: Boolean) {
    Timber.d(V50, Thread.currentThread().name)     // V50 = "stopMicStream"
```

Isse `V13` kar do (`[V13] startPhoneMicStreaming isRepeatable=%b`), warna log padhte
waqt start aur stop ek jaise dikhte hain — mujhe khud isse do baar dhokha hua.

```kotlin
Timber.d(V13, isRepeatable)
```

---

## Change 5 — `resetPVARequest()` ka mid-session risk band karo

`resetPVARequest()` `endDialogSession()` bulata hai, jo ab mode restore karta hai.
Agar Activity ise session ke beech bulaye to route beech mein gir jaayega.
Guard add karo:

```kotlin
override fun resetPVARequest() {
    // Mic zinda ho aur TTS na bhi chal raha ho, tab bhi session ko beech se
    // mat kaato -- SpeechManager ka route/mode teardown yahan chalne se turn
    // silent ho jaata hai.
    val micLive = _streamingState.value is StreamingState.STREAMING ||
            _streamingState.value is StreamingState.TRANSCRIPT

    jobPlayAudioRequest?.cancel()
    jobStreaming?.cancel()
    jobGASSState?.cancel()
    _streamingState.value = StreamingState.IDLE

    val currentSessionId = repeatingSessionID.value
    val isTTSPlaying = _isTTSPlaying.value

    if (currentSessionId.isEmpty() || !isTTSPlaying) {
        repeatingSessionID.value = ""
        pendingSlotFill = null
        micRetryUsed = false
        if (micLive) speechManager.stopListening() else speechManager.endDialogSession()
    }

    pushToTalkStatus = PushToTalkStatus.Idle
    stopPushToTalkRecordingTimer()
}
```

---

## Hatane wali cheezein

`VoiceAssistantService` se:

- `private var jobSpeechResponse: Job? = null` — ab nahi chahiye
- `V11` constant — collector ab "launched but not subscribed" state mein nahi rehta

`SpeechManager` se (nayi file mein pehle se hatai hui hain):

- duplicate `SpeechFailure` enum + `classifySpeechError()` — kabhi call hi nahi hote the
- `onRmsChanged` ka double log (`diag` + `Timber.d` dono)
- `LOG_ROUTED` const — kahin use nahi
- `lastStartElapsedMs` companion field
- `clearCommunicationDeviceOnly()` + `resetAudioInputRoute()` ka jodi-clear
- per-generation `SpeechListener` inner class

---

## Ek chhota behaviour change jo aapke haq mein hai

Purana `onResults` khaali match par `SpeechEvent.Error(-1)` bhejta tha → VA mein
`-1` kisi branch mein nahi tha → `FATAL` → dialog khatam.
Ab `ERROR_NO_MATCH (7)` jaata hai → `USER_SILENT` → slot dobara poocha jaata hai.

---

## Verify kaise karein

```bash
adb logcat -s Timber:* | grep -E "\[S[0-9]+\]|\[V[0-9]+\]"
```

Healthy multi-turn dialog ab aisa dikhna chahiye:

```
[S1]  startListening() session=1 state=IDLE
[S55] setCommunicationDevice target=Mukesh Hearing Aids accepted=true mode=3
[S60] route confirmed via callback device=Mukesh Hearing Aids   <- ab START se PEHLE
[S4]  recognizer created                                        <- SIRF pehli baar
[S5]  startListening(intent) issued session=1
[S10] onReadyForSpeech session=1
[S12] onBeginningOfSpeech
[S15] onResults n=1 session=1
[S20] emit Result subs=1 sent=true                              <- subs 0 nahi
...TTS...
[S1]  startListening() session=2 state=IDLE
[S57] route already live device=Mukesh Hearing Aids mode=3      <- re-acquire nahi
[S5]  startListening(intent) issued session=2                   <- naya [S4] NAHI
[S10] onReadyForSpeech session=2
```

Teen cheezein confirm karo:

1. **`[S4]` sirf ek baar.** Dobara aaya matlab `ERROR_CLIENT` hua tha — legit recreate.
2. **`[S16] code=11` bilkul nahi.** Aaya to `[S5]` aur uske beech ka gap dekho.
3. **`[S58]` bilkul nahi.** Aaya to koi VA ke bahar se mode badal raha hai — us
   stack ko `[S58]` ke timestamp se pakdo.
