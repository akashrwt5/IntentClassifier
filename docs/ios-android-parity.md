# iOS ↔ Android parity — VoiceIntentKit se Android tak

Dono platform **ek hi NLU pack** use karte hain. Ye document batata hai ki iOS
(`VoiceIntentKit`, Swift) mein kya hai, Android mein uska jodidar kya hai, aur
kahan Android peeche hai.

---

## 0. Pack: kaunsa canonical hai

iOS ka source pack **`Sources/VoiceIntentSeedPackEN/packs/pack-en-v1.0.34`** hai.
(`.build/` mein `v1.0.30` pada hai — wo stale build artifact hai, usse compare mat
karna.)

50 files mein se **47 semantically identical** hain. Teen alag:

| File | Farq | Matlab |
|---|---|---|
| `bundle.json` | iOS `v1.0.34`, `checksums_root` real, `key_id: dev-key-golden`. Android `v1.0.0`, zeros, `unsigned`. | iOS pack **signed** hai |
| `calibration.json` | `temperature: 0.671457` **dono mein same**. iOS mein `temperature_coreml`, `temperature_coreml_full`, `temperature_int8` extra. | ONNX path par koi parity problem nahi |
| `model.onnx` | Size same (1,847,132), bytes alag. `training.run_ids`: iOS `en-6a280057`, Android `en-6e29269c` | **iOS ka model naya hai** |

**Faisla: Android bhi `pack-en-v1.0.34` ship kare**, `dist/bundle-en` nahi. Wo naye
training run se hai, signed hai, aur integrity files ke saath aata hai jinke bina
`PackIntegrity` ka port bekaar hai.

iOS pack mein jo extra hai aur Android ko **nahi** chahiye: `*.mlpackage`,
`*.mlmodelc`, `models/intent/en/iOS/`, `nlu_schema.json`, `nlu_entities.json`
(compiler input), `labels.pkl`, `.DS_Store`.

Jo extra hai aur **chahiye**:

- `integrity/manifest.sha256`, `integrity/signature.sig` — signature verification
- `models/intent/en/intent_classifier_weights_full.json` (2.78 MB) — TF-IDF
  vectorizer aur OOV guard ke liye
- `models/intent/en/tflite/` — abhi nahi, par ONNX ka replacement banega to yahin se

---

## 1. Class mapping

| iOS (Swift) | Android (Kotlin) | Haalat |
|---|---|---|
| `BundleDataLoader` | `NluManager.Pack` | naam align karna hai |
| `PackIntegrity` | — | **missing** |
| `PackTrustPolicy` / `PackLoadPolicy` | — | **missing** |
| `NLUBundle` | `NluManager.bundle` (raw JSONObject) | typed banana hai |
| `ResolvedPack` | `NluManager.Pack` | naam align karna hai |
| `PackSections` (`PackPolicies`, `PackCascade`, `PackRouting`, `PackGuards`, `CapabilityManifest`, `IntentWorkflow`) | raw `JSONObject` accessors | typed banana hai |
| `PackLexicon` / `DateTimeGrammar` | `lexicon: JSONObject` (sirf contractions padhi jaati hain) | adhoora |
| `PackTFIDFVectorizer` | — | **missing** |
| `PackIntentClassifier` | `OnnxIntentClassifier` | partial |
| `PackClassifierAdapter` | — | `IntentClassifying` ka Android jodidar nahi |
| `PackEntityExtractor` | `NluManager.resolveEntity` | logic same, jagah alag |
| `PackSlotResolver` / `SlotResolving` | `NluManager.getSlots` + app-side | **missing abstraction** |
| `PackDateTimeParser` | — | **missing** (app ka `parseReminderDateTime()` use hota hai) |
| `PackEngineFactory` | — | wiring bikhri hui hai |
| `NLUEngine` | `OfflineNluServiceImpl` | ladder same, shape alag |
| `NLUResponse` (enum) | `OfflineNluResult` (struct) | **shape alag** |
| `NLUSession` / `NLUConversationContext` | `VoiceAssistantService.pendingSlotFill` | app ke andar bikhra hua |
| `ConfirmationGate` | `NluManager.requiresConfirmation` | Boolean, enum nahi |
| `ClassificationResult` / `ClassificationBreakdown` | `OnnxPrediction` | breakdown missing |
| `SlotAnswerAssessment` | — | **missing** |
| `VoiceIntentSession` | `VoiceAssistantService` (app-owned) | Android mein kit ke bahar |
| `PackProvider` / `StaticPackProvider` | `NluModelFileStore` | alag maqsad, thoda overlap |
| — | `NluPackGuard` | Android-only; iOS mein iska kaam `PackIntegrity` + `BundleDataLoader` mein hai |
| — | `NluModelDownloader` / `NluModelUpdateManager` | Android-only (iOS mein pack SPM resource hai) |

---

## 2. Jo Android mein missing hai, importance ke hisaab se

### 2.1 `PackIntegrity` — signature verification

iOS: `verify(packRoot:trust:policy:) -> Verified`

- `bundle.json` se `signature_info.key_id` padhta hai
- `integrity/signature.sig` ko Curve25519 (ed25519) se verify karta hai
- `checksums_root` ko `manifest.sha256` ke SHA-256 se milata hai
- manifest ki har file ka digest check karta hai
- **unsigned extra files** detect karta hai (`unsignedFiles`) — manifest mein nahi
  hai par pack mein maujood hai
- `PackTrustPolicy.refusesDevelopmentPacks`, `skipsSignatureVerification`

Android: `NluPackGuard` mein sirf `TODO(security)` hai.

Port ho sakta hai: ed25519 verification BouncyCastle ya Tink se, SHA-256
`MessageDigest` se. Pack files pehle se ship hoti hain.

### 2.2 `PackTFIDFVectorizer` — aur uske saath OOV guard

iOS: `tokenize`, `features`, `vectorize`, `denseVector`, `producesNoFeatures`.

`intent_classifier_weights_full.json` ke `vocab` + `idf` se sublinear TF-IDF
banata hai. Isi se do cheezein milti hain jo Android mein nahi:

1. **`producesNoFeatures(text)`** — utterance ka koi bhi feature vocabulary mein
   nahi. `PackIntentClassifier.Prediction.isVacuous` yahi hai.
2. **OOV ratio** — `thresholds.oov_reject` / `oov_bypass` ka istemaal.

Maine pehle kaha tha ki Android par ye possible nahi kyunki ORT graph ki vocabulary
expose nahi karta. **Wo galat tha** — vocabulary weights JSON mein hai, graph mein
dhoondhne ki zaroorat hi nahi.

Keemat: 2.78 MB APK.

### 2.3 `PackDateTimeParser` — 829 lines

`lexicons/<lang>.json` ke `datetime_grammar` se poora parser: weekdays, months,
ordinals, `day_anchors`, `time_of_day`, `relative_markers`, `relative_units`,
`clock_idioms`, `am_pm`, `clock_hour_markers`, `quantifiers`, `strip` lists.

`Match(date, span, timeExplicit, dayExplicit)` lautata hai.

Android abhi app ke `parseReminderDateTime()` par nirbhar hai, jo pack ki grammar
nahi padhta. Matlab **reminder ka date/time do platform par alag parse hota hai**,
aur nayi language add karne par Android ko code change chahiye jabki iOS ko sirf
lexicon.

`tests/datetime_parity/` mein en/fr/de/da ke fixtures pehle se hain
(`nlu_datetime_parity_*.csv`), to port ko verify karne ka tarika maujood hai.

### 2.4 `NLUResponse` — sealed enum

```swift
case prompt(intent:question:filled:)
case confirm(intent:action:question:)
case fulfill(intent:action:parameters:message:confidence:semanticRescue:breakdown:)
case fallback(url:confidence:breakdown:)
case interrupted(cancelledIntent:result:)
```

Android `OfflineNluResult` ek flat struct hai jismein `allRequiredParamsPresent`
jaise boolean se state nikaalna padta hai. iOS wala shape har turn ka natija
exhaustive banata hai — `when` mein koi case chhoot nahi sakta.

`interrupted` ka Android mein koi jodidar hi nahi: naya confident command ek chalte
hue slot-fill ko cancel kar sakta hai. Abhi Android mein har follow-up utterance
slot answer maana jaata hai.

### 2.5 `NLUSession` / `NLUConversationContext`

Lifespan wale contexts (`setContext(name:lifespan:parameters:)`,
`decrementContexts()`, `resetSlotFilling()`, `resetAll()`) — DialogFlow ka model.

Android mein ye `VoiceAssistantService.pendingSlotFill` hai: ek nullable data class,
lifespan nahi, aur session ka maalik NLU layer nahi balki app hai.

### 2.6 `ConfirmationGate`

```swift
case always
case never
case whenAmbiguous(floor: Double, ceiling: Double)
```

`v1.0.34` pack mein sirf `always`/`never` hain, to `whenAmbiguous` abhi inert hai —
`PackEngineFactory.confirmationGates` use tabhi banata hai jab pack band bhejta ho.
Android ka `requiresConfirmation(): Boolean` aaj sahi jawab deta hai, par shape
teen values wali honi chahiye taaki pack band wapas laaye to client badalna na pade.

### 2.7 `SlotAnswerAssessment` + `assessSlotAnswer`

`complete` / `freeform` / `incomplete` — batata hai ki user ka slot answer poora
hai, free text hai, ya adhoora. Android bas `resolveFollowUpSlot` ka khaali/non-khaali
dekhkar retry counter badhata hai.

### 2.8 `deriveTopic` / `fillOpenTopics`

Reminder jaise open-text slots ke liye carriers aur leading connectors hata kar
topic nikalta hai (`lexicons` ke `carriers` / `leading_connectors`). Android ye
dono keys padhta hi nahi.

### 2.9 `ClassificationBreakdown`

Per-stage detail (`stage`, `intent`, `confidence`) + `winningStage`. Debug panel
aur telemetry ke liye. Android sirf final intent aur confidence rakhta hai.

### 2.10 `loadStage3` / `releaseStage3`

Semantic (MiniLM) stage ka lifecycle. `cascade.json` mein aaj `semantic: disabled`
hai, to dono jagah inert — par API iOS mein maujood hai.

---

## 3. Jo Android mein hai aur iOS mein nahi

Ye galtiyan nahi, platform ka farq hai:

- **`NluModelDownloader` / `NluModelUpdateManager` / `NluModelFileStore`** — iOS
  pack SPM resource ki tarah bundle karta hai (`PackProvider` / `StaticPackProvider`),
  to over-the-air update ka poora layer wahan hai hi nahi.
- **Encrypted-at-rest cache** — Android-only requirement.
- **`NluPackGuard`** — iOS mein iska kaam `PackIntegrity` (crypto) aur
  `BundleDataLoader` (structure) mein bata hua hai. Android par ise
  `PackIntegrity` + loader mein baant dena chahiye taaki naam match karein.

---

## 4. Port ka order

Value aur risk ke hisaab se:

| # | Kaam | Kyun pehle |
|---|---|---|
| 1 | Pack ko `v1.0.34` par le jao | Naya model + signed pack. Isse pehle baaki sab galat base par banega. |
| 2 | `PackIntegrity` | Pack signed hai; verification ke bina signature decoration hai. Download feature isi par blocked hai. |
| 3 | `PackTFIDFVectorizer` + OOV guard | Out-of-scope utterances ko device action tak pahunchne se rokta hai — measured 15.4% → 8.7%. |
| 4 | `NLUResponse` sealed shape + `ConfirmationGate` enum | Turn ka natija exhaustive; PTT confirmation bug isi ke saath theek hota hai. |
| 5 | `PackSlotResolver` + `SlotAnswerAssessment` | Slot flow abhi app ke andar bikhra hai. |
| 6 | `PackDateTimeParser` | Sabse bada (829 lines) par parity fixtures maujood hain. |
| 7 | `NLUSession` contexts, `deriveTopic`, `ClassificationBreakdown` | Baaki sab settle hone ke baad. |

Naming: iOS ke naam hi le rahe hain (`PackIntegrity`, `PackTFIDFVectorizer`,
`PackEntityExtractor`, `PackSlotResolver`, `PackDateTimeParser`, `NLUEngine`,
`NLUResponse`, `ConfirmationGate`, `ClassificationResult`), taaki dono side ek hi
review mein padhi ja sakein.

Iska matlab abhi ke Android naam badlenge:

| Abhi | iOS ke hisaab se |
|---|---|
| `NluManager` | `ResolvedPack` + `BundleDataLoader` |
| `OnnxIntentClassifier` | `PackIntentClassifier` |
| `OfflineNluServiceImpl` | `NLUEngine` |
| `OfflineNluResult` | `NLUResponse` |
| `NluKeywordMatcher` | `NLUEngine.matchKeywordTrigger` (iOS mein alag class nahi) |
| `NluGuards` | `ResolvedPack.helpRedirect` (iOS mein alag class nahi) |
| `NluTextNormalizer` | `PackTFIDFVectorizer.tokenize` ke andar |

Aakhri teen par ek raay: iOS ne inhe alag class nahi banaya, par Android mein alag
rakhna behtar hai — unit test karna aasaan hai. Naam align karne ke liye inhe
`PackKeywordMatcher`, `PackGuards`, `PackTextNormalizer` kiya ja sakta hai, jisse
`Pack*` prefix ka pattern dono jagah ek jaisa dikhega.

---

## 5. Verify karne ka tarika

`tests/datetime_parity/` mein en/fr/de/da fixtures hain aur
`examples/android/GoldenParityTest.kt` ka pattern maujood hai. Port ke baad dono
platform ek hi fixtures par chalein — tabhi "parity" kehna claim se zyada kuch
hoga. Abhi dono taraf ka comparison sirf code padhkar kiya gaya hai.
