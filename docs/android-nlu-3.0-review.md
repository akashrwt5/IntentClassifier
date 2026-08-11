# On-device NLU: format-3.0 client — review guide

Android side ka poora migration: 2.x wale do-file NLU (`nlu_schema.json` +
`nlu_entities.json`) se **format-3.0 pack** par.

Baarah Kotlin files — teen nayi stage classes, ek naya pack validator, aur aath
rewritten. Neeche har function documented hai: kya karta hai, aur jahan behaviour
obvious nahi hai wahan **kyun** aisa hai.

---

## 1. Ek page mein: kya badla

### Purana contract

Client do file padhta tha. `nlu_schema.json` mein thresholds, keyword triggers,
intent config, slots, prompts aur fulfillment text — sab ek blob mein.
`nlu_entities.json` mein entity values flat synonym arrays ki tarah. `NluManager`
har lookup par dono dobara padhta tha.

### Naya contract

Pack wahi blob **decompose karke** bhejta hai, aur wahi decomposed files client
ka contract hain:

| Kaam | File |
|---|---|
| Thresholds, limits, per-intent confirmation | `runtime/policies.json` |
| Ordered keyword pre-filter | `keywords/<lang>.json` |
| Yes/no words, contractions, datetime grammar | `lexicons/<lang>.json` |
| Help-marker + polarity redirects | `runtime/guards.json` |
| Intent → capability | `runtime/plan_facts.json` |
| Kaunse stage on hain | `runtime/cascade.json` |
| Action keys | `capabilities/<cap>/capability.json` |
| Slots, confirmation, completion action | `capabilities/<cap>/workflows.json` |
| User ko dikhne wali strings | `capabilities/<cap>/responses/<lang>.json` |
| Entity values (per-language synonyms) | `entities/shared/content.json` |
| Model, labels, calibration | `models/intent/<lang>/` |

`nlu_schema.json` aur `nlu_entities.json` ab bhi pack ke **andar** hain — reference
Python engine ke liye. **Client inhe padhe nahi.** Wo compiler ka input shape hai:
ek blob jismein platform config aur content mila hua hai, bina per-capability
versioning ke.

Isi split ki wajah se nayi language ab ek **content change** hai — sirf
`responses/<lang>.json`, `keywords/<lang>.json` aur `lexicons/<lang>.json` badalte
hain, client ka koi logic nahi.

### Teen breaking shape changes

1. **`system_messages` ab hai hi nahi.** Strings
   `capabilities/<cap>/responses/<lang>.json` mein hain, key se address hoti hain,
   aur key `workflows.json` se aati hai. Client ab string ke naam jaanta hi nahi.
2. **Entity values per-language nested hain**: `values.<Canonical>.<lang> = [...]`,
   pehle `values.<Canonical> = [...]` tha. Purane shape ke liye likha reader ko
   `JSONArray` ki jagah `JSONObject` milta hai, har value skip ho jaati hai, aur
   kuch bhi resolve nahi hota — **na crash, na log**.
3. **Slot ke naam badle.** `MemoryName` → `memory_name`. Purana naam khaali lautata
   hai, aur khaali "user ne kabha hi nahi" se alag nahi kiya ja sakta.

---

## 2. File map

```
com.starkey.device.features.voiceaikit.nlupack
  NluConstants.kt            har path, JSON key, threshold naam, log format
  NluPackGuard.kt            NAYA — tay karta hai ki pack use ho sakta hai ya nahi
  NluTextNormalizer.kt       NAYA — surface-form normalisation (TF-IDF path)
  NluKeywordMatcher.kt       NAYA — stage 1, ordered rule pre-filter
  NluGuards.kt               NAYA — help-marker aur polarity redirects
  NluManager.kt              pack loader + accessors
  OnnxIntentClassifier.kt    ORT session, logits -> calibrated probabilities
  OfflineNluServiceImpl.kt   the turn (decision ladder)

com.starkey.device.features.voiceaikit.download
  NluModelFileStore.kt       encrypted-at-rest pack cache
  NluModelDownloader.kt      conditional GET + validate + install
  NluModelUpdateManager.kt   startup sync, locale change, invalidation signal

com.starkey.device.features.smartassistant
  IOfflineNluService.kt      wo contract jispe VoiceAssistantService depend karta hai
```

---

## 3. `NluConstants.kt`

`internal object`. Har literal ka ek hi ghar: pack paths, JSON keys, thresholds,
fuzzy-matching tuning, log formats, rejection reasons.

**Paths constants se banti hain, templates se kyun nahi.** Project ka
`HardcodedString` lint rule string template ke andar literal text par fire karta
hai, par plain `const val` initializer exempt hai — to `"keywords/$lang.json"`
error hai aur `DIR_KEYWORDS + SEP + lang + EXT_JSON` nahi.

Rule se alag bhi fayda hai: pack path ka har segment ab ek jagah named hai.
"Client dhoondhta kya hai" ka jawab yahi file padhkar mil jaata hai, aur pack
layout rename hone par ek line badalni hai, aath templates nahi.

Baaki module ke liye thumb rule: **values interpolate karo, punctuation kabhi nahi.**

### Functions

| Function | Kya lautata hai | Note |
|---|---|---|
| `packArchiveName(lang)` | `"pack-en.nlu"` | Ek hi downloadable unit. |
| `downloadableFiles(lang)` | ek-element list | Purani chaar-file list ki jagah. |
| `plainOnnxModelName(lang)` | `"intent_model_en.onnx"` | **Per-language.** §15 bug 5 dekho. |
| `join(vararg segments)` | `/` se juda | Pack har OS par forward slash use karta hai. |
| `assetPath(path)` | `"nlu_pack/<path>"` | Bundled asset copy ke andar resolve. |
| `pathKeywords(lang)` | `keywords/en.json` | |
| `pathLexicons(lang)` | `lexicons/en.json` | |
| `pathCapability(cap)` | `capabilities/<cap>/capability.json` | |
| `pathWorkflows(cap)` | `capabilities/<cap>/workflows.json` | |
| `pathResponses(cap, lang)` | `capabilities/<cap>/responses/en.json` | |
| `pathOnnxModel(lang)` | `models/intent/en/model.onnx` | |
| `pathIntentLabels(lang)` | `models/intent/en/labels.json` | |
| `pathCalibration(lang)` | `models/intent/en/calibration.json` | |

### Dhyan dene layak constants

- **`REQUIRED_THRESHOLDS`** = `confidence`, `agreement`, `interrupt`.
  `oov_reject` / `oov_bypass` jaan-boojh kar required **nahi** hain: wo optional
  hain aur ek *pair* hain. Pack legitimately dono chhod sakta hai, aur unhe use
  karne wala guard khud ko disable kar leta hai. Ek ko required karna valid pack
  reject kar deta.
- **`THRESHOLD_UNREACHABLE = 1.1`** — jab koi pack load na ho tab lautata hai.
  Softmax probability 1.0 se upar ja hi nahi sakti, to koi ise paar nahi kar
  sakta. `0.7` jaisa plausible default rakhte to khaali workflow table ke against
  turn fire kar jaata.
- **`FUZZY_MIN_LEN = 5`** — isse chhote synonyms fuzzy matching se bahar. Car/care,
  Pub/pup, Gym/gum aam ASR output se ek edit door hain, to chhote memory names ko
  fuzz karne se **galat** memory select hoti hai, koi nahi ke bajaye.
- **`FUZZY_STOPWORDS`** — ek const string load par split hoti hai, kyunki `val`
  collection initializer const initializer nahi hota aur har element lint
  violation ban jaata.
- **`APOSTROPHE_VARIANTS`** — curly aur modifier apostrophes jo ASR bhejta hai,
  contraction expansion se pehle fold hote hain — warna `"what’s"` us table se
  miss ho jaata hai jo `"what's"` par keyed hai.

### Hataye gaye (reason file mein likha hai)

`FILE_NLU_SCHEMA`, `ASSET_NLU_SCHEMA`, `FILE_NLU_ENTITIES`, `ASSET_NLU_ENTITIES`,
`KEY_SYSTEM_MESSAGES`, `KEY_FULFILLMENT`, `ASSET_INTENT_MODEL`,
`ASSET_INTENT_LABELS`, aur chaar-entry wali `DOWNLOADABLE_FILES`.

---

## 4. `NluPackGuard.kt` (naya)

`internal object`. Tay karta hai ki pack use ho sakta hai ya nahi — **kuch bhi
padhe jaane se pehle**.

Do callers isse jaan-boojh kar share karte hain:

- **`NluModelDownloader`** fail hone wale pack ko *install* karne se mana kar deta
  hai. Reject hua pack disk tak pahunchta hi nahi, to pehle wala chalta rehta hai.
- **`NluManager`** jo *load* karta hai use dobara check karta hai. Download-time
  check akela kaafi nahi: contract 1 bolne wale build ka install kiya pack us app
  update ke baad bhi disk par hai jo contract 2 bolta hai.

**Dono mein se koi throw nahi karta.** Dono bundled APK assets par gir jaate hain,
jo ek sahi pack hai. Yahan throw karna "server ne kuch ajeeb bheja" ko har launch
ke crash loop mein badal deta, jabki sahi copy poore waqt APK ke andar padi rehti.

### `Result` (sealed interface)

- `Ok(entries)` — har zip entry `Map<String, ByteArray>` mein
- `Rejected(reason)` — sirf log hone wali string. Kaunsa reason tha ispar kuch
  branch nahi karta, kyunki reject hue pack ka ek hi sahi jawab hai.

### `inspect(archiveBytes, skip): Result`

Memory mein unzip karke structurally validate karta hai:

1. Archive unzip hota bhi hai ya nahi.
2. **`TODO(security)`: signature / `checksums_root` verification yahan aayega**,
   baaki har check se pehle. Abhi implement nahi — §16 dekho.
3. `bundle.json` maujood hai aur parse hota hai.
4. `format_version == "3.0"`.
5. `RUNTIME_CONTRACT` `engine_compat.min..max` ke andar. **Range**, equality nahi:
   1..2 ke liye bana pack contract 1 bolne wala client padh sakta hai, aur use
   reject karna us din har purane install ko phansa deta jis din server range
   chauda karta.

`skip` bade binaries ko map se bahar rakhta hai — jise model chahiye wo use path
se leta hai, yahan se nahi.

### `isComplete(entries, lang, modelRequired): String?`

Complete hone par null, warna reason. Checks:

- `runtime/policies.json` maujood, aur `REQUIRED_THRESHOLDS` sab maujood.
- `runtime/plan_facts.json` maujood aur parse hota hai.
- Pack jo bhi capability **declare** karta hai, uski teeno file maujood hon
  (`capability.json`, `workflows.json`, `responses/<lang>.json`).
- Per-language files: policies, entities, keywords, lexicons, labels,
  calibration, aur (sirf install time par) model.

**Sabse zaroori check yahi hai.** Missing `agreement` threshold zor se crash karta
hai, to pakda jaata hai. Purani `plan_facts.json` **koi error deti hi nahi** — har
intent ka workflow lookup miss hota hai, har turn GenAI par chala jaata hai, aur
ye bilkul aisa lagta hai jaise model kharab perform kar raha ho.

---

## 5. `NluTextNormalizer.kt` (naya)

`nlu_engine/text_norm.py::normalize_text` ka port.

**Kyun hai.** skl2onnx ka ONNX TfidfVectorizer apostrophe ke aas-paas Python ke
`\w` semantics replicate nahi karta, to exported graph aur sklearn ek hi string ko
alag tokenise karte hain. Fix — training **aur** inference dono mein — apostrophe
ko vectorizer tak pahunchne se pehle hi hata dena.

Practical asar: vocabulary **normalized** text se bani thi, to `"what's my battery"`
`what is my battery` ki tarah fit hui. Raw form tokenizer ko doge to wo
`["what", "s", "my", ...]` mein tootegi — `"s"` ka koi slot nahi, aur bigrams
`"what s"` / `"s my"` ka bhi nahi.

1470-row honest holdout par (full-vocab head): raw lowercase 0.9184, normalized
first 0.9204, 9 predictions alag.

**Scope sirf TF-IDF path.** Keyword stage raw text match karta hai, kyunki rules
usi ke against likhe gaye hain jo user asal mein bolta hai.

### `normalize(text): String`

lowercase → apostrophes unify → contractions expand → bache hue apostrophes hatao
→ whitespace collapse. Idempotent.

Contraction table **data** hai, `lexicons/<lang>.json` se. English wali inline mat
karna: doosri languages alag contract karti hain (fr `j'ai`, da `det's`), aur
English hardcode karna hi wo tarika hai jisse Python engine mein negation
suppression teen languages ke liye chup-chaap no-op ban gaya tha.

Alternation regex longest-first banta hai (taaki ek key doosri ko shadow na kare)
aur construction par ek baar compile hota hai.

---

## 6. `NluKeywordMatcher.kt` (naya)

Stage 1: hand-authored rule pre-filter, `keywords/<lang>.json` se.

Compiler pehle hi schema ke `exact` / `contains` / `regex` forms ko **ek ordered
regex list** mein flatten kar chuka hai, to client sirf order mein chalta hai. Wo
order load-bearing hai — first match wins — aur isi liye keywords ek single
top-level file mein aate hain, per-capability split nahi hote.

- tier 1 — anchored (`^mute$`), pehle `exact` trigger tha
- tier 2 — free regex
- `guards` — exclusion patterns; koi guard match kare to hit suppress ho jaata hai

### `match(rawText): Hit?`

Pehla fire hone wala rule, ya null. `Hit` mein intent aur tier.

**Rule confidence nahi deta, jaan-boojh kar.** Wo deterministic hai; probability
express kar hi nahi sakta. Jis engine ne yahan constant laga diya tha (`regex` =
0.75), wo ek banaya hua number un thresholds se compare kar raha tha jo calibrated
softmax probabilities par fit hue the. `0.75 < 0.91` arithmetic mein sach hai aur
matlab mein bekaar — isi wajah se saare 28 regex rules us din permanently
un-fireable ho gaye jis din wo band khiska, jabki "increase volume" 0.9992 par
confirmation ke liye roka ja raha tha.

~32 patterns construction par ek baar compile hote hain, har turn par nahi.

---

## 7. `NluGuards.kt` (naya)

Post-prediction redirects, `runtime/guards.json` se. Dono guards sirf ye badal
sakte hain ki **kaunsa** intent report hoga. Na koi intent invent karta hai, na
confidence badhata hai.

### `applyHelpGuard(rawText, intent, known): String`

**Feature kaise use karein ye poochhna use trigger nahi karna chahiye.** "how do i
turn up the volume" ek help request hai jise system ka har signal volume command
padhta hai: keyword rule "turn up" par fire karta hai, aur model ne questions se
kahin zyada commands dekhe hain. Jis state-changing action ka paired `Help_*`
sibling ho, aur utterance mein saaf question markers hon, wo sibling par redirect
ho jaata hai.

Read-only queries jaan-boojh kar paired **nahi** hain: "how do I check my battery"
poochhna aur battery check karna itne kareeb hain ki redirect ka nuksaan fayde se
zyada hai.

Redirect sirf `known` intents par hota hai — pack aisa intent pair kar sakta hai
jiski capability is bundle mein hai hi nahi.

### `applyPolarityGuards(rawText, intent, known): String`

Aisi prediction redirect karta hai jise utterance ke apne polarity words jhutlate
hain. Sirf tab fire hota hai jab **thik ek** rule match kare aur ulta cue maujood
na ho: "lower how LOUD it is" mein dono cues hain, model use pehle hi sahi resolve
kar leta hai, aur akele "loud" par guard fire karta to sahi jawab galat ho jaata.

Shipped English pack mein zero polarity rules hain, to abhi ye inert hai. Isliye
rakha hai ki pack bina client change ke rules add kar sake.

---

## 8. `NluManager.kt`

Pack load karta hai aur expose karta hai. **Construction kabhi throw nahi karta.**

### Ye kyun maayne rakhta hai

Pehle karta tha. Jis pack ke `policies.json` mein `agreement` threshold nahi tha —
`0ad05a7e` se pehle wala label space, jo assets mein ship ho gaya tha — usne app ko
`Application.onCreate` mein maar diya, main thread par, Dagger graph ke through.
"Voice commands kaam nahi karte" nahi: **app khulta hi nahi tha**.

Hearing-aid app ek content file se brick nahi hona chahiye. Isliye load fenced hai:
kuch bhi galat ho to `isReady` false rehta hai aur har accessor **inert** value
lautata hai, aur caller turn ko GenAI par bhej deta hai — bilkul waise hi jaise
low-confidence prediction par.

Inert values aise chune gaye hain ki aadha-load hua pack action fire na kar sake:

| Accessor | Inert value | Asar |
|---|---|---|
| `knownIntents` | khaali | kuch pehchana nahi jaata |
| `getWorkflow` | null | kisi ka action nahi |
| `fireThreshold` | 1.1 | koi confidence bar paar nahi kar sakti |
| `keywords` | koi rule nahi | koi rule intent naam nahi de sakta |

Inmein se har ek akele fallback force karne ke liye kaafi hai. Ye redundancy
jaan-boojh kar hai: ye wo raasta hai jo tab chalta hai jab hum pehle se jaante hain
ki kuch galat hai, to ek flag sahi check hone par nirbhar nahi hona chahiye.

### Public surface

| Member | Kaam |
|---|---|
| `language` | Jis language ke liye instance bana. Badle to rebuild karo. |
| `isReady` | False ⇒ offline NLU unavailable; caller GenAI par bheje. |
| `contentVersion` | `bundle.json` se. |
| `fireThreshold` | `thresholds.confidence` (0.70). |
| `agreementThreshold` | `thresholds.agreement` (0.50) — corroboration bar. |
| `interruptThreshold` | `thresholds.interrupt`. |
| `oovReject` / `oovBypass` | Optional pair; ONNX path par aaj dono null. |
| `maxSlotAttempts` | `limits.max_slot_attempts` (3). |
| `sessionTimeoutSeconds` | `limits.session_timeout_s` (120). |
| `knownIntents` | Pack jo bhi intent produce kar sakta hai. |
| `actionKeys` | App jo action keys receive kar sakta hai (→ `NLUActionKey.kt`). |
| `normalizer` / `keywords` / `guards` | Teen stage helpers. |
| `affirmative` / `negative` | Lexicon se yes/no word sets. |
| `temperature` | `softmax(logits / T)` ka T. |

### Functions

**`getCapability(intent): String?`** — `plan_facts.json` se.

**`getWorkflow(intent): JSONObject?`** — `getIntentConfig()` ki jagah. *Workflow*
lautata hai (completion action, slots, confirmation), purana schema blob nahi.

**`completionAction(intent): String?`** — jaise `memory.change`. Capability layer
ko isi par dispatch karna chahiye.

**`completionResponseKey(intent): String?`** — ek **key**, text nahi. Jo callers
`fulfillment` padhte the, wo ab ise padhkar `getSystemMessage` se guzarein.

**`getSlots(intent): List<Slot>`** — name, entity, required, promptKey. Slot ka
`name` hi parameters JSON mein use hone wali key hai.

**`requiresConfirmation(intent): Boolean`** — `policies.confirmation` padhta hai,
`workflows.json` nahi. Workflow aise intent ka confirm prompt rakh sakta hai jise
policy kabhi confirm nahi karti, aur policy hi fitted, reviewed artifact hai.

Confirmation **sirf wahan hai jahan kisi insaan ne per-intent likha ho** — is
taxonomy mein thik ek: `Cmd.SendMessage`, akela irreversible aur bahar dikhne wala
action. Koi confidence band nahi hai. Wo mechanism fire threshold ke **upar** baitha
tha aur honest holdout par 103 friction turns diye 16 useful catches ke badle:
user ne jo bhi confirmation dekha, uska 85% ek **sahi** prediction par poocha gaya
tha.

**`confirmPromptKey(intent): String?`**

**`getSystemMessage(key): String`** — `schema.system_messages` wale purane method ki
jagah. Key na mile to warning log karke khaali lautata hai.

**`materializedModelPath(): String?`** — `OrtSession` ke liye file path. ORT graph
mmap karta hai, to use real path chahiye; encrypted archive nahi padh sakta.

**`readLabels(): List<String>`** — pack unusable ho to khaali; classifier phir start
hi nahi hota.

**`resolveEntity(entityName, userInput, allowFuzzy = false): EntityMatch?`**

1. Longest synonym pehle, word-boundary match → confidence 1.00 canonical /
   0.95 synonym.
2. Edit distance, sirf tab jab entity `fuzzy` declare kare **aur** `allowFuzzy` ho.

`allowFuzzy` one-shot full-sentence scan ke liye **false** hona chahiye aur true
sirf tab jab user explicit slot prompt ka jawab de raha ho. Poore sentence par ek
awaara fuzzy hit wrong-action risk hai: "who is the prime minister of india" ek
baar stopword ke through enum se fuzzy-match hokar slot chup-chaap bhar chuka hai.

**No match par null lautata hai, `userInput` nahi — jaan-boojh kar.** 2.x contract
fail hone par raw utterance lautata tha, jisse unresolved slot resolved slot se
alag pehchana hi nahi ja sakta tha — "change memory to blargh" ne `MemoryName` mein
poora vaakya bhar diya aur app us par act kar gaya.

**`isOpenEntity(entityName): Boolean`** — true jab entity value list se bahar free
text bhi lete (jaise `remind`).

**`fuzzyMatch(table, text)`** (private) — teen tarah se gated, har ek ki keemat
chukayi ja chuki hai: multi-word synonyms skip (space ke aar-paar token distance
bemaani hai); `FUZZY_MIN_LEN` se chhote synonyms skip; edit budget synonym length
ka 30% taaki "restraunt" → "restaurant" ab bhi mile. Stopwords typo candidates se
bahar — "the" memory "three" ki galat spelling nahi hai.

### `Pack` (private inner class)

Jo kuch bhi throw kar sakta hai. Bahar wala constructor hi use catch karne ka
adhikari hai.

`packEntries` **sabse pehle declare hai, jaan-boojh kar**. Kotlin properties
declaration order mein initialize karta hai aur neeche ki har property usi se
padhti hai. Neeche khiskaya to un reads ke waqt wo null hoga, aur jis device par
pack download hai wahan bhi chup-chaap bundled load hoga — dikhne mein sahi
behaviour, par server-pushed content update permanently no-op.

Capabilities us list se chalti hain jo pack **declare** karta hai
(`plan_facts.json`), kabhi directory listing se nahi — listing chup-chaap purana
folder utha leti.

`synonymIndex` `values.<Canonical>.<lang>[]` ko synonym → canonical mein ek baar
flatten karta hai, primary language fallback ke saath: aadhi translate hui entity
"English mein match karti hai" par girni chahiye, "kuch bhi match nahi karti" par
nahi.

Thresholds **construction par ek baar** padhe jaate hain, per turn nahi. Getters
mein `getDouble(key)` hone ki wajah se missing key pehli utterance par
`Dispatchers.Default` worker se throw karti thi — process-level crash, classifier
ke naam par report, jabki wajah wo file thi jo startup se pehle hi disk par thi.

---

## 9. `OnnxIntentClassifier.kt`

### Jis bug ke liye ye file rewrite hui

Model `zipmap: False, raw_scores: True` ke saath export hota hai
(`nlu_training/train.py:260`). To `output_probability` ek **float tensor of
logits** hai — na label→probability map, na probabilities.

Purana `extractConfidence()` us output ko `Map<*, *>` maan raha tha, null milta
tha, null lautata tha, aur caller `?: 1.0f` laga deta tha.

**Har prediction 1.0 confidence report karti thi.** Kuch fail nahi hua, kuch log
nahi hua, aur argmax ab bhi sahi tha — to classifier theek dikhta raha jabki wo ek
number jispe poori routing ladder khadi hai, ek constant tha.

Confidence `softmax(logits / T)` hai, T `calibration.json` se. Temperature kabhi
un weights ke saath mat jodo jinpar wo fit nahi hui:

| File | Vocab | T |
|---|---|---|
| `calibration.json` | ONNX / server head | 0.671 ← yahi class |
| `intent_classifier_weights.json` | 1592 pruned | 0.822 |
| `intent_classifier_weights_full.json` | 5896 full | 0.544 |

Galat jodi compile hoti hai, chalti hai, aur **sahi intent** bhi chunti hai —
temperature rank-preserving hai, argmax kabhi nahi badalti. Sirf har confidence
galat hoti hai. Yahi blocker B8 hai; ek baar ship ho chuka hai.

### `OnnxPrediction`

`topIntent`, `confidence`, `labels`, `distribution`. Poori distribution carry hoti
hai, sirf top nahi, kyunki guard fire hone par ladder ko us intent ki probability
chahiye jispe wo **switch** kar sakta hai.

`confidenceOf(intent): Double?` — kisi ek label ki probability.

### Members

| Member | Note |
|---|---|
| `session: OrtSession?` | Usable pack na ho to null. |
| `isAvailable` | False ⇒ caller GenAI par bheje. |
| `labels` | Score vector ka class order; khaali matlab unusable. |
| `temperature` | Manifest se ek baar padhi. |

**`init`** — session options CPU arena aur memory-pattern planning dono off karte
hain (dono arena reuse ke liye hain, jo ek utterance ke chhote inference ke liye
sirf overhead hai) aur ek thread par pin karte hain (~1 ms single-threaded).

Labels na hon to **session banta hi nahi**. Aise model ko mat chalao jiski classes
ka naam hi na pata ho: argmax kisi aise index par girega jiske peeche kuch nahi.

Session **file path** se banta hai, byte array se nahi: byte array ke saath model
RAM mein do baar hota hai (Java `byte[]` + ORT ki copy); path se ORT weights mmap
kar leta hai.

Warmup **sabse aakhir mein** chalta hai, har property assign hone ke baad — warna wo
uninitialized property par throw karta hai, `runCatching` use nigal jaata hai,
warmup no-op ban jaata hai, aur cold start ka kharcha pehli asli utterance ke sar
par aata hai.

**`ensureModelOnDisk(): File`** — pehle downloaded pack, warna bundled asset ko APK
se copy karta hai (asset entries jagah par compressed hoti hain, mmap nahi ho
saktin). Fallback filename bhi **per-language** hai — wajah §15 bug 5.

**`classify(text): OnnxPrediction`** — `classifyInternal` ko
`android.os.Trace` section `ONNX_IntentClassification` mein wrap karta hai; Studio
profiler mein yahi string search karni hai.

**`classifyInternal(normalizedText)`** — input **pehle se** `NluTextNormalizer` se
guzra hona chahiye. Session absent/closed ho ya koi exception aaye to all-zero
prediction (⇒ fallback).

**`readScores(output): DoubleArray?`** — scores output padhta hai, **label output
nahi**. Batch size 1 by design hai (graph static batch of one ke saath export hota
hai). **Width mismatch par refuse karta hai**: agar graph aur pack alag-alag count
dete hain to dono alag builds se hain, aur argmax phir bhi *kisi* label par girta
jabki har downstream number kisi doosre intent ke baare mein hota.

**`softmax(logits, t)`** — max-subtracted, Python ke `_stable_softmax` jaisa.

**`close()`** — idempotent, null-safe.

---

## 10. `OfflineNluServiceImpl.kt` — the turn

`nlu_engine/engine.py::_handle_new_intent` ka port, DialogFlow-shaped
`OfflineNluResult` ke saath, taaki `VoiceAssistantService.onPVAResponse()` waisa ka
waisa rahe.

### Routing contract, poora

```
conf >= fireBar   ->  intent fire hota hai
conf <  fireBar   ->  Default Fallback Intent (app GenAI par route karta hai)
```

**Purane version mein fire test tha hi nahi.** Wo argmax leta tha, sirf ye check
karta tha ki label maujood hai, aur fulfil kar deta tha. Har utterance ek device
action banati thi — out-of-scope wali bhi, jinhe rokne ke liye hi 0.70 threshold
ship hoti hai. Ye isliye invisible tha kyunki confidence bhi hamesha 1.0 thi — to
us waqt threshold laga bhi dete to sab paar ho jaata.

### `classifyIntent(transcript): OfflineNluResult`

**Step 0 — availability.** `!isReady || !isAvailable` ⇒ turant fallback, taaki
kharab pack ek branch ka kharch ho, khaali label set ke against session run ka
nahi.

**Step 1+2 — arbitration.** Model har turn chalta hai aur **confidence** ka akela
author hai. Rule, jab fire kare, **label** ka akela author hai. Inhe alag rakhna hi
poora point hai: rule matlab ke baare mein hand-authored faisla hai aur probability
bana hi nahi sakta, aur us scale par number sirf model deta hai jispar thresholds
fit hui thin.

Model **normalized** text dekhta hai; rules **raw** text.

- Keyword hit nahi → model intent, model confidence.
- Keyword hit == model intent → *corroborated*, model confidence.
- Keyword hit != model intent → *contested*, confidence `0.60`.

`contestedConfidence = 0.60` **provisional hai — chuna gaya hai, fit nahi hua.**
Corroborated predictions honest holdout par 99.1% sahi hain; contested ~45%, yaani
sikka. Python ke `IntentClassifier.CONTESTED_CONFIDENCE` ke saath sync rakho jab
tak wo `train.csv` par out-of-fold fit na ho — holdout par kabhi nahi.

**Step 3 — guards, phir confidence dobara padho.** Ye re-read safai nahi hai. Guard
badal deta hai ki kaunsa intent report hoga, aur jo number wo saath le aaya wo us
intent ka hai jo **block** hua. "how do i turn up the loudness": rule
`Cmd.VolumeIncrease` propose karta hai, model `Help_Volume` kehta hai, help guard
sahi redirect karta hai — aur phir turn block hue action ki contested 0.60 leke
chalta hai, ek bilkul sahi help request bar ke neeche gir jaati hai aur GenAI par
chali jaati hai.

**Step 4 — fire test.** Har intent ke liye ek threshold. Slot-bearing intents ko
pehle kam bar milta tha is soch se ki prompt pehle ambiguity resolve kar dega — par
jis flow ke saare slots classifying utterance se hi bhar jaate hain wo turant
complete hota hai, to kam bar ek live action par lagta tha.

Ek hi exception hai — **corroboration**: do independent recognisers ka ek hi intent
bolna kisi ek se zyada majboot saboot hai, to bar 0.70 se 0.50 par aa jaata hai.
"turn it up its too quiet" wahi case hai — rule aur model dono VolumeIncrease kehte
hain, par "quiet" mass ko VolumeDecrease ke saath baant deta hai aur top class 0.66
par reh jaati hai.

### `fulfill(transcript, intent, workflow)` (private)

Har slot `allowFuzzy = false` se resolve hota hai (full-sentence scan). Unresolved
required slots `MissingSlot` bante hain, prompt **response key se resolve hokar** —
purana code fulfillment text par fallback karta tha, matlab memory ka naam maangne
ki jagah user ko "Memory changed." bolta tha.

Confirmation **tabhi** dekhi jaati hai jab har required slot bhar chuka ho:
aadhi-specified action confirm karna user se aisi cheez ki manzoori maangna hai
jise abhi koi bhi bayaan nahi kar sakta.

`allRequiredParamsPresent` false rehta hai jab slot missing ho **aur** jab
confirmation pending ho.

### `fallback(transcript, confidence)` (private)

Sirf routing decision leke chalta hai — na action, na fulfillment text. App ke paas
utterance ab bhi hai aur wo apni GenAI request khud banata hai; pack jaan-boojh kar
koi endpoint ship nahi karta.

### `resolveFollowUpSlot(entityName, transcript): String`

Ye dedicated follow-up turn hai: poora utterance ek hi sawaal ka jawab hai, isliye
fuzzy matching **yahan aur sirf yahan** safe hai. Closed-list match na mile to raw
transcript — open/free-text slots aur date/time jaisi system entities ke liye sahi.
2.x ka "fail par raw input" contract asal mein yahan ka tha.

### `resolveConfirmation(transcript): Boolean?`

Lexicon ke word lists se yes/no. Null matlab dono nahi, aur caller ko re-ask
**bound** karna hai: jisne mann badal liya use nikalne ka raasta chahiye, aur
har baar context reset karke hamesha poochte rehna ek infinite loop hai jo ye
codebase pehle ship kar chuka hai.

---

## 11. `IOfflineNluService.kt`

Teen methods (`classifyIntent`, `resolveFollowUpSlot`, `resolveConfirmation`) aur
do data classes.

**`OfflineNluResult` mein do naye field:**

- `requiresConfirmation` — authored confirmation pending hai.
- `action` — pack ka action key (jaise `memory.change`).

**Dispatch `action` par karo, `displayName` par kabhi nahi.** Capability
availability pehle label se derive hoti thi — `intent.startsWith(capabilityId)` —
jo `device.volume.mute` ko `device.volume` se match karti thi, aur labels `Cmd.*`
hote hi chup-chaap **kuch bhi** match karna band kar diya — matlab app ne jitni
capabilities unavailable push ki thin, wo sab wapas fire karne lagin.

`MissingSlot.prompt` **display text** hai, `responses/<lang>.json` se pehle hi
resolve hokar — key nahi.

---

## 12. `NluModelFileStore.kt`

Encrypted-at-rest cache — per language ek signed pack, AES256-GCM
(`EncryptedFile`/`MasterKey`), wahi pattern jo `CouchbaseOfflineEncryptionModule`
use karta hai.

### Ek archive kyun, extracted tree kyun nahi

Pack poore ka poora sign hota hai, to poora hi land karna chahiye. ~50 alag-alag
encrypted files mein extract karna wo aadha-likha state wapas le aata hai **jo phir
bhi parse ho jaata hai** — `bundle.json` pehli aur sabse chhoti entry hai, app us
par boot kar jaata hai, capabilities missing milti hain, aur asli commands fallback
par chali jaati hain bina kisi report ke. Uske saath zip-slip guard bhi chahiye
(input attacker-reachable hai) aur startup path par ~50 keystore-backed file opens,
ek ki jagah.

Chaar-file download bhi isiliye gaya: chaar alag fetch chaar alag content version
par land kar sakti hain — build N ka model, build N−1 ki responses — aur downstream
koi ise detect nahi kar sakta. Archive aadha apply nahi ho sakta.

### Functions

| Function | Note |
|---|---|
| `writeFile(lang, name, bytes)` | Pehle delete (`EncryptedFile` overwrite se mana karta hai), phir mtime explicitly bump — `materializeOnnxModel` isi par nirbhar hai. |
| `readBytes(lang, name)` | Absent ya undecryptable par null. **Null, exception nahi**: caller ka fallback bundled asset hai, aur throw karna ek corrupt cache entry ko us device par crash bana deta jiske APK mein sahi copy padi hai. |
| `readText(lang, name)` | `readBytes` par delegate — ek hi decrypt path. |
| `hasFile` / `hasPack` | Existence checks. |
| `installPack(lang, bytes)` | Patla wrapper, sirf isliye ki "signature pehle verify karo" ek jagah likha rahe. |
| `readPackArchive(lang)` | Raw archive bytes. Jaan-boojh kar "unzip karke do" helper **nahi** — unzip aur validate `NluPackGuard` ka kaam hai, aur yahan doosra unzip path banta to wahi hota jispe validation kabhi add nahi hoti. |
| `materializeOnnxModel(lang)` | Graph ko app-private plaintext file mein nikaalta hai taaki ORT mmap kar sake. Sirf jab archive naya ho. Temp-file + rename. |
| `pruneMaterializedModels(keep)` | Doosri languages ke plaintext models delete. |
| `clear(lang)` | Cached pack aur usse bani har cheez. |
| `getStoredETag` / `setStoredETag` | Conditional GET ke liye encrypted metadata. |

**Known limitation (pehle jaisi hi):** `OrtSession` real file path se banta hai
taaki weights mmap kar sake; encrypted stream nahi padh sakta. Isliye graph ek baar
app-private plaintext file mein decrypt hota hai — wahi on-disk exposure jo bundled
asset model ka pehle se hai.

---

## 13. `NluModelDownloader.kt`

`{BASE_URL}/{lang}/pack-{lang}.nlu` download, validate, install.
`BuildConfig.NLU_MODEL_BASE_URL` blank ho (default) to poori tarah inert.

### `syncLanguage(lang): Boolean`

True lautata hai jab naya pack install hua — caller ko `NluManager` rebuild karna
hai. Blocking OkHttp; Main par safe nahi. Yahan kuch throw nahi karta: fallback ka
matlab hi ye hai ki wo kharab server response se bach jaaye.

### `downloadIfChanged(baseUrl, lang)` (private)

Teen ordering decisions, har ek ek failure mode:

1. **`If-None-Match` sirf tab bhejte hain jab pack sach mein installed ho.** Orphan
   ETag — install fail hua, ya cache clear hua — server se hamesha 304 lata hai aur
   device kabhi recover nahi karta.
2. **Install se pehle validate.** `NluPackGuard.inspect` phir
   `isComplete(modelRequired = true)`. Reject hua pack likha nahi jaata, to purana
   chalta rehta hai.
3. **ETag sirf successful install ke baad store hoti hai.** Pehle store karo aur
   write fail ho jaaye to device aisa content claim karta rahega jo uske paas hai
   hi nahi.

Install ke baad `pruneMaterializedModels` chalta hai. Pichle pack ka plaintext
model jaan-boojh kar chhoda jaata hai — `materializeOnnxModel` mtime compare karke
agle classifier start par replace kar dega, taaki live `OrtSession` ke saath race
karti sync file ko uske neeche se na kheenche.

---

## 14. `NluModelUpdateManager.kt`

Do cheezein jo loaded pack ko invalidate karti hain, dono isi ke paas: naya
download, aur device language change.

### Ab revision publish kyun karta hai

Iske purane doc comment mein likha tha ki download mid-session complete ho to agle
PVA turn par asar ho jaata hai. Wo 2.x `NluManager` ka sach tha, jo apni do JSON
files har lookup par dobara padhta tha. 3.0 manager poora pack **construction par ek
baar** padhta hai, kyunki har turn archive dobara decrypt karna hot path par
daalne wali cheez nahi hai.

To ab apne aap kuch asar nahi karta. **`packRevision` hi rebuild ka signal hai.**
Iske bina downloaded pack process restart tak invisible rehta hai, aur download
feature chalte hue jaisa dikhte hue kuch bhi nahi karta.

Consumers ko **dono** rebuild karne hain — `NluManager` aur `OrtSession`. Sirf
manager rebuild karoge to classifier pichli language ke model par reh jaayega jabki
responses aur keywords switch ho jaayenge — intent ek language se, strings doosri
se, jo switch na karne se bhi bura hai. Rebuild turns ke beech, mid-utterance kabhi
nahi.

### Downloads off hon tab bhi language change invalidation hai

`NluManager` bhi `Locale.getDefault().language` construction par capture karta hai.
German par switch karo aur live instance English pack serve karta rahega: English
keywords, English responses, English model. Har label ab bhi valid hai aur har
confidence ab bhi plausible, to kuch surface hi nahi hota.

Isiliye receiver ab `downloader.isEnabled` ke **bina shart** register hota hai.
Pehle wo us check ke peeche tha, matlab shipping configuration mein — koi download
URL nahi, sirf bundled assets, jo default hai — language change se kuch hota hi
nahi tha.

### Functions

| Function | Note |
|---|---|
| `initializeApplicationService()` | `IApplicationService` entry point. |
| `start()` | Receiver bina shart register, phir enabled ho to sync. |
| `stop()` | Unregister. |
| `onLocaleChanged()` | **Sirf region badalne par kuch nahi karta** (`en-US` → `en-GB`) — un par revision bump karna `OrtSession` ko bewajah teardown aur rebuild karta. **Pehle invalidate, baad mein sync**: nayi language ka pack pehle se cached ho sakta hai, to consumers turant switch kar sakein, us network call ka intezaar kiye bina jo shayad enabled bhi nahi hai. |
| `syncCurrentLanguage()` | Background; successful install par invalidate. |

---

## 15. Integration ke dauraan mile bugs

Har ek evidence se mila, inspection se nahi.

**1. Confidence hamesha 1.0 thi.** `extractConfidence` zipmap `Map<*, *>` expect kar
raha tha; graph `raw_scores: True, zipmap: False` se export hota hai, to output ek
logits tensor hai. Null → `?: 1.0f`. Fix: logits padho aur `softmax(logits / T)`.

**2. Fire test tha hi nahi.** Argmax bina confidence dekhe fulfil ho jaata tha. Fix:
poori ladder port ki.

**3. Bundled asset pack do generation purana tha.** Runtime log dump se diagnose
hua: `confirmation` action ids par keyed (`device.memory.change`), values mein
`when_ambiguous`, thresholds mein `uncertain_confirm_below: 0.91`. Ye `a6cbb81c`
label space hai, `0ad05a7e` se pehle ka. `bundle.json` bilkul ek jaisa dikhta tha,
to usse farq pata nahi chal sakta tha — **`policies.json` hi discriminator hai.**
Fix: asset tree regenerate, aur `NluPackGuard` mein threshold + label-space checks.

**4. Kharab pack se app startup par crash.** `Application.onCreate` → Dagger →
`NluManager.<init>` → `JSONException: No value for agreement`. Fix: load fence karke
GenAI par degrade.

**5. Materialized plaintext model ka ek shared filename tha.** en se de par switch
karo aur mtime check de ke fresh archive ko en ke pehle se materialized plaintext se
compare karta hai; en wala baad mein likha ho to **German input par English model
chalta hai**. Har label ab bhi valid, har confidence plausible. Fix: filename
per-language, aur `pruneMaterializedModels`.

**6. `MemoryName` vs `memory_name`.** Format 3.0 ne slot rename kiya.
`performMemoryChange` purani key padhta tha, khaali milta tha, use "user ne kabha
hi nahi" se alag nahi kar sakta tha, dobara poochta tha, aur ek aise memory ke liye
"Memory changed." bolta tha jo kabhi badli hi nahi — aur mic ek aise sawaal par
khula chhod deta tha jiska jawab dene ki user ke paas koi wajah nahi thi. Yahi wo
dead turn hai.

---

## 16. Baaki kaam

### Faisla ya kisi aur team ki confirmation chahiye

- **PTT confirmation tooti hui hai.** `"Cmd.SendMessage - yes"` / `" - no"`
  dialogue-act labels hain jo classifier **kabhi emit nahi karta** —
  `legacy_labels.json` ka `confirm_compound` kehta hai ki client inhe resolved
  confirmation se banata hai. `handlePushToTalkResponse` `displayName` ko inse
  compare karta hai, jo sirf fail hi ho sakta hai, aur har record ki hui message
  cancel path par chali jaati hai. `resolveConfirmation()` use karo.
- **`MAX_SLOT_RETRY_ATTEMPTS = 2` hardcoded hai** jabki pack
  `limits.max_slot_attempts: 3` deta hai. Ek policy ke do number;
  `nluManager.maxSlotAttempts` use karo.
- **Sirf `Cmd.ActivityStep` explicitly handle hota hai.** Saat aur activity
  intents, chaar volume intents, `Cmd.FindMyPhone`, do streaming intents,
  `Cmd.TranscribeStart` aur `Cmd.TranslationStart` — sab
  `displayName.contains("Cmd")` → `PVACOMMAND` par gir jaate hain. Confirm karo ki
  command handler ye naam jaanta hai; miss hone par kuch log nahi hoga.
- **33 `Help_*` intents `isCMSContent()` se jaate hain.** Confirm karo ki CMS tags
  inhi naamon par hain.
- **`logIntentReturned` `R.string.command_*` se compare karta hai**, jo DialogFlow
  ke display names hain. Sirf analytics, par har intent shayad `NOT_SUPPORTED`
  bucket mein gir raha hai.

### Abhi wire nahi hua

- **`packRevision` ka koi consumer nahi.** Invalidation par `NluManager` +
  `OrtSession` rebuild karne wala code nahi hai. Downloads off hain to aaj sirf
  language switch par asar — jo aaj bhi pichli language ka pack serve karta rehta
  hai.
- **`NluManager` main thread par `Application.onCreate` mein banta hai** — archive
  decrypt aur ~50 JSON parse, har launch par. Lazy ya startup path se bahar karna
  chahiye.
- **Teen `NLU-DIAG` log lines hatani hain** jab pack loading settle ho jaaye.

### Jaan-boojh kar nahi kiya

- **OOV guard nahi hai.** Usko featurizer ki unigram vocabulary chahiye, jise Python
  graph ke apne `TfIdfVectorizer` node se padhta hai; ORT wo Kotlin mein expose nahi
  karta. `oovReject`/`oovBypass` expose hain par unused, jo "aadha pair na hone se
  bura hai" rule ke hisaab se sahi hai.

  Nateeja: aisi out-of-scope utterance jo model tak ek chhoti in-vocabulary phrase
  ki tarah pahunchti hai — "help me find a paper" `help me find` bankar pahunchta
  hai kyunki "paper" ka koi slot hi nahi — poori confidence se fire kar sakti hai.
  Ise band karne ke liye pack ko ek plain unigram list ship karni padegi; aaj wo
  data sirf 2.78 MB ki `intent_classifier_weights_full.json` ke andar hai.
- **Signature verification nahi hai.** `NluPackGuard` mein `TODO(security)` marker
  usi position par hai jahan wo chalna chahiye. `checksums_root` dev pack mein saare
  zeroes hai aur signing scheme decide nahi hui. **Jab tak wo na bhare, download
  feature production mein enable mat karna** — tab tak pack utna hi trusted hai
  jitna TLS aur CDN.

---

## 17. Pack verify kaise karein

`bundle.json` se current aur stale pack ka farq **pata nahi chalta** — dono
`format_version: 3.0` aur 12 capabilities declare karte hain. `policies.json`
dekho:

```bash
python3 -c "
import json
d=json.load(open('device/features/voiceaikit/src/main/assets/nlu_pack/runtime/policies.json'))
t,c=d['thresholds'],d['confirmation']
print('thresholds :', sorted(t))
print('confirm key:', next(iter(c)))
print('confirm val:', sorted(set(c.values())))
print('VERDICT    :', 'CURRENT' if 'agreement' in t and 'oov_reject' in t
      and not any(k.startswith('uncertain') for k in t) else 'STALE PACK')
"
```

Expected: thresholds mein `agreement` aur `oov_reject`, aur koi
`uncertain_confirm_*` nahi; pehli confirmation key `Cmd.ActivityAerobics`; values
sirf `always` / `never`.

Repeatable raasta — regenerate karke verify karta hai, aur dono taraf kuch galat
lage to copy karne ke bajaye refuse karta hai:

```bash
scripts/build_android_assets.sh
scripts/sync_android_assets.sh [ANDROID_REPO]
```

Assets APK mein bake hote hain aur incremental install unhe refresh nahi karta:

```bash
adb uninstall com.starkey.mystarkey.integration
./gradlew :app:engage:clean :app:engage:installMystarkeyWorldwideIntegration
```
