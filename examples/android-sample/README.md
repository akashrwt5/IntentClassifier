# Android sample — iOS-style pack download flow

Chhota sample jo dikhata hai ki `VoiceIntentKit` (iOS) wala architecture Android
par kaisa dikhta hai: **NLU module network ko haath nahi lagata**, app pack laata
hai, aur module verify karke load karta hai.

---

## Ek line mein farq

| | Purana Android | Ye sample (iOS jaisa) |
|---|---|---|
| Download kaun karta hai | `NluModelDownloader`, NLU module ke andar | `PackDownloader`, app ke andar |
| Kab download hota hai | module ne khud tay kiya | `PackSyncWorker`, app ke constraints |
| Verify kab hota hai | install ke baad, ya bilkul nahi | install se **pehle**, staging par |
| Seed pack (APK wala) | alag code path, kabhi verify nahi hota | wahi `PackSource`, wahi verification |
| Naya CDN URL chahiye | NLU module rebuild | app ka code |

---

## Flow

```
Application.onCreate
      │
      ├─ PackSyncWorker.schedule()        unmetered + battery-not-low + storage-not-low
      └─ locale receiver                  language badla -> packRevision++
                │
                ▼
        PackSyncWorker.doWork()
                │
                ▼
        PackRepository.sync(language)
                │
                ├─ PackDownloader.download()      ETag conditional GET -> staging file
                │                                  (sirf tab ETag bhejo jab pack installed ho)
                │
                └─ PackInstaller.install()
                        ├─ extract -> <lang>.staging/     zip-slip guard
                        ├─ PackIntegrity.verify()         ◄── THE GATE
                        │     signature over manifest ‖ bundle.json
                        │     dev-pack refusal (production build)
                        │     checksums_root == sha256(manifest)
                        │     har listed file ka digest
                        │     koi unsigned file nahi
                        └─ rename staging -> <lang>/      atomic swap
                │
                ▼
        packRevision++          consumers rebuild karte hain, turns ke BEECH
                │
                ▼
        AppPackProvider.packSource(lang)
                ├─ installed hai?  -> DirectoryPackSource(filesDir/nlu_packs/<lang>)
                └─ warna seed      -> AssetPackSource(assets/nlu_pack)
                                        dono ek hi PackIntegrity.verify se guzarte hain
```

---

## Files

| File | Layer | Kaam |
|---|---|---|
| `SampleApp.kt` | app | wiring, worker schedule, locale receiver |
| `pack/PackSyncConfig.kt` | app | base URL, archive naam (`PackDownloader.kt` ke andar) |
| `pack/PackDownloader.kt` | app | bytes laata hai. Trust ke baare mein kuch nahi jaanta |
| `pack/PackInstaller.kt` | app | extract → **verify** → atomic swap |
| `pack/PackETagStore.kt` | app | per-language ETag |
| `pack/PackRepository.kt` | app | sync lifecycle, `state`, `packRevision` |
| `pack/PackSyncWorker.kt` | app | kab aur kis condition mein |
| `pack/AppPackProvider.kt` | app | `PackProvider` ka implementation |
| `di/PackModule.kt` | app | **seam** — module ko sirf `PackProvider` + `PackTrustPolicy` dikhta hai |
| `ui/MainActivity.kt` | app | states dikhane ke liye |

NLU module se sirf ye use hota hai: `PackProvider`, `PackSource`,
`DirectoryPackSource`, `AssetPackSource`, `PackIntegrity`, `PackTrustPolicy`,
`PackLoadPolicy`, `VoiceIntentError`.

---

## Chaar cheezein jo dhyan se dekhna

**1. Verify install se pehle hota hai.** `PackInstaller.install()` staging
directory par `PackIntegrity.verify()` chalata hai, aur pass hone par hi rename
karta hai. Baad mein verify karte to kharab pack pehle hi live ho chuka hota.

**2. Seed pack special nahi hai.** `AppPackProvider` `AssetPackSource` lautata
hai, aur wo bhi usi verification se guzarta hai. Yahi wo bug tha jo Android mein
hua: purana pack `assets/` mein pada tha aur har turn serve ho raha tha, kyunki
bundled path kabhi verify hi nahi hota tha.

**3. Refused ≠ Failed.** `PackState.Refused` ka matlab hai bytes theek-thaak
aaye aur acceptable nahi the — usko retry karna wahi jawab dobara laayega.
`PackState.Failed` network hai, wo retry hota hai. `PackSyncWorker` dono ko alag
treat karta hai.

**4. ETag install ke baad store hoti hai.** Pehle store karte aur install fail ho
jaata, to device aisa content claim karta jo uske paas hai hi nahi, aur agla
conditional GET 304 deta — hamesha ke liye.

---

## Chalane ke liye

Seed pack assets mein daalo:

```bash
cp -R <pack-en-v1.0.34> app/src/main/assets/nlu_pack
```

Download on karne ke liye `PackSyncConfig.baseUrl` set karo. Blank chhodo to app
seed pack par chalta rahega — aur wahi default hai.

Signing keys `di/PackModule.kt` mein hain. Abhi placeholder `ByteArray(32)` hain;
asli raw 32-byte ed25519 public key daalni hai, `bundle.json` ke `key_id` ke
against. Production build `refusesDevelopmentPacks = true` par hai, to
`dev-key-golden` se signed pack release mein refuse ho jayega.

**Dependency:**

```kotlin
implementation("org.bouncycastle:bcprov-jdk18on:1.78.1")   // ed25519
implementation("androidx.work:work-runtime-ktx:2.9.1")
implementation("androidx.hilt:hilt-work:1.2.0")
implementation("androidx.security:security-crypto:1.1.0-alpha06")
```

`java.security` ka Ed25519 KeyFactory API 33 se hai, isliye BouncyCastle.
`PackIntegrity` low-level `Ed25519Signer` use karta hai, JCE provider register
nahi karta — koi global provider mutation nahi.
