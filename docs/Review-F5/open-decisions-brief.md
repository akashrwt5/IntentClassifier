# Open owner decisions — brief for ND-8, ND-9, ND-12

Status: **Awaiting owner.** These three are business / legal / infrastructure
calls, not engineering blockers. Everything the codebase can do around each of
them is already built and waiting behind them. This document gives the context,
the concrete options, and the trade-offs so each can be decided quickly when the
owner returns to it.

Cross-references: the live queue lives in `EXECUTION_STATUS.md`; the signing
mechanics are in `docs/runbooks/key-rotation.md`; the GenAI routing design is
ADR-004; the bundle/signing design is ADR-005.

---

## ND-8 — Production bundle signing: key custody & rotation

**What it is.** NLU bundles (`.nlu`) are Ed25519-signed; the device refuses any
bundle whose signature doesn't match a pinned public key. Today everything is
signed with the **dev key** (`spec/keys/dev/`, id `dev-key-golden`), which
production runtimes deliberately reject. To ship a real release we need a
**production** signing key that lives somewhere trustworthy.

**What's already done (not blocked).**
- Full sign + verify pipeline (compiler stages 11–15), dev-key end-to-end.
- Two-pin trust model in `BundleManager` (active + standby) so keys can rotate
  without a flag day.
- A written, dev-key-rehearsable rotation runbook: `docs/runbooks/key-rotation.md`.

**The decision(s) owner must make.**
1. **Where does the private key live?** — Cloud KMS (AWS KMS / GCP KMS / Azure
   Key Vault) vs. a hardware HSM vs. a CI-native secret store.
2. **Who can trigger a signing?** — which people/roles approve, and is a second
   human approval required.
3. **Rotation cadence** — annual is assumed in the runbook; confirm or change.

| Option | Pros | Cons |
|---|---|---|
| Cloud KMS (recommend) | Private key never exportable; audit log built-in; CI integrates via IAM; cheap | Ties signing to that cloud vendor |
| Hardware HSM | Strongest custody; offline-capable | Ops overhead, slower, needs physical process |
| CI secret store only | Simplest | Key material is exportable — weakest custody; not recommended for a medical-adjacent product |

**Recommendation:** Cloud KMS with non-exportable keys, signing gated in CI on
green eval + one recorded human approval. Rehearse rotation with dev keys before
GA and record the date in the runbook. **Unblocks:** real production releases.

---

## ND-9 — GenAI / assist.cloud consent & legal review

**What it is.** When the on-device classifier is not confident, the engine can
route the utterance to a cloud GenAI assistant instead of failing (ADR-004).
That means user speech/text would leave the device. Before this is switched on,
the **consent flow and data handling need legal/privacy sign-off**.

**What's already done (not blocked).**
- The routing seam exists; when no GenAI URL is configured the fallback is a
  no-op (it just returns a FALLBACK result carrying no text — the app decides).
- Unknown-utterance logging already defaults to **counters only**; raw text is
  gated behind an explicit `NLU_COLLECT_RAW_UNKNOWN` opt-in (ND-5, done).
- Telemetry events are closed-enum with **no raw-text field by construction**.

**The decision(s) owner must make.**
1. **Consent model** — opt-in vs. opt-out, and at what granularity (per-session,
   per-feature, global setting).
2. **Data retention** — how long, if at all, cloud-routed utterances are stored,
   and where.
3. **Vendor & region** — which GenAI provider, and whether data residency
   (EU/US) constraints apply given the user base.
4. **Disclosure** — privacy-policy language + in-app disclosure copy.

**Recommendation:** Treat as opt-in, off by default, with a clear one-time
consent screen and no raw-text retention beyond the live request unless the user
separately opts into improvement data. This is a **legal decision** — engineering
should not pick it. **Unblocks:** the cloud-escalation feature going live.

---

## ND-12 — Phase-3 infrastructure hosting

**What it is.** Phase 3 (lifecycle + observability) needs three server-side
pieces to live somewhere: **remote config** (feature flags, rollout %), a
**bundle registry** (hosts signed `.nlu` bundles for OTA), and **telemetry
ingestion** (receives the on-device counter events). The *client* side of all
three is built; the question is what they talk to.

**What's already done (not blocked).**
- On-device: two-slot OTA lifecycle, telemetry aggregation module, availability
  snapshots — all client-side, done and tested.
- Telemetry event schema is finalized (closed-enum, no raw text).

**The decision(s) owner must make.**
1. **Host** — extend the existing app backend, stand up a new dedicated service,
   or use a vendor (e.g. a feature-flag SaaS + an object store + an analytics
   pipeline).
2. This choice determines how staged rollout, tripwires, and dashboards get
   built — so it gates the *server* half of Phase 3.

| Option | Pros | Cons |
|---|---|---|
| Existing app backend | Reuses auth/infra/on-call; fastest | Couples NLU lifecycle to app release cadence |
| New dedicated service | Clean ownership; independent scaling | New infra, new on-call, more setup |
| Vendor (flags SaaS + object store) | Least to build; dashboards for free | Recurring cost; data leaves your infra; per-vendor lock-in |

**Recommendation:** For a first production cut, host bundle registry on existing
object storage + CDN, remote config via a lightweight flag service (existing
backend or a small SaaS), and telemetry into whatever analytics pipeline the app
already uses — minimize new infra until rollout volume justifies a dedicated
service. **Unblocks:** the server half of Phase 3 (staged rollout + dashboards).

---

## Not in this brief

- **ND-13** (5 German intents with zero real German training data) — tracked
  separately; needs native German authoring or a reviewed MT pass, a
  content/translation task rather than one of these three infra/legal decisions.
- Mechanical owner-machine actions (dvc push tracking fix, local semantic-model
  regen, `make export-coreml`, iOS-repo fixture handoff, GitHub branch
  protection) — listed in `EXECUTION_STATUS.md`, not decisions.
