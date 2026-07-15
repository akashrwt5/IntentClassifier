# Bundle signing key rotation runbook — DRAFT (pending ND-8)

Status: **DRAFT.** Becomes operational only after ND-8 (production keys in
KMS/HSM, custody decisions) is approved. Per ADR-005 Part 11 an unrehearsed
rotation plan is a plan to fail during an incident — this runbook must be
**rehearsed with dev keys** before GA, and the rehearsal date recorded here.

## Key model

- Ed25519 signing keys. The app binary pins **two** public keys: A (active)
  and B (standby). `BundleManager` accepts a signature from either pin
  (`trusted_pubkeys` list) — that overlap window is what makes rotation
  possible without a flag day.
- Signing happens ONLY in CI, key material in KMS/HSM-backed secret storage,
  gated on green evaluation gates + recorded human approval.
- The dev key (`spec/keys/dev/`, key id `dev-key-golden`) is deliberately
  public and refused by production runtimes. It never rotates and must never
  share material with production keys.

## Planned rotation (routine, e.g. annual)

1. Generate key C in KMS. Never export private material.
2. App release N pins {A, B} → release N+1 pins {B, C} (A dropped, C standby).
3. Wait for fleet saturation of N+1 per telemetry (bundle-verify events by
   pin id).
4. Switch CI signing from A to B for one release cycle. Bundles signed by B
   verify on both N (pins A,B) and N+1 (pins B,C).
5. After the deprecation window, retire A in KMS (disable, then destroy per
   retention policy). B becomes active, C standby. Record completion below.

## Compromise response (break-glass)

1. **Halt publishing** — freeze the CI signing job (single switch, owner: TBD
   under ND-8).
2. **Repoint fleet** via remote config to the last-known-good bundle signed
   by the SURVIVING key. Devices roll back on next config fetch; bundles
   already active keep working (they were verified at install — assess
   whether the compromise implies malicious bundles may exist; if so force
   rollback to baked).
3. **Emergency app release** rotating pins: drop the compromised key, pin a
   freshly generated replacement.
4. **Forensics:** signature-failure security events per `bundle_id`/pin id
   from telemetry; KMS access audit.
5. Postmortem + update this runbook.

## Invariants (enforced in code today)

- Two-pin verification: `BundleManager(trusted_pubkeys=[A, B])` —
  tests/test_bundle_lifecycle.py::test_unpinned_key_rejected.
- Signature failure ≠ corruption: security event flag — ::test_tampered_bundle_is_security_event.
- Downgrade protection stands during incidents: replaying an old validly
  signed bundle is refused without an authenticated rollback directive —
  ::test_downgrade_refused_without_directive.

## Rehearsal log

| Date | Scope | Outcome |
|---|---|---|
| — | not yet rehearsed (blocked on ND-8) | — |
