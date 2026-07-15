# DEV signing key — NOT a secret, NOT for production

This Ed25519 keypair signs **dev-channel** bundles only (`key_id:
dev-key-golden`). It is deliberately committed so every developer and CI
run produces byte-identical, verifiable dev bundles (ADR-005 stage 14/15).

Production runtimes categorically refuse artifacts signed with this key
(channel + key-id gate, ADR-005 Part 11) — see
`tests/test_bundle_build.py::test_production_runtime_refuses_dev_channel`.

Production signing (KMS/HSM-held keys, two-pin rotation, rehearsed
runbook) is decision **ND-8** in `docs/Review-F5/EXECUTION_STATUS.md` and
must never reuse this key material.
