# Todo 8 Reproducibility Remediation

Verdict: **PASS for the bounded short gate; no 3,600-step run was started.**

## Cause and fix

The fixed-GPU four-cell matrix isolated `torch.use_deterministic_algorithms(True)` as
the causal toggle. Current settings and TF32-off-only diverged after the first CUDA
update; deterministic-algorithms-only and the strict bundle were bitwise stable.

Phase-A now defaults to an explicit canonical policy: cuBLAS workspace `:4096:8`,
deterministic algorithms enabled, matmul and cuDNN TF32 disabled, cuDNN deterministic
enabled, and cuDNN benchmark disabled. The policy is applied before Torch seeding/CUDA
work and is bound into the v2 Phase-A identity and fixture/real manifests. Checkpoints
persist the active CUDA-device RNG as a tensor and restore it through the restricted
`weights_only=True` loader. Old checkpoints remain loadable on CPU; CUDA resume without
a CUDA RNG tensor fails explicitly.

## Environment and identity

- HEAD at start: `c0ee19c5cb34e8e1afe35f25ef4d2f3217cc2055`.
- GPU: fixed physical GPU 0, NVIDIA GeForce RTX 5090, compute capability 12.0.
- Driver `595.71.05`; Python `3.11.15`; PyTorch `2.13.0+cu130`; CUDA runtime `13.0`;
  cuDNN `92000`.
- Catalog: 463 accepted episodes, digest
  `8265809a528e41eaae646cb1cae9d577d7f34fd99b85b859bb14f07a479c6beb`.
- Short configuration: seed 20260807, steps 18, batch 2, warmup 0, production model,
  strict config identity
  `5bce7161409264fb2a4d37d1602ec6403092b929f336908f4c6cc4da4067bd31`.

## Four-cell causal matrix

Invocation: two fresh processes per cell under
`CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
python -`, using the full production model, seed 20260807, batch 2, and two updates.
Each process recorded initial/pre/post model-section and optimizer digests plus loss to
`.omo/evidence/todo8-remediation/repro-short/05-four-cell.jsonl`.

| Cell | Paired result | Final model digests | Final optimizer digests |
| --- | --- | --- | --- |
| current | FAIL | `c6cf155c...` / `3a363054...` | `99ad3d57...` / `d82dafd3...` |
| deterministic algorithms only | PASS | `c7c6ee87...` both | `15d5e0da...` both |
| TF32 off only | FAIL | `53739a8c...` / `806bfcab...` | `f277b2bf...` / `78d58e57...` |
| strict bundle | PASS | `316a9baa...` both | `af65e118...` both |

Independent verdict: `.omo/evidence/todo8-remediation/repro-short/06-four-cell-verdict.log`.

## Real paired and resume gates

Invocation: fixed GPU 0 and the strict environment above, one read-only
`RealPhaseData` snapshot, then two freshly seeded 18-step trainers and an independent
18-step uninterrupted / 9-step prefix / two RNG-perturbed 9-step resume branches.
The inline driver emitted complete per-step JSON records for ordered state IDs, the
nine-key schedule, exact input tensors, host/CPU/CUDA RNG, pre/post model sections,
loss/LR/momentum, optimizer pre/post, and actual runtime settings.

All fourteen binary checks passed in
`.omo/evidence/todo8-remediation/repro-short/08-real-resume-verdict.json`:

- paired traces, logical model, optimizer, RNG, and checkpoint bytes are exact;
- every one of nine `(delta, regime)` keys occurs exactly twice;
- prefix steps 0-8 equal uninterrupted steps 0-8;
- both RNG-perturbed resume branches restore CUDA RNG and equal uninterrupted steps 9-17;
- resume state IDs, input tensors, pre-state, loss, and post-state are exact;
- duplicate unchanged-state archives are byte-identical.

Representative final logical digests: model
`5705ed0bec5d0ad4c3d768e53e502e1d8c0eb91605d8dbcc7942db093bbb7abb`;
optimizer `c89307573ec5132011573c3e328274c5c5d0d35e848b7d31b68d5db7c36c7426`;
CUDA RNG `3e6a20e8db6c4e2dd91d70814b4b91d4835a8c086c2c235e7e0bf1a2c6007c67`.
Paired checkpoint SHA-256 is
`2aa1e5afdd49c2fe64b3cd92c8125e727b52069dd07a38e38aacdf0dfb5b0051`;
unchanged prefix saves both hash to
`4085ad07cfe025fb9f143adb7ffa955f6317607741d106df815d64d8d8789c37`.

Independent field/trace verification is retained at
`.omo/evidence/todo8-remediation/repro-short/09-independent-verifier.log`.

## Tests and cleanup

- Red: `python -m unittest -v tests.test_world_model_reproducibility` failed because the
  reproducibility module did not exist (`01-red.log`).
- Green: the same focused suite passes five tests, including real CUDA RNG round-trip and
  missing-RNG CUDA rejection (`04-green-focused-v2.log`). The strict-policy and CUDA RNG
  round-trip tests now run in child interpreters with the cuBLAS workspace configured
  before Torch import, so they remain valid after an enclosing suite has initialized CUDA
  without weakening the production pre-context guard.
- Adjacent: grid-run, restricted checkpoint security, and real-data tests pass 11/11
  (`03-adjacent.log`).
- Full relevant: the exact 12-module surface passes 348/348 in a fresh process after all
  production edits (`repro-final-20260809/10-green-full-348-after-all-edits.log`). The failing-first rerun captured the two
  suite-order errors before the isolation fix (`repro-final-20260809/01-red-full.log`),
  and the focused post-fix suite passes 5/5 (`repro-final-20260809/02-green-focused.log`).
- The final strict-type audit reports zero violations and the focused suite remains 5/5
  after replacing the erased batch value type
  (`repro-final-20260809/05-no-excuse-final.log` and
  `repro-final-20260809/06-green-focused-after-type-fix.log`).
- `compileall`, `git diff --check`, and LOC gates passed; the new product module is 135
  nonblank/non-comment lines and the focused test is 149. Ruff and basedpyright are not
  installed, so those tools could not be run.
- Temporary checkpoint inventory is retained in `10-temp-checkpoint-inventory.log`.
  `/tmp/novphy-repro-short-cdKlR5` was removed after independent verification
  (`11-cleanup.log`). Old runs and the protected dataset were not written.

This is a temporal-only reproducibility result. It does not alter unavailable symbolic or
physical claims and does not establish the 3,600-step acceptance result until the bounded
gates are reviewed and a new full run is explicitly authorized.
