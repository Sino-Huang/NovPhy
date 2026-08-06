# Debug Journal - Bridge Bounds and Published Evidence

Started: 2026-08-06
Goal: Bound peer-controlled request-38 record counts and request-62 payload lengths before loops or reads, and bind docs verification to the current published receipt and accepted final smoke.

## Environment Snapshot

- Runtime: CPython 3.13.9, launched with `python`.
- Tests: `python -m unittest`.
- Git state: dirty with extensive pre-existing evidence and stage artifacts; preserve all unrelated edits.
- Owned tracked files: `src/webui/bridge.py`, `tests/test_webui_bridge.py`, `scripts/verify_physics_capture_docs.py`, `tests/test_verify_physics_capture_docs.py`, and `docs/physics_capture_v1.md` only if required.
- Protected: all other tracked files, the plan, ledger, Boulder state, published stage, and existing publication evidence.
- References read: debugging Python runtime; setup, investigation, fix, QA, cleanup methodology; programming Python README and typed error handling; full `.omo/plans/world-model-physics-instrumentation.md`.

## Hypotheses

1. [CONFIRMED BY SOURCE] Request 38 trusts the peer-controlled four-byte record count and enters `range(ground_truth_count)` without a cap. Distinguishing evidence: `ScienceBirdsBridge.shoot_and_record_ground_truth` reads the count at `src/webui/bridge.py:242` and loops at line 243. If true, the fix is: pre-loop cap.
2. [CONFIRMED BY SOURCE] Request 62 trusts the peer-controlled payload length and calls `_read_exact(payload_length)` before validating it. Distinguishing evidence: `_read_ground_truth` reads the length at `src/webui/bridge.py:255` and the body at line 256. If true, the fix is: pre-read cap.
3. [CONFIRMED] Docs verification read historical `task-8-smoke.json` instead of resolving the current final publication authority. The alternative stage-mismatch hypothesis was refuted: the staged archive, `archive.sha256`, final DoneClaim, publication receipt, and final smoke all bind archive SHA-256 `1c2a1bbcea87175150451ad8981e7f28ca09195be98f7da4cb8af577d431fef4`. The historical report binds `c7f9fa4c98480c1c1c8e580cb00454beda4fed4bf28a4822d31c561997906992`.

## Artifacts and Cleanup

- Retain: source/test fixes in the owned tracked files.
- Retain: all evidence created under this `bridge-docs/` directory.
- Remove: any temporary repositories, sockets, processes, or scratch files created by manual QA.
- Do not create: root debug journal, plan edits, ledger/Boulder edits, published-stage edits, commits.

## Findings

- Historical knowledge records `task-8-smoke.json` digest `c7f9fa4c...6992`; current final publication records `1c2a1bbc...fef4`.
- Current final DoneClaim has `status=complete`, publication receipt pointer, accepted runtime report pointer, request-38/62 compatibility, and protected-root success.

## Red Phase

- Bridge regression tests failed before implementation because request 38 entered the peer-controlled record loop, request 62 attempted the peer-controlled body read, and no typed legacy protocol error existed.
- Documentation regression tests failed before implementation because the verifier selected the stale historical task-8 report and did not validate the final DoneClaim/receipt/report chain.
- The failing-first tests were retained and passed unchanged after the implementation.

## Green Phase

- `python -m unittest tests.test_webui_bridge -v`: exit 0, 29 tests passed. Repeated twice after the final source annotations; both runs passed.
- `python -m unittest tests.test_verify_physics_capture_docs -v`: exit 0, 13 tests passed. Repeated twice after the final source annotations; both runs passed.
- `python scripts/verify_physics_capture_docs.py docs`: exit 0, `physics capture documentation verified`.
- `python -m py_compile src/webui/bridge.py tests/test_webui_bridge.py scripts/verify_physics_capture_docs.py tests/test_verify_physics_capture_docs.py`: exit 0.
- `python /home/sukai/.config/opencode/skills/programming/scripts/check-no-excuse-rules.py ...`: exit 0, `no violations in 4 file(s)`.
- `git diff --check -- <four owned tracked files>`: exit 0 with no output.
- LSP diagnostics were not available: basedpyright is not installed and installation was previously declined.

## Manual QA

- Used an explicitly synthetic, in-memory protocol fixture over a real ephemeral `127.0.0.1` TCP listener; no production server or dummy implementation was used.
- Request 62 emitted byte `3e`, read the valid seven-byte `{}xxxxx` legacy response, and decoded `{}`.
- Request 38 emitted `260000000100000002000000030000000400000001`, exactly `struct.pack("!Biiiii", 38, 1, 2, 3, 4, 1)`.
- A peer count of 10,001 produced `LegacyGroundTruthProtocolError(request_code=38, field="record_count", value=10001, limit=10000)` before reading the queued record body.
- Linux exposed the intentional poisoned-stream discard as `ECONNRESET`, proving the connection closed while unread bytes remained.
- A request-70 encoded envelope decoded PNG/state/events with a shared `render_frame=42`.

## Final Review

- Source SHA-256 values: bridge `8be91ea685049843ceee5b33b825b1df46cd6f859f8fa85da5cc1c963953af10`; bridge tests `fb2edaec9ace7bc7e8dcf09e7b1346c27cfd41281b17ea098b87d7c806f4973f`; docs verifier `463069b44a116b2947d8f7bf441cf878e45e54afd04171fc96fb4a65d0d7eea6`; docs tests `fd83c70a65b10d90fde9ea1ba90d604e42e17861950f3e29f64dbc1fdeaa9d33`.
- The final publication archive remained unchanged at SHA-256 `1c2a1bbcea87175150451ad8981e7f28ca09195be98f7da4cb8af577d431fef4`.
- Residual limitation: final smoke validation confirms the accepted-shot directory and digest chain but does not rerun full shot artifact validation.
- Residual limitation: request 38 caps record count but has no aggregate byte budget across all valid-sized records.
- Both residual limitations exceed the requested count/length and authority-selection remediation scope.
