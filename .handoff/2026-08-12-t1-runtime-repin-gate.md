# T1 runtime re-pin gate — failure handoff

## Completed

- Implemented the authorized T1 producer/callback bundle in local WIP commit `d5be336be778103ac2ae883d4d946a1df3eaf540` (`wip: implement T1 runtime gate fixes`). Nothing was pushed.
- F1/F2: finalized `raw_contacts` and `support_edges` are canonicalized by the frozen parser keys before snapshot construction.
- F3: collision evidence resolves by exact canonical collider pair.
- F4: the recorder call is contained by a narrow no-throw physics callback boundary with a stable refusal log.
- F5/F6: capture model types moved to `PhysicalCaptureModels.cs`; production files are under 800 lines; write-only fields removed.
- F7: bounded retention keeps the last two full fixed steps plus collision-cited rows and updates byte accounting.
- ABEgg now records collision evidence before gameplay handling.
- Smoke refusal logs are scanned after engine teardown and fail the run closed; the mutation harness includes the new assertion.
- Added the required C#/Python regression fixtures and `finding-same-step-duplicate-ingestion.json` for the adjacent issue deliberately left open.
- Oracle phase-1 review passed after correcting the refusal-log fixture.
- Required C# ordering RED proof passed: `PhysicsShotRecorderTests` discovered 25 tests, 23 passed and exactly the two ordering regressions failed; NUnit XML sha256 `787cba1af6aae45c0dcb6ac14dde56ab09efe32a5597711d2ad6ffc78a5456db`. Source restored byte-identically to sha256 `d6bc41af198c986e8ce371131c617f2c0d125b88f02a30388bd33cdcc6d3a2cd`.

## Failures

The session stopped before Python RED/GREEN, full EditMode GREEN, mutation checks, deterministic builds, smoke, or re-pin because the packaging-blocking Python cache appeared after preflight.

Failing state check, verbatim:

```text
scripts/__pycache__ present
```

Exact blocker inventory:

```text
/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/scripts/__pycache__/physics_capture_contract.cpython-311.pyc
size=4284
mtime=2026-08-12T03:53:21.473677
sha256=59d38ecba469c7f588885debf4f1a709edc96227f6464c49cd8113c0d76e8d5b

/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/scripts/__pycache__/physics_capture_parsing.cpython-311.pyc
size=33680
mtime=2026-08-12T03:53:21.491228
sha256=360bb9c4ec4823ec6e00fc1900f85cd9786120d560132e88fe9be56897c3985d

/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/scripts/__pycache__/physics_capture_types.cpython-311.pyc
size=9345
mtime=2026-08-12T03:53:21.479177
sha256=622495dbca045c2d491006ce151d42504ae8539896e85c47d8d339684c251b85

/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/scripts/__pycache__/physics_rollout_contract.cpython-311.pyc
size=7193
mtime=2026-08-12T03:53:21.494910
sha256=61a6bb8588cfe4780a155d8cedaebb4c481bac454ac46d1a7077aaa6470944df
```

The bounded C# RED command itself returned a passing RED verdict:

```text
{"class": "PhysicsShotRecorderTests", "process_exit": -6, "editor_pid": 2009092, "reaped_package_managers": [2009156], "nunit_xml": "PhysicsShotRecorderTests.xml", "nunit_xml_sha256": "787cba1af6aae45c0dcb6ac14dde56ab09efe32a5597711d2ad6ffc78a5456db", "total": 25, "passed": 23, "failed": 2, "skipped": 0, "non_pass": [{"test": "PhysicsShotRecorderTests.Retention_KeepsLastTwoFixedStepsAndFinalizedContactsAreGloballyOrdered", "result": "Failed"}, {"test": "PhysicsShotRecorderTests.SupportEdges_FinalizedOrderFollowsSupporterSupportedSupportId", "result": "Failed"}]}
restored sha256: d6bc41af198c986e8ce371131c617f2c0d125b88f02a30388bd33cdcc6d3a2cd (identical)
overall verdict: PASS
```

No cleanup, verification continuation, build, smoke, re-pin, publication, cohort collection, or push occurred after the blocker was observed.

## Suspected Root Cause

The first long-running validation worker returned no output but left `session-4-verification/step-0-initial-hashes.txt` and refreshed full EditMode receipts. The four `.pyc` files share one timestamp during that unreported attempt. Suspected cause: that worker executed an import without the required effective `PYTHONDONTWRITEBYTECODE=1`, despite the task instruction. **Confidence: medium.**

## Next Action

Remove the generated `scripts/__pycache__/` directory, confirm it is absent, and resume Phase 2 at the Python RED proof.

Remaining gate sequence:

1. Python RED and GREEN verification.
2. Full per-class EditMode GREEN verification.
3. Mutation proof: 9/9 RED with byte-identical restoration.
4. Two deterministic builds.
5. One bounded full smoke.
6. Conditional re-pin only if the smoke accepts.

## Authority / Limits

- Conditional re-pin authorization remains unused and applies only after an accepting smoke.
- Publication is not authorized.
- Cohort collection is not authorized.
- The staged pin remains unchanged at sha256 `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de`.
- No commits from this worktree were pushed.
