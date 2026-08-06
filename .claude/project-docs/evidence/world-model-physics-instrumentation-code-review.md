# Code Quality Review: Unity Protocol Boundary

## Verdict

- `codeQualityStatus`: BLOCK
- `recommendation`: REQUEST_CHANGES
- `blockers`: synchronous response write can block Unity's main thread; JSON serializer can emit invalid JSON for control characters.

## Scope and Evidence Inspected

Reviewed the active isolated-worktree implementation and its actual recorded test XML, not only the done claim:

- `tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs`
- `tasks/task_template_designer/Assets/Tests/Editor/PhysicsCaptureProtocolTests.cs`
- `f2-unity-socket-stop-verification-3.xml` (2/2 passed)
- `f2-unity-socket-fixture-protocol.xml` (10/10 passed)
- `f2-unity-socket-editmode-partition-receipt.json` (37/37 passed)

The recorded SHA-256 identities in `f2-unity-socket-stop-verification-3.json` match the reviewed source and test files.

## Findings

### CRITICAL

None.

### HIGH

1. `PhysicsCaptureProtocol.cs:372` writes up to the declared 64 MiB envelope with synchronous `NetworkStream.Write` from the `Serve` coroutine, which runs on Unity's main thread. A loopback client can send request `70` and cease reading; once the TCP send buffer fills, this call blocks the frame loop indefinitely. The prior repair only replaces the pre-request blocking read with polling (`:343-352`) and its tests cover only silent clients and queued-client overflow (`PhysicsCaptureProtocolTests.cs:202-282`), not a non-reading response client. The direct-socket safety criterion therefore remains unmet.

2. `PhysicsCaptureProtocol.cs:248-261` only escapes quote, backslash, newline, carriage return, and tab. Other JSON control characters, for example `\\u0001`, are appended literally, producing syntactically invalid state/events JSON. Unlike the existing serializer at `PhysicalSnapshotJson.cs:130-158`, this writer does not emit `\\uXXXX` for control characters below U+0020. A valid capture value reaching `capture_id`, a participant, a reason, or a contact-derived field can make the whole request-70 artifact unreadable. No test exercises this value class.

### MEDIUM

1. `PhysicsCaptureProtocol.cs` has 357 nonblank/noncomment LOC, exceeding the remove-ai-slops 250-LOC module threshold; `PhysicsCaptureProtocolTests.cs` similarly has 327. The new production file combines binary envelope framing, JSON serialization, and socket lifecycle in one class/file, making changes to one boundary needlessly risk the others. Split by responsibility after fixing the blockers; do not extract a generic helpers module.

2. The two socket regression tests inspect private `listener`, `clients`, and `Serve` members through reflection (`PhysicsCaptureProtocolTests.cs:215-219`, `:320-330`). This implementation mirroring is brittle and makes legitimate encapsulation refactors look like regressions. Preserve the useful real-loopback scenarios but assert observable client behavior through a small intentional test seam or public lifecycle surface.

### LOW

1. The invalid-request branch builds the same failure envelope twice (`PhysicsCaptureProtocol.cs:354-355`). Store it once before writing; it is a small avoidable allocation.

## Skill-Perspective Check

The required `remove-ai-slops` and `programming` perspectives were explicitly loaded and applied. The diff violates the remove-ai-slops perspective through the oversized new source/test modules and violates the programming perspective through brittle implementation-mirroring socket tests. I found no deletion-only, prose-prompt, tautological, or production-constant-mirroring test in this scoped Unity protocol fixture.

## Test Assessment

The XML receipts substantiate the listed tests, but they do not cover either high-severity failure mode above. In particular, neither passes nor the reported 37-test partition exercises a request-70 client that withholds response reads, nor JSON control-character serialization.
