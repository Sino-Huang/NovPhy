# F2 Unity socket manual QA

Editor: Unity 2019.4.41f2 (6b23d448b533)
Editor SHA-256: `32252cb8eca087743e500596e093061a906203703915c2d3c2fb2f8a372bc150`
Project: isolated migrated `tasks/task_template_designer`
Environment: pinned private `LD_LIBRARY_PATH` from `unity-2019-4-lts-resume.md`

## Silent loopback client

Scenario: attach `PhysicsCaptureDirectSocket` to an ephemeral loopback port, connect a real `TcpClient`, send no byte, and advance the real request coroutine.

Invocation: exact Unity EditMode filter `PhysicsCaptureProtocolTests.SilentLoopbackClientDoesNotBlockTheRequestCoroutineBeforeItsFirstYield`.

Binary observable before fix: the first coroutine step did not complete within 250 ms. Closing the socket released the blocked `NetworkStream.Read`; red XML result was failed.

Binary observable after fix: first coroutine step completed and yielded; a second step executed on the Unity test main thread in less than 50 ms. Focused XML result was passed in two independent runs.

Artifacts:

- `f2-unity-socket-red.xml`
- `f2-unity-socket-focused-1.xml`
- `f2-unity-socket-focused-2.xml`

## Over-capacity loopback burst

Scenario: attach to an ephemeral loopback port, connect eight real `TcpClient` instances, and observe server queue depth plus remote FIN state.

Invocation: exact Unity EditMode filter `PhysicsCaptureProtocolTests.LoopbackConnectionsBeyondPendingCapacityAreRejectedAndClosed`.

Binary observable before fix: pending queue depth was 8, exceeding fixed capacity 4; red XML result was failed.

Binary observable after fix: pending queue depth never exceeded 4 and at least four excess clients observed remote closure; focused XML result was passed in two independent runs.

## Compatibility and compile

The complete `PhysicsCaptureProtocolTests` fixture passed 10/10. All seven EditMode fixture partitions passed 37/37 with no C# compiler errors. The request-70 constant and envelope tests remained green. Requests 38 and 62 remain on the separate, untouched legacy connection path; the direct capture listener remains explicitly bound to `IPAddress.Loopback`.

The exact Editor's pre-existing headless CEF fault occurs after short runs save NUnit XML (`SIGSEGV`, process 134). The monolithic all-EditMode attempt faulted before saving XML, so it is not counted. Complete coverage is established from the seven passing fixture XML files instead.
