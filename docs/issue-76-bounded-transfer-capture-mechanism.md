# #76 Phase 3a paired-input capture mechanism

This is capture-side support only. It does not authorize or perform a Unity
build, rendered capture, model fit, fresh access, or gameplay comparison.

## Frozen-player investigation

The v3 player reused by the v4 fixture already contains both required pixel
paths while the native decision barrier is paused:

- Corrected request 72 is handled in
  `/home/sukaih/.cache/novphy-shared-history-v3/project/Assets/Scripts/CanonicalCapture/PhysicsCaptureProtocol.cs:479-507`.
  Lines 494-497 call
  `PhysicsCaptureV2AlignedObservationRecorder.RenderCanonicalRgb`; that renderer
  is defined at
  `/home/sukaih/.cache/novphy-shared-history-v3/project/Assets/Scripts/CanonicalCapture/PhysicsCaptureV2AlignedObservationRecorder.cs:31-60`.
- The retained request-70 physics endpoint is handled in the same frozen source
  at `PhysicsCaptureProtocol.cs:518-539`. Lines 527-535 wait for the render-frame
  end, snapshot the active `PhysicalSnapshotRuntime`, and call
  `ScreenCapture.CaptureScreenshotAsTexture`. Its PNG is therefore the
  HUD-inclusive equivalent of the pre-v2 request-72 image, even though corrected
  request 72 now owns request id 72.
- The barrier records corrected world-camera frames after bookkeeping at native
  steps target-100, target-50, and target, then pauses physics at the target:
  `/home/sukaih/.cache/novphy-shared-history-v3/project/Assets/Scripts/CanonicalCapture/NativeDecisionBarrier.cs:56-83`.
  For the frozen target 30000 these are 29900/29950/30000.

Thus the frozen player can supply all three paired views today. Request 70 and
request 72 are sequential render requests, but both carry the same runtime
capture id, fixed step, and fixed time while the barrier prevents native physics
from advancing. No C# change or player rebuild is required.

## Additive Python mechanism

`scripts/issue_76_bounded_transfer_capture.py` wraps the byte-unchanged v4
collector. On its first paused corrected endpoint request it first obtains the
retained request-70 HUD-inclusive PNG, then request 72's corrected PNG, and seals
those with the existing aligned history before allowing the frozen collector to
continue to the shot.

The paired manifest binds member/base-cluster/engine-seed and a canonical
scenario digest to the runtime capture id, decision fixed step/time, and aligned
history capture id. It records:

1. `legacy-single`: exactly one request-70 `ScreenCapture` PNG at relative step 0;
2. `corrected-single`: exactly one request-72 canonical world-camera PNG at
   relative step 0;
3. `corrected-history`: three actual canonical frames at relative native steps
   -100/-50/0.

Single-frame assembly accepts exactly one real frame. It does not repeat or pad
that image: prior availability, elapsed time, per-slot motion, and motion
availability are zero. Every input frame must be at or before the decision step;
the exact history offsets are enforced, and any post-action frame is rejected.
Engine state returned with request 70 is used only to verify pairing and is not
an agent input.
