# Shared-history synchronization fixture v1: retained failure

The isolated player built successfully (Unity exit 0, 142.6 s); 16 targeted
Python tests passed. The source/numeric plan was frozen before both rendered
attempts. Both assigned TRAIN-lineage attempts failed the strict RGB equality
check, and larger capture remains unauthorized. No shot or optimizer update was
executed and no fresh evaluation was accessed.

In attempt 01, all three history frames occurred at the prospective native steps
29900/29950/30000. Both public endpoint captures remained at step 30000 and fixed
time 13.1419992 s. The before/after two-second hold images were byte-identical.
The latest history and first endpoint differed in 6482 pixels (maximum channel
difference 246, mean absolute channel difference 1.3158214).

Visual/source inspection establishes a capture-path mismatch, not resumed
physics: the aligned history renderer uses `Camera.Render` into a render texture
and omits screen-space HUD, whereas request 72 uses
`ScreenCapture.CaptureScreenshotAsTexture` and includes it. The world looks the
same; menus/score/zoom overlays account for the visible difference. A tiny camera
size change (13.1249981 to 13.124999) is also retained, not asserted to explain
the mismatch. Both hold captures have the same camera state.

This also demonstrates a real observation-contract difference in the earlier
workflow (one endpoint decision image versus camera-only shot/training frames).
It is not proof that this caused the failed utility transfer or that repairing
the contract will confer hybrid advantage. Preserve all earlier conclusions.

Before another development fixture, unify the new isolated player's endpoint
and aligned render implementation. Keep the same fixed-step/RGB equality signal,
two assignments, zero-retry rule and resource caps. Record the correction as a
new version; do not overwrite v1 source plans, frames, receipts or results.

Full source-bound evidence is retained under
`.local-artifacts/issue-76-shared-history-smoke-v1/` (`plan.json`, `validation.json`,
`results/`, `receipts/`, `attempts/`). The build/source archive is
`/home/sukaih/.cache/novphy-shared-history-v1/`. These are local archives only,
not a pushed release or advancement evidence.
