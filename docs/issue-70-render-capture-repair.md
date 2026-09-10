# #70: repair the graphics-disabled gameplay capture

## Cause and reproduction

The original v6 pilot has 48/48 `LegacyGroundTruthProtocolError: request-38
record_count: is truncated` results, zero completed shots, and zero videos.
All 48 Unity logs contain a native crash in rendering from
`PhysicalSnapshotRuntime.BeginV2Shot`. They also explicitly report
`Forcing GfxDevice: Null`, `Renderer: Null Device`, and `-nographics`.

The caller in `scripts/issue_70_live_pilot.py` passed `headless=True` to the
aligned RGB collector. The launcher interprets that as disabling graphics, not
merely hiding the window. The collector then calls `Camera.Render`, which crashed
the Null graphics device. The Java proxy remained connected, leaving the Python
client waiting for its 300-second response timeout. The protocol error was a
secondary symptom, not the initial cause.

The gallery was only an index of failed trials; its existence was not successful
capture evidence. Previous planner-only smoke checks did not exercise this live
Unity path and missed the mismatch.

## Fix and recovery contract

- Keep graphics enabled (`headless=False`) for the pilot. `--start-display` still
  provides an off-screen virtual display. No Unity rebuild or model retraining.
- Reject `headless=True` before launching the aligned RGB collector, with an
  actionable error instead of triggering the Null renderer.
- Add `--smoke-pilot`: fixed prior and corrected CEM on the first frozen level,
  in separate diagnostic output with WebM audit videos.
- Pause the full pilot after a new zero-shot capture failure. Existing failures
  are retained, never silently retried. Resuming skips their cached attempts and
  continues the frozen schedule. The gallery identifies partial runs and errors.
- Refuse to publish a wholly failed capture run as a gameplay comparison.
- Add `--repair-pilot` for this exact wholly uncaptured request-38 failure case.
  The immutable `pilot-render-repair-plan.json` binds the original frozen pilot
  and all 48 failed result snapshots. It explicitly discloses one additional
  technical attempt for **every** scheduled trial, with unchanged seeds, level
  membership, roles, models, policies and scoring. No outcome-conditioned selection.
- Corrective trials, game runtime/logs and results go under
  `.local-artifacts/issue-70-parser-repair-v6/experiment/pilot-rendered-v1/`.
  Old pilot data/logs are retained. Models and rankings are reused unchanged.
- Corrected gallery: `data/issue-70-pilot-audit-v6-rendered/index.html`.
  Corrected publication: `data/runtime_evidence/issue-70/action-design-v6-rendered.json`.
  The report includes the repair declaration and original failed results, rather
  than silently presenting the repeat as the first attempt. No new hashes.

## Verification

The regression test initially failed because the live caller requested headless
capture. After the fix, actual Unity captures succeeded:

- Fixed prior: one completed shot, 348 aligned frames, no capture error.
- Corrected CEM: one completed shot, 293 aligned frames, no capture error.
- Corrected CEM's physics trace contains launch, collision, destruction and
  stability events. Initial and mid-flight RGB frames were visually inspected.
- WebM inspection: 840x480, VP8, 50 FPS; the CEM video is 5.86 seconds and decodes
  without errors. These checks establish capture correctness, not planner success.

Review the two diagnostic videos at:
`data/issue-70-pilot-rendering-smoke-v1/index.html`.
They are not included in the corrective pilot comparison.

93 focused collection/planning tests passed, covering graphics enforcement,
failure-cache isolation, immutable correction routing, early pause, and rejection
of all-infrastructure-failure publication. A broader test attempt had two failures
in legacy smoke tests requiring the unavailable `sciencebirdsgames/physics-v1`
archive, plus one skipped test; those fixtures were not changed by this repair.

The correction plan is prepared. No full corrective pilot was run by the agent.
#70 stays open pending captured gameplay, video review and exact publication
validation. No final evaluation or #64 authorization is opened.

## Commands

Start the corrective pilot (no training steps are repeated):

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --repair-pilot --device cuda --start-display \
  2>&1 | tee -a data/issue-70-pilot-v6-rendered.log
```

After completion and reviewing the corrected gallery:

```bash
python -u -m scripts.run_issue_70_parser_repair \
  --publish 2>&1 | tee -a data/issue-70-publish-v6-rendered.log
python -u -m scripts.run_issue_70_parser_repair \
  --validate 2>&1 | tee -a data/issue-70-validate-v6-rendered.log
```

The repair plan automatically routes these commands to corrected outputs. Keep
the old artifacts: validation uses them to substantiate the disclosed correction.
To repeat the bounded diagnostic check, use `--smoke-pilot --device cuda
--start-display`; completed diagnostic captures are reused from their own cache.
