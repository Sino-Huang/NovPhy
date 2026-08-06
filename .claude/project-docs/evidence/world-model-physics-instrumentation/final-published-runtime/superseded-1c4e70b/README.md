# Superseded publication authority (pre-repin)

These artifacts were the publication authority when HEAD was `1c4e70b` and the published archive was
`ed6f5972...`. They are retained unmodified for audit.

Two independent reasons they no longer apply:

1. The wave re-pinned to `f90c31e` after Phase 0 finding P0-1, which changed the published archive to
   `961d99d4...` and the stage receipt to `be0b9f6f...`.
2. `done-claim.json`, `publication-receipt.json`, and `smoke-report.json` here were hand-written after
   an interrupted run. They carry no invocation log, no exit code, and no protected before/after, and
   `smoke-report.json` lists state-*header* keys where a real run reports state-*record* keys.

`accepted-shot/` here is genuine, but it was produced against the superseded archive.

Current authority: `../done-claim.json`, `../publication-receipt.json`, `../smoke-report.json`,
`../accepted-shot/`, all derived from machine-produced artifacts in
`../final-wave-exact-final-native/f90c31e/`.
