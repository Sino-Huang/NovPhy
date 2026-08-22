# Issue 48 replay determination audit

These directories preserve earlier determinations for audit only. They are not
members of, inputs to, or substitutes for the canonical accepted bundle at
`data/runtime_evidence/issue-48`.

- `determination-1-failed`: failed intervention and observation checks under the
  initial policy.
- `determination-2-failed`: failed intervention and observation checks under the
  revised initial policy.
- `determination-3-superseded-relative-action`: passed a superseded policy that
  re-derived interface coordinates; it is not evidence for exact socket replay.
- `determination-4-failed-continuous-equality`: failed a policy that incorrectly
  required fresh Unity runs to reproduce every continuously valued measurement.

Determination 5 is the sole canonical publication. It replays each immutable
original socket command exactly, preserves the additional issue-48 observation
and full-physics authorities, and binds code revision `14afe6d`.
