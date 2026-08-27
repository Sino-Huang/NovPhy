# Cohort-v2 aligned observations v1

Issue #59 adds a supplementary recollection for the frozen issue-17 visual
predicate stress test. It does not replace or amend the conclusions of issues
#15 or #16.

## Capture boundary

When `NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT` is set, the Unity physics-v2
runtime renders the main game camera immediately after every retained physics
fixed step, including the pre-intervention state and a forced terminal state.
The render and the physics frame therefore share one fixed-step identity. The
capture source is `synchronized_fixed_step_camera_render`; desktop screenshots
and asynchronous request-72 frames cannot satisfy this contract.

The collection bridge requires the ordered observation fixed steps to equal the
ordered `physics_capture_v2.frame_records` fixed steps exactly. Missing, extra,
duplicated, reordered, foreign-capture, or ordinary-screenshot records reject
the whole rollout. A successful collection publishes the images through the
existing observation-trace access boundary:

- agent RGB is available to the matching exposure-role model workflow;
- canonical RGB remains restricted to alignment or capture diagnosis;
- training, calibration, model-selection, and final-evaluation lineages and
  observation identities cannot cross roles.

Each aligned rollout retains the authoritative physics-v2 state and regenerates
the accepted contact, directed-support, steady-state, structure-unstable, and
endpoint-violation derivations. Camera matrices and collider geometry provide
the source-bound supervision needed to align visual object slots with engine
entity identities; neither is a permitted inference input.

## Release and use

The immutable release schema is
`cohort_v2_aligned_observation_release_v1`, with identity
`cohort-v2-aligned-observation-release-v1:issue-59`. It contains exactly six
complete rollouts in each of the four frozen exposure roles. The issue-59 reader
requires one validated source-role reader and exposes `load_frame_observation`
only for an exact retained frame.

The no-write preflight and foreground production commands are:

```bash
python -u -m scripts.issue_59_aligned_observation_collection --dry-run

python -u -m scripts.issue_59_aligned_observation_collection \
  --implementation-commit <IMPLEMENTATION_COMMIT> \
  --authorization-identity github-issue-authorization-v1:59:aligned-final-recollection

python -u -m scripts.issue_59_aligned_observation_collection --validate
```

Production prints each role/stratum slot and reports captured-frame progress at
five-second intervals within a rollout. Dry-run does not materialize scenario
outcomes, authorize final access, or write collection artifacts.
