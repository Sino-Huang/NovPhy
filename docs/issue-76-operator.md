# #76 implementation, requirement coverage and operator handoff

## Latest operator-run native collection/refit stage

The approved censored-policy smoke has passed, and the new collection/refit
commands are implemented, frozen and dry-run/CUDA-smoke tested. See the
[native operator handoff](issue-76-native-operator.md) for the actual long-run
commands, progress logs, failure/resume rules and remaining scientific gates.
Those long research jobs have not been run. The earlier commands below describe
historical stages and do not replace this new workflow.

The Phase A workflow is implemented. The operator completed the full diagnostic;
saved-evidence validation passed after the CSV correction documented below.
#76 is not yet closed. Protocol: [frozen diagnostic definitions](issue-76-diagnostic-protocol.md).

Final interpretation and missing-comparator reporting are documented in
[findings](issue-76-findings.md); [the working checklist](issue-76-todo.md)
separates completed closeout from conditional, unexecuted experiments. The
separate `scripts.run_issue_76_closeout` publication/validation commands there
preserve the original diagnostic artifacts.

The subsequently requested broader-coverage work has its own
[compatibility findings and commands](issue-76-compatibility-findings.md).
All eight C1 attempts completed and validated, with failed runtime/perception
readiness. This does not unlock fresh evaluation or replace the original stop.

## What each part of the long ticket now does

| Ticket requirement | Implementation / disposition |
| --- | --- |
| #73/#74/#75 inputs and preserved negative evidence | Exact source-bound plans, completed checkpoints and #75 repair receipt checked. |
| Dynamics training versus symbolic execution versus adaptation | Twelve fixed systems (three pure continuous, nine hybrid pairs), with original/covered adaptive controls separately reused at225. |
| Error accumulation versus local accuracy | Recursive15/30/60/150/225 curves and separate observed-context local probes; no intermediate truth resets in action scoring. |
| Temporal/perception alignment | Original endpoint parser batch convention, immediate predecessor motion, B=1 initial agent carrier and same-image cache audit; stable absorption/unavailable targets explicit. |
| Metrics, masks, task outcomes and uncertainty | Fieldwise and aggregate carrier errors, parsed count errors, settled action regret/top-k/outcomes, headroom and descriptive paired-state contrasts. |
| Capacity/compute/cost | Frozen matched #74 fits, actual operator MACs/calls/wall, source #74/#75 costs, local/target overhead separated; no equal-FLOP claim. |
| All physical scenarios/novelties | Metadata inventory of80 templates,40 counterparts and45 cells with constraints, authored slots, action direction, exposure/observability warnings. |
| Wider population recommendation | Costed, metadata-selected normal rolling/falling/sliding and appearance-pair candidates; no performance-based selection. |
| Larger interface/collector changes and rendered compatibility smoke | Not executed; need explicit approval and exact role/seed/runtime freeze. Static fit is not parser or gameplay success. |
| Fresh numeric protocol, power/precision, partition audit and once-only experiment | Not executed behind the pre-access readiness stop. No invented sample sizes, power result or completed fresh-test claim. |
| Fresh graphics, isolated engines, gameplay success/shots and physical validity | Conditional on an eligible fresh protocol; no new capture/engine path was launched. Existing source frames/videos are linked for review. |
| Archive before fresh access | Not complete: source and large artifacts remain local/uncommitted/unpushed; no archive claim. |
| Advance #64/#65 | Not authorized. #75 failed its candidate screen; an explanatory curve cannot override it. |
| Publication/validation and stop handoff | Machine-readable full/compact reports, CSV/SVG/HTML review and a24-entry requirement ledger. Complete diagnostic or explicit budget-stop validation precedes closure review. |

The final disposition for this Phase A execution is
`readiness_or_precision_insufficient` **before fresh access**, not an executed
fresh negative experiment. That disposition does not suppress the diagnostic
results: all fixed/model/controller comparisons and all seeds are reported.
It does not imply a general impossibility result for hybrid world models.

## Commands (run sequentially)

Initialize `novphy` and source `env.sh`, then:

```bash
python -u -m scripts.run_issue_76_dynamics_diagnostic \
  --prepare \
  2>&1 | tee -a data/issue-76-prepare.log

python -u -m scripts.run_issue_76_dynamics_diagnostic \
  --inventory \
  2>&1 | tee -a data/issue-76-inventory.log

python -u -m scripts.run_issue_76_dynamics_diagnostic \
  --run-diagnostic --device cuda \
  2>&1 | tee -a data/issue-76-diagnostic.log

python -u -m scripts.run_issue_76_dynamics_diagnostic \
  --publish --device cuda \
  2>&1 | tee -a data/issue-76-publish.log

python -u -m scripts.run_issue_76_dynamics_diagnostic \
  --validate --device cuda \
  2>&1 | tee -a data/issue-76-validate.log
```

Preparation and inventory have been tested here and can be repeated. Preparation
only freezes settings; it intentionally does not score images or run models.
The diagnostic covers24 preselected states, three seeds, twelve fixed systems
and twelve actions:10368 candidate records. Expect tens of minutes rather than
new training or a long game-collection job; the actual command reports progress,
elapsed/ETA, seed, state, system, candidate and failures.

One cumulative3600-second active-work allowance,2GiB derived artifacts,3GiB CPU
RSS and2GiB CUDA allocation. Targets and candidate records resume from completed
atomic files. Repeat the same command after an ordinary interruption. A complete
inventory is not rescored by a repeated run; use validation for checks. A hard
kill may lose work since the last saved unit; no outcome-conditioned retries.

If a budget stop occurs, do not increase/delete the counter or restart with a
favorable subset. Run `--publish` and `--validate` to record the partial diagnostic
and declared stop. `diagnostics_complete=false` stays visible; validation is not
a certificate that incomplete experiments were completed.

## Output and review

`.local-artifacts/issue-76-dynamics-diagnostic-v1/`:

- `plan.json`, `novelty-inventory.json`, `diagnostic-access.json`;
- `targets/`, `fixed/`, `budget-diagnostic.json`;
- `result.json` (full), `summary.json` (without per-state detail), `smoke.json`.

After publication, open `data/issue-76-review/index.html` for error curves,
ranking/work summaries, the45-cell inventory, all requirement statuses and links
to original agent frames/WebMs. `curves.csv` provides exact per-seed field values;
`manifest.json` binds existing source images/videos. No new/edited video is claimed.
The review's logarithmic error plot floors zero at1e-12 for display; CSV retains
the exact values. Adaptive intermediate curves remain unavailable.

The metadata inventory found27 templates meeting static slot/action constraints,
48 with missing authored slots and five further action-direction failures. These
are NOT passed runtime tasks. Normal falling `type010104` exceeds six platforms,
but other falling templates fit; the proposal chooses the first static-compatible
normal template per added scenario before any performance measurements. Current
recommendations are normal `type010103` (rolling), `type010204` (falling),
`type010105` (sliding), plus the normal/novel `type010102` appearance pair.
New collector packaging and rendered perception checks are still required.

## CSV validation correction

The original validator compared CSV text containing CRLF line endings with
`read_text()`, which normalizes them to LF. This incorrectly reported
`review/CSV/source linkage differs from publication` even when the CSV bytes
matched exactly. Validation now compares exact UTF-8 bytes; numerical results
and the published files are unchanged.

For an existing run frozen before this correction, record the explicit source
amendment once before validating:

```bash
python -u -m scripts.run_issue_76_dynamics_diagnostic --repair-validation --device cuda
python -u -m scripts.run_issue_76_dynamics_diagnostic --validate --device cuda \
  2>&1 | tee -a data/issue-76-validate.log
```

`validation-csv-repair.json` binds the exact original and corrected runner source.
It preserves `plan.json`, diagnostic records, results and review artifacts. It
does not authorize changes to other frozen sources, definitions or inputs. No
diagnostic rerun or republication is needed. The CLI publish/validate regression
test covers the actual CSV file round-trip and still rejects a corrupted CSV.

The corrected CUDA validation passed on 2026-09-10 with
`diagnostics_complete=True` (24 states, three seeds, 10,368 fixed candidate
records). Output is appended to `data/issue-76-validate.log`. All 22 issue-specific
tests passed. Fresh evaluation remains unrun and #64 remains unauthorized.

## Bounded verification already performed

```bash
python -u -m scripts.run_issue_76_dynamics_diagnostic --dry-run
python -u -m scripts.run_issue_76_dynamics_diagnostic --smoke-test --device cuda
python -m unittest tests.test_issue_76_dynamics_diagnostic -q
```

The real smoke checks both current families, low/high-bound actions1/12, all12
fixed systems, all five endpoints/local probes, exact cached target/control
alignment, target resume, strict JSON and the source-linked review writer.
All four original/covered adaptive policies on both states are also validated
against actual cached costs/work traces. Temporary derived smoke data is removed;
the compact smoke evidence is retained and is not production evidence. No new
weights, training, game capture, checksums or full-image integrity scans are added.
