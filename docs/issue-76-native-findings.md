# #76 canonical-native engineering findings and next decision

The approved runtime/history prerequisite work is implemented in part; the
ten-item scientific workflow is **not complete**. No new model was fitted and
no fresh or sealed benchmark was opened. Canonical capture, observed-history
modules and their tests are implemented; the production 350-lineage collector,
shared fitting and paired controller/readiness integration are not ready.

## What ran, with original failures retained

The original game has a 0.0004-second fixed timestep. C2's original 600-tick
window therefore stopped before its one-second drag released. That failed
fragment contains 610 actual images (including watchdog overshoot), spanning
0.2436 seconds. It is preserved, not retried. The earlier aligned player and
all C1 attempts are also unchanged.

The separately frozen native continuation keeps that timestep, streams every
physics sample/contact/event in bounded gzip chunks and renders every 50 native
steps. It ran only C2's originally unattempted episodes 2–4, without replacement
or retry. Every native capture contains one actual launch.

| Assignment | Native samples | RGB images | Simulation duration | Capture result |
| --- | ---: | ---: | ---: | --- |
| appearance episode 2, shot 1 | 12,845 | 258 | 5.1376s | Complete stable stop; not a gameplay win |
| appearance episode 2, shot 2 | 30,001 | 601 | 12s | Failed: native time-window limit; shot 3 unattempted |
| normal rolling episode 3, shot 1 | 30,001 | 601 | 12s | Failed: native time-window limit |
| normal falling episode 4, shot 1 | 30,001 | 601 | 12s | Failed: native time-window limit |

All four native traces structurally validate, including full contiguous
coverage of the three failed prefixes: **102,848 native samples and 2,061 RGB
images**. Including C2, five captures actually started; the three unused
original shot allowances were not reassigned. No pig-death or level-clear event
was observed in these four native traces. Entity-death events must not be
mistaken for pig deaths or gameplay wins.

The cumulative C2-plus-continuation capture cost was 355.261 seconds active wall,
626,492,528 bytes of attempt artifacts, and 1,805.977MiB peak aggregate RSS.
These remain below the 30-minute/2GiB/3GiB limits. `budget.stopped=false` means
the global resource limit was not exhausted; it does **not** mean captures or
readiness passed. Player/build caches and source archives are separate from
this attempt-data measurement.

The final appearance timeout has only an 80-sample quiet recorded-body suffix,
short of the observer's required 100-sample persistence. Rolling and falling
still have a moving bird at the final sample. Thus this evidence does not
establish a stable-stop implementation bug, nor justify inventing a terminal.
The runtime scans all active Rigidbody2D objects; the offline quiet-body
diagnostic alone is not an equivalent runtime terminal detector.

## Implemented and verified components

- Recovered canonical normal/novel prefab lookup, original pig physics and
  distinct-startup-bird hit-count behavior; separate source/player recovery.
  The Unity 2019.3-to-2019.4 rebuild is explicit, not equivalence-certified.
- Ten native-player Unity fixtures passed, including resource/XML identities,
  distinct-bird counting, native clock/RNG, chunk boundaries and failed-prefix
  semantics. The native player built successfully. The C# emitted fixture
  also passed the Python reader; synthetic fixtures are not research members.
- Native collector uses isolated runtime/ports/display, graphics enabled,
  one worker, bounded RSS/storage/time, progress logs and no-replay resume.
- Final main report, all-prefix findings and retained-frame gallery validate
  against saved evidence without recapture. Gallery timing is recorded time;
  the short C2 fragment is explicitly 50x slow motion, with no generated frames.
- Shared 64-value observation/action GRU memory plus the 236-value visual
  carrier; both independent dynamics arms predict the full 300-value state.
  Pure dynamics have no symbolic heads/mode embedding. These remain unfitted.
- The new history dynamics use 50/250/750 native-step horizons, or
  20/100/300ms, and an 11,250-step/4.5s equal-time endpoint. Old 1/5/15-step
  models, players and checkpoints are unchanged. Shared fitting/deployment
  cost remains symmetric; parameter counts do not assert equal active compute.
- The latest focused Python suite passed 25 tests; the preceding broader
  prerequisite suite passed 160. These are separate runs, not a fabricated
  single combined test count.

## Requirement disposition

| Requested item | Current disposition |
| --- | --- |
| 1. Direction and symmetric engineering/refit costing | Operator approval recorded; runtime/history prerequisites implemented; production fitting integration still incomplete |
| 2. Exact protocol before rendered expansion | C2 and native continuation frozen prospectively; no production or fresh freeze |
| 3. Bounded graphics-enabled compatibility/power smoke | Engineering smoke executed, all failures retained; capture-readiness failed; no success-power claim |
| 4. Viable candidate, paired population, baselines, legal gameplay | Not established; no new fitted candidate or verified controller |
| 5. Fresh numerical/statistical/compute design | Not frozen; cannot infer power or practical margins from these engineering cases |
| 6. Commit/push and durable asset archival | Scoped source commits/push authorized and performed; archive destination still required; no archived release claim |
| 7. Once-only disjoint fresh inventory | Not generated; existing engineering lineages remain excluded |
| 8. All-system fresh gameplay/equal-time evaluation | Not run; no replacements, screening or privileged agent inputs introduced |
| 9. Wider novelty and adaptation effects | Untested; no appearance-only, paired novelty-effect, zero/few-shot or superiority claim |
| 10. Advance #64/#65 only after supported gates | Not authorized or advanced; #76 remains open |

## Reproduce the completed checks (no new capture)

Run from the repository root:

```bash
source /home/sukaih/miniconda3/etc/profile.d/conda.sh
conda activate novphy
source env.sh
set -o pipefail

python -u -m scripts.run_issue_76_native_continuation --dry-run
python -u -m scripts.run_issue_76_native_continuation --validate 2>&1 | tee -a data/issue-76-native-validation.log
python -u -m scripts.issue_76_native_findings --validate 2>&1 | tee -a data/issue-76-native-findings-validation.log
python -u -m scripts.issue_76_native_media --validate
```

The dry-run checks the frozen assignment definition; its description of unused
assignments is relative to the original C2 stop, not a claim that today's three
continuation assignments remain unexecuted. The actual run command skips their
recorded results. Do not delete results or prepare a new root to replay them.
Findings validation prints completed-episode progress. Full instructions for
the engine build/test commands remain in the earlier migration progress record.

Open local `data/issue-76-native-media/completed-episodes-4/index.html` for
playback. Numeric evidence is in `data/issue-76-native-findings/report.json`
and `data/issue-76-native-continuation/attempts-3/report.json`.
The earlier live `episodes-2` gallery snapshot did not validate and is retained
as such; use the finalized `completed-episodes-4` publication.

There is **no tested long production collection/refit command to hand off yet**.
The commands above validate completed engineering evidence only. Launching the
proposed 350-lineage job now would bypass the unresolved data/stopping contract.

## Decision requested before expanding collection

Recommended next revision: permit structurally intact fixed-window traces as
explicitly **censored segments**, supervise only endpoints actually observed,
never supply a synthetic terminal, and retain episode timeouts as gameplay
failures with the predeclared penalty. Missing/corrupt traces would still be
capture failures. This changes the current terminal-complete data contract and
must be prospectively approved/frozen; none of the C2/native failures above
would be relabeled, reused for fitting or replayed.

If approved, specify and test that new data contract, finish the common/paired
fitting integration, then freeze a distinct follow-up engineering smoke within
the three unused capture slots and remaining approved resource envelope. Those
would be new engineering assignments, not replacements or extra shots in the
completed episodes. Do not increase the native window or adjust actions based
on these outcomes. Leave full collection/fitting operator-run only after its
dry-run and bounded smoke actually pass, with foreground progress and ETA.

Also specify a durable destination for the large code/checkpoint/repair/player
assets and failed-run inventory. Git source pushes alone do not satisfy that
archive gate. Fresh evaluation remains blocked regardless of that choice until
the supported-candidate and complete scientific gates are satisfied.
