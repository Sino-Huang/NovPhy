# Night-window GPU scheduler

Owner rule, 2026-10-19 (binding; round-2 `CONVENTIONS.md`, section "GPU time window"): GPU jobs run only between
18:00 and 07:00 Australia/Melbourne. Daytime work is CPU only (`CUDA_VISIBLE_DEVICES=""`). Every GPU job is launched
by `scripts/gpu_window.py`.

## Window and tick

cron runs `tick` every 5 minutes:

```
TZ=Australia/Melbourne
*/5 * * * * /mnt/array/sukaih/Project/NovPhy/scripts/gpu_window_cron.sh
```

`scripts/gpu_window_cron.sh` activates the `novphy` env, sources `env.sh` from the repo root, and appends the tick's
output to `.local-artifacts/gpu-window/cron.log`. Launched jobs inherit this environment. Local time uses
`zoneinfo` (DST-aware). Whether a non-resumable job fits is checked in elapsed seconds, so DST nights are handled
correctly: the window is 1 h shorter on the first Sunday of October and 1 h longer on the first Sunday of April.

| local time | tick |
| --- | --- |
| 18:00–06:50 | Records ended runs. Logs foreign GPU processes (`nvidia-smi`, only inside the window). Launches every ready job in queue order; before relaunching a job it SIGKILLs any surviving process of the job's finished runs (`stray_killed`). |
| 06:50–07:00 | Sends SIGTERM once to every running job. Launches nothing. |
| 07:00–18:00 | Sends SIGKILL to every process still alive of any run of any job: `overrun_killed` for a running job, `stray_killed` for a finished one. Launches nothing. |

A job is *ready* when all of these hold:
- it is `queued` or `paused`;
- every `depends_on` job is `completed`;
- its GPU is free: no running job is on that GPU, unless one of the two jobs has `shares_gpu`;
- it is resumable, or its `est_minutes` end before 06:50.

A failed dependency blocks its dependents.

## Runs, signals and outcomes

- **Launch.** Each run starts in its own session (`setsid`) as
  `gpu_window.py _wrap --stop-at T1 --kill-at T2 EXIT_FILE -- COMMAND`. The wrapper forwards SIGTERM and SIGUSR1 to
  the command. When the command exits, the wrapper writes its exit code and duration to `runs/<job>.<n>.exit.json`.
- **Window enforced without cron.** T1 and T2 are the 06:50 and 07:00 that end the launch's window (DST-aware
  instants). On its own clock the wrapper SIGTERMs the command at T1 and SIGKILLs its process group and every
  process with its tag at T2. It logs both (`window_stop`, `window_kill`) to `events.jsonl` and the job log. The
  command gets at most one SIGTERM, whether from the wrapper or from the tick.
- **Identity.** The wrapper and every process under it inherit `GPU_WINDOW_RUN=<job>#<n>`. The scheduler finds a
  run's processes by this variable and signals no other process. At 06:50 it signals the wrapper only, so DataLoader
  workers never see the signal. At 07:00 it kills every process carrying the variable.
- **Outcomes.**
  - Exit 0: `completed`.
  - Exit 75: `paused`. The job checkpointed and resumes in the next window.
  - Any end after a scheduler stop or kill: `paused`. A schedule interruption is not a failure.
  - Any other exit: `failed`. A failed job is never retried. After a fix, `enqueue --requeue ID` returns it to the
    queue.
- **Restart safety.** Ticks hold an exclusive `flock`, and an overlapping tick skips. Stops are sent once per run.
  A job with live processes from a run that was never recorded is not launched again (`launch_skipped`).
- **Compute.** Each run's `seconds` is its active GPU time. Compute caps count the sum of these, not calendar time.

## Files (`.local-artifacts/gpu-window/`)

| file | contents |
| --- | --- |
| `queue.json` | jobs and their runs |
| `events.jsonl` | launch, stop, ended, overrun_killed, foreign_gpu_processes, … |
| `logs/<job>.log` | stdout and stderr of every run, with a header per run |
| `runs/` | exit files |
| `cron.log` | one line per tick |
| `lock` | the tick lock |

## Commands

```
python scripts/gpu_window.py status
python scripts/gpu_window.py dry-run            # what a tick would do now, then the projection under the tick rules
python scripts/gpu_window.py enqueue SPEC.json  # {"jobs": [...]}; an identical spec is a no-op
python scripts/gpu_window.py enqueue --requeue ID
python scripts/gpu_window.py enqueue SPEC.json --update-deps  # queued/paused jobs take SPEC's depends_on (under the lock)
python scripts/gpu_window.py stop-now [--kill]  # SIGTERM (SIGKILL) now; no launch before the next 18:00
python scripts/gpu_window.py --now 2026-10-19T19:00 dry-run
```

**Job fields:**
- `id`
- `gpu`: an index, or `null` for a job that needs no GPU slot
- `command`: an argv list
- `cwd`: relative to the repo
- `env`: added to the cron environment
- `depends_on`: jobs already in the queue or earlier in the spec
- `resumable`
- `est_minutes`: minutes of work left; required for a non-resumable job and used by the projection
- `shares_gpu`
- `note`
- `state`: set by the scheduler

A resumable job must checkpoint on SIGTERM and exit 75, or lose at most work it redoes deterministically. If it
needs more than 10 minutes to reach its next checkpoint, it loses that work at the 07:00 kill.

## #123 jobs (`scripts/gpu_window_issue_123.json`)

The jobs follow the #123 remaining command sequence:

1. **Training.** Queues 0–3 on GPUs 0–3 (`--train --gpu g`), resumable. Disclosed post-freeze amendment A1 of
   `run_hjepa_comparison.py`:
   - SIGTERM or SIGUSR1 makes the current arm-seed write `resume.pt` atomically (tmp + `os.replace`, as the
     2000-step checkpoints) after its current optimizer step; then the queue exits 75.
   - The next run resumes from that state: model, optimizer, scheduler and step. Batch *b* is `PCG64([seed, b])`,
     so the sampler position is the step.
   - The CPU and CUDA RNG states are restored after the DataLoader iterator is created. Its worker-seed draw had
     shifted the CPU stream of a resumed run.
   - `tests/test_issue_123_training_resume.py` checks this on CPU. The run is N steps uninterrupted vs k steps,
     SIGTERM, resume and N − k steps. Model, optimizer state, scheduler, CPU RNG and the final `model.pt` are
     bitwise identical.
   - On the GPUs, the restored state is exact. The kernels themselves (cuDNN autotune, bf16) are not bitwise
     deterministic run to run, with or without a pause.
2. **Tier T H-JEPA-3, seeds 42/43/44.** On GPUs 0/1/2 with `shares_gpu`, beside the shorter training queues.
   GPU 3 runs the longest queue alone. Each job runs
   `run_hjepa_fidelity.py --queue tier-t --jobs hjepa_l3:<s>` and resumes from the official
   `lightning_resume/last.ckpt`. Amendments A1 of `hjepa_train_ckpt.py` and `run_hjepa_fidelity.py`, and the
   Lightning resume semantics, are in "Tier T stop and resume" below.
3. **Tier E.** Finished (9/9, exit 0). If it is ever rerun: `planning_eval` writes one chunk file per task
   atomically, and `metrics.yaml` only after every chunk, which the queue checks. An interrupted task is redone and
   never counted.
4. **Readouts, freeze and scoring.**
   - `--readouts --gpu g` is resumable per arm-seed. `readouts.json` is written last and atomically, and an
     interrupted arm-seed is refitted under `deterministic_scoring()`.
   - `--freeze-readouts` takes about 10 min.
   - `--score` on GPU 0 is resumable per record. Records are written atomically, and an interrupted record is
     re-scored.
5. **Fidelity table, publish and validate.** `run_hjepa_fidelity.py --table` takes about 5 min and runs after the
   three Tier-T jobs. `--publish` (about 30 min) runs after the score and the table. `--validate` (about 60 min,
   GPU 0) runs after publish.

**#125 trigger before #123 scoring.** `125-trigger` is in this file, and `123-score` depends on it: #123's first
evaluation read waits for the #125 trigger, which reads the fit split only. If the trigger fails, scoring is blocked
until `enqueue --requeue 125-trigger` succeeds. `run_hjepa_idm.py --trigger` also refuses, writes
`trigger-refused.json` and exits 1 if #123 has read the evaluation split (records, evaluation anchors or position
cache) at its start or its end.

## Review fixes of 2026-10-19

These fixes land before the first live 06:50 stop (ops review M1 and note 1, spec review P1):
- Stray tagged processes of finished runs are killed at 07:00, and before the job is relaunched.
- The wrapper enforces 06:50 and 07:00 by itself.
- `123-score` waits for `125-trigger`. The live queue was updated with `enqueue scripts/gpu_window_issue_123.json
  --update-deps`.

The new code went live at the 19:00 tick of 2026-10-19. The stray kills apply from then on. The wrapper enforcement
applies only to runs launched after that, for example `123-fidelity-table` if it launches tonight. The seven runs
launched at 18:00 keep the old wrapper, so tonight's 06:50 stop and 07:00 kill for them still depend on the cron
ticks.

## Tier T stop and resume

These are amendment A1 of `scripts/hjepa_train_ckpt.py` and `scripts/run_hjepa_fidelity.py`. Both are registered
in `scripts/post_freeze_amendments.py`. The tests are in `tests/test_hjepa_tier_t_stop.py`: a toy Lightning
model through the real launcher, and the real `main_hjepa.py` at tiny scale, on CPU.

**Stopping.**
- The fidelity queue forwards SIGTERM to its H-JEPA child. It records the child's exit in the ledger, starts no
  further job and returns 75.
- A job that fails returns 1. So does a Tier-T job that ends without planning-eval metrics.
- On SIGTERM, the launcher saves the official `ResumeCheckpoint` state (`lightning_resume/last.ckpt`) through
  that callback's own atomic `_save`, at the end of the current train batch. The run then exits 75.
- A SIGTERM during validation or the epoch-end hooks also exits 75. The run then restarts from the last periodic
  save (every 2000 batches) or the epoch-end save.
- A SIGTERM during the train-end planning eval does not stop it. The 07:00 kill does. The finished per-task chunk
  files are atomic and stay on disk.

**Resuming.** The relaunch resumes through the official `ResumableManager`:
- Global step, optimizer and LR schedule continue exactly.
- The interrupted epoch runs only its remaining batches.
- The data order is **not** restored. The official loader re-seeds at launch, so the remaining batches are drawn
  from a new shuffle sequence. The official resume path behaves the same way after any preemption.
- If the resume file is from a finished run, Lightning skips its train-end hooks. The launcher then calls the
  skipped hooks once (`fit_loop.on_run_end`): the final object dump, then the planning eval, which resumes from
  its chunks.

## Incident 2026-10-21: open-file limit under cron

cron starts jobs with a soft `RLIMIT_NOFILE` of 1024. Interactive shells on this host run at the hard limit, 1,048,576.

The first `123-score` run (06:25 AEDT) crashed after 26 s in `build_eval_cache`. PyTorch passes tensors returned by pool workers as file descriptors, the 1024 limit ran out, and the pool broke: `RuntimeError: received 0 items of ancdata`, then `BrokenProcessPool`. No score was computed, and the cache file (written atomically) was never created.

Fix: `gpu_window_cron.sh` now raises the soft limit to the hard limit (`ulimit -Sn "$(ulimit -Hn)"`) before each tick, so every scheduled job inherits it. No runner code changed. The job was requeued for the next window with `enqueue --requeue 123-score`.

## Incident 2026-10-23: SIGSEGV in the first #125 publish

The first `125-publish` run (launched 01:15:07 AEDT) died with SIGSEGV (exit −11) after 25 s, during table derivation, after `records-manifest.json` was written. It left no Python traceback, so the native cause is unknown. This was an infrastructure crash, not a typed experimental failure: scoring had already finished, and publishing only derives tables from the frozen records, so no record was touched.

Fix: `gpu_window_cron.sh` now exports `PYTHONFAULTHANDLER=1` before each tick (`ed5acdd3`), so a fatal signal in any scheduled job dumps a Python stack into the job log. No runner code changed. The job was requeued once (`enqueue --requeue 125-publish`, 01:25:25). `125-publish#2` (01:30:08–01:32:10) exited 0, and `125-validate#1` re-derived and byte-compared every table (exit 0). Disclosed in `docs/issue-125-hjepa-idm.md` (Deviations 3).
