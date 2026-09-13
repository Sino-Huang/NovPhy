# Parallel deterministic preparation, 2026-09-13

The fully automated workflow may use available CPU cores to prepare the remaining
native development records. The sequential job was using about one of 16 available
cores and about 1GiB RSS. This is an execution-only change, not a scientific
intervention, extra fitting or permission to change a failed result.

Before execution, cleanly pause the sequential job at its saved-record boundary.
Freeze a receipt with its entire current budget, exact completed-shard membership,
the original plan identity and this executor's source. Retain the base plan,
execution-amendment-v1, every raw collection attempt, and every saved shard.

Use six isolated spawned CPU workers, each with one torch thread and exactly one
assigned previously unprepared member. Use the original prepare_episode function,
same observation validator, same labels, same tensors, same ValueError/OSError
failure-to-empty-record convention, same atomic shard writer and original output
paths. A worker must never overwrite an existing shard. No new collection, labels,
entity truncation, time interpolation, endpoint substitution or data-role change.

The parent uses the same data-preparation budget with all prior elapsed cost and
the existing 24-hour ceiling. It checks aggregate process-tree RSS against the
unchanged 12GiB limit and working storage against 256GiB. Worker wall/CPU time and
peak RSS are separately recorded. Aggregate CPU time is not equated with wall
time, GPU time or FLOPs. Initialization and terminated-child CPU usage are charged
through parent child-process accounting. A process crash is an explicit stop, not
an automatic member replacement/retry. On a requested pause, stop every active
worker, preserve completed atomic shards and charge elapsed work; unfinished pure
derivations may be resumed later under the same cost record, never as a new raw
capture or hidden failure replacement.

When all workers finish, the ORIGINAL sequential _prepare_data function reuses and
validates every prepared shard against its original result and writes the index
in original order, with every assigned member and all 15 original per-family/role
90% coverage gates. Failed and unattempted collection rows remain present. No
whole raw-corpus scan or new hash pass is added. Fitting remains the original
paired-seed, source-bound workflow under run_issue_76_automated_refit.

Before switching, CPU fixtures must compare the new worker's successful and failed
shard contents against the original sequential routine, and confirm it refuses to
overwrite a completed shard. Publish the prospective execution amendment before
starting the worker pool. The advancement gate is unchanged.
