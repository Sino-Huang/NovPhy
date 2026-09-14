# Prospective event-capture wall-time amendment

The corrected display helper is in use, but native-shot manifest deadlines
still make some otherwise progressing attempts unavailable. For example,
021-a03 recorded an attempted shot and decision observation, retained 566
frames, and failed with `native shot manifest deadline`. Its supervisor
receipt reports 214.36571637890302 total wall seconds, approximately 1,122.5
MiB peak RSS, and no supervisor resource stop. This differs from the earlier
display failures, which produced no decision observation or shot.

## Scope

After a controlled idle-wave stop and exact ledger audit, permit only these
wall-time changes for assignments that have never started:

- Shot-wait allowance: 180 to 360 wall seconds.
- Total attempt allowance: 420 to 600 wall seconds.

Keep eight isolated workers, 4 GiB per worker, 32 GiB aggregate, the cumulative
48-hour collection allowance, 768 GiB artifact cap, and 256 GiB free-space
reserve. Carry forward all elapsed collection time and retained artifacts.
The additional attempt allowance covers startup and publication around the
longer shot wait; it does not guarantee that every remaining capture succeeds.

The physical capture contract is unchanged: native step 0.0004 seconds,
maximum 30,000 steps / 12 physical seconds, RGB stride 50, and at most 601
frames. Do not extend a censored trace or replace a missing endpoint. Keep
every lineage, action, seed, exposure role, and assignment ordinal unchanged.
Continue using the tested complete-line display helper.

## Evidence and continuation

Freeze the terminal boundary budget, all existing results and receipts, the
remaining assignment IDs, and the implementing source before continuation.
Add a per-attempt resource-amendment record for the new execution period so
offline validation can distinguish the applicable wall-time limits. Earlier
display-amendment bindings remain in force.

All earlier failures and partial evidence remain unavailable; zero retries
or replacement assignments are allowed. The entire original 2,600-assignment
inventory remains the reporting denominator. The 90% valid-capture threshold
in every role/family cell and independent clear-lineage thresholds of 10/5/5
remain unchanged. No outcome-dependent cohort extension, fitting, fresh
access, or advancement is authorized by this resource amendment.

This document is prospective. The idle-wave stop, terminal audit, implementing
adapter, source freeze, and continuation must each be completed and verified;
writing this document alone does not change the live collection's limits.
