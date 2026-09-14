# Event collection display-startup diagnosis

The running event-development collection retains six known pre-shot timeouts:
006-a11, 008-a00, 011-a03, 011-a05, 011-a06, and 011-a07. All six display logs
contain the Xvnc fatal error `Cannot write display number to fd` (descriptor
7 or 8). These are unavailable captures, not negative gameplay targets.

## Isolated reproduction

On 2026-09-14, twelve sequential calls to the actual
`scripts.issue_76_live_episode.start_display` helper reproduced seven failures.
Each call started only Xvnc, waited 0.3 seconds after the helper returned, and
checked the returned process and display log. Surviving diagnostic processes
were terminated and waited for. No Unity player or gameplay capture was run.
Logs remain in `/tmp/novphy-display-start-repro-w2657lzj`.

A second experiment alternated twelve original calls and twelve calls with
only the pipe read changed to accumulate bytes through the terminating
newline. The original helper failed nine times; complete-line reads failed
zero times. Logs remain in `/tmp/novphy-display-pipe-compare-dt00b_8s`.
These temporary diagnostic logs are not a durable cohort artifact.

The observed original reads were either `b'8'` or `b'8\n'` (the display number
varied with availability). Every original call that read the number without
the newline in that experiment subsequently failed. Complete-line reads
observed the number and newline separately when necessary and remained live.

## Cause and boundary

`start_display` treats one `os.read(reader, 32)` as the entire display-number
message, strips it, and closes the read descriptor in `finally`. A pipe read
may return the number before Xvnc writes its newline. The early close then
causes the Xvnc fatal write failure. The immediate `process.poll()` check can
precede that failure and return a display whose server subsequently exits.

This identifies and reproduces the display-helper bug; it does not establish
that the production collection has been repaired. No frozen source has been
changed, failed assignment retried, or gameplay outcome replaced. The next
engineering step is a regression test for a split number/newline message and
a source-bound transition for the correction, preserving all existing
results and the original no-retry inventory. Dataset readiness and #76
advancement remain unproven.

## Separately tested correction

`scripts/issue_76_display_start.py` now provides the complete-line read with a
single 15-second deadline shared across all partial reads. The split-message
regression failed against the old helper, then passed against the correction.
Four tests cover split and complete messages, premature EOF cleanup, and the
shared deadline. Twelve actual Xvnc-only starts using the corrected module
all remained alive without the fatal error; logs are temporarily retained at
`/tmp/novphy-display-corrected-8wfk6y9u`. Diagnostic processes were cleaned up.
The original capture source binding still validates. This separate helper has
not yet been connected to the running collection.

## Controlled transition boundary

The original collector was stopped on 2026-09-14 after its active-worker list
became empty at 110 sealed attempts. SIGINT reached its between-wave sleep;
the process exited with code 130 and no active assignment was interrupted.
The terminal budget records 2,958.8943494241685 elapsed collection seconds,
`stopped=true`, and `supervisor_KeyboardInterrupt: ` as its stop reason.

A read-only audit confirmed that all 110 result files, supervisor receipts,
attempt directories, and sealed byte-accounting entries agree. There are 103
records marked complete, six startup timeouts, and one native-shot manifest
deadline failure with partial evidence retained. No receipt has a supervisor
stop reason. These counts describe collection status, not offline validity.
The old supervisor is absent and all 24 assigned worker ports are available.
Exactly 2,490 assignments remain unattempted; none of the 110 may be retried.

The stopped budget has not been reset or resumed. Continuation must first
archive a source-bound display-only amendment, preserve the boundary budget
and old receipts/results, bind new assignments to the corrected helper, and
carry forward elapsed time and retained artifact accounting. Original
lineages, actions, exposure roles, native clock, no-retry rule, and readiness
thresholds remain unchanged. No fitting or fresh access follows from this
engineering correction.
