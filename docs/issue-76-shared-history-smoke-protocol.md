# #76 shared native history synchronization fixture

This is a prospective development transport fixture, not a fresh advancement
cohort or an outcome-based attempt to pass #76. No optimizer updates are allowed.
The canonical recovered parent source, player, datasets and failed findings remain
unchanged. The new build changes observation timing and synchronization only;
physics timestep, physics rules and recovered gameplay behavior are unchanged.

Version 2 corrects the retained v1 capture-path failure: request 72 and aligned
history/shot renders use the same world-camera RGB implementation, excluding
screen-space HUD. This shared observation correction applies to every model arm;
it neither changes physics nor grants scene/oracle information to policies. V1
source, numeric plan, both failed attempts and findings remain retained.

Version 3 sets the same public-readiness fully-zoomed-out camera size exactly
before history rendering. V2 preserved the native clock but a final tiny camera
interpolation altered world-edge pixels. This projection correction changes no
physics or strict equality threshold. V1/v2 failed evidence remains retained.

Version 4 makes one capture-path correction: the shared-history collector uses
the newline-complete display starter. The v3 compiled player is reused unchanged.
V3's failed case-02 and all of its partial evidence are retained and not retried;
v4 writes to a new fixture root. The failure was a single-read EPIPE race: Xvnc
writes its display number exactly once at startup in two writes (digits, then
newline), and closing the pipe after the digits can break the newline write.

Before either rendered attempt, freeze the source, compiled player origin and two
assignments in `plan.json`. Select the first existing TRAIN member in normal family
`type010204` by membership order, regardless of its outcomes. Replay that same
generation/engine seed with fixed drags (-80,10) and (-60,45), no tap. Declare the
recovered player's one-second drag delay (requested release time 1000 ms).

After all native step bookkeeping, render three RGB frames at level-relative
native steps 29900, 29950 and 30000 (20 ms spacing). Pause before subsequent manual
physics. Policy callbacks receive only transformed shared RGB and observation
timestamps. Engine scene/object/body metadata remains outside the policy. A shot
must capture its first pre-intervention sample/RGB at the paused decision state,
then release the barrier. No candidate or future frames are supplied predecision.
The first shot image is already the latest history observation: insert accepted
action history once at that timestamp, without a duplicate zero-duration RGB
event, then consume only subsequent real shot frames.

The timing signal is exact fixed-step/time equality and byte-identical canonical
RGB across the latest history frame, public actuator readiness, a two-second
inference hold, and the chosen shot's pre-intervention frame. Both branches must
include all authored objects and have initial world-position/velocity absolute
differences <= 0.0001. Retain all native launch offsets and genuine terminal or
intact 12-second censored windows; neither success nor failure selects a retry.
Any timing/branch mismatch blocks larger capture, not just model scoring.

Envelope: one worker, two shots maximum, 1800 s global wall, 420 s per attempt,
180 s shot manifest deadline, 90 s history readiness, 8192 MiB process-tree RSS,
2 GiB unique-inode artifact bytes, 256 GiB minimum free disk before each attempt,
zero technical retries. All failures/partial artifacts and unattempted assignments
are retained. This fixture does not establish training coverage, useful utility,
matched full compute, statistical power, fresh readiness, or advancement.

Runtime asset storage hardlinks only immutable player binaries, transport jar and
`9001_Data` assets outside `Levels`. Level XML, config, root controls, runtime logs
and XDG paths are private per attempt. Charge shared inodes once and disclose that
this is unique logical file bytes, not filesystem allocated-block accounting.
Do not modify or delete old evidence to fit the new resource envelope.
