# Shared-history synchronization fixture v2: retained failure

Version 2 unified the isolated player's request-72 and aligned camera renderer.
Unity build exited 0 in 133.6 s; nine directly related tests passed before source
and two assignments were frozen. Both TRAIN-lineage attempts still failed the
same strict RGB equality check. No shots, optimizer updates or fresh access.

Attempt 01 latest history, before hold and after hold all remained at native step
30000 and fixed time 13.1103992 s. Before/after hold PNGs were byte-identical. The
latest history camera size was 13.1249981; both endpoint sizes were 13.124999. The
render-time field moved from the fixed-loop time 13.1103992 to the final rendered
frame time 13.1220617 without another manual physics step.

The latest-history/first-endpoint difference dropped from v1's 6482 pixels and
mean channel error 1.3158214 to 1282 pixels and mean channel error 0.0242079.
Maximum channel difference was 132, with 405 pixels exceeding 8. Differences
were within x=1..637/y=156..325, concentrated on world/ground edges. The HUD
capture difference is repaired, but fixed-loop versus rendered-frame camera
convergence remains a leading explanation; it has not yet been corrected or
proven causal. Keep the strict equality signal.

The independently bounded v1 capture-path perception probe also demonstrates
that HUD-inclusive RGB measurably alters the three frozen shared parsers' output:
maximum presence-probability changes 0.21559/0.11982/0.08065 and maximum normalized
centre changes 0.05496/0.03124/0.02143. Pig presence stays near one while centres
shift. This establishes perception drift, not utility failure causality or a
hybrid advantage. Details are saved in
`data/issue-76-shared-history-smoke-v1/capture-path-perception-probe.json`.

Before v3, explicitly set the shared fully-zoomed-out camera size before native
history capture, matching public readiness's intended projection instead of
leaving its asymptotic interpolation to finish after the barrier. Retain the
same timing/RGB checks, two assignments and caps. V1/v2 remain unchanged local
evidence under `.local-artifacts/issue-76-shared-history-smoke-v{1,2}/` and
`/home/sukaih/.cache/novphy-shared-history-v{1,2}/`; they are not pushed archives
or advancement evidence.
