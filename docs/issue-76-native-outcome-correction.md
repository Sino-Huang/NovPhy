# #76 native outcome reporting correction

An audit during automated live-policy integration found that six development
segments have genuine native `level_clear` terminals, while every episode's
later interface query returned `EVALUATION_TERMINATED`. The original collector
requires that later query to equal `WON`, so it reported zero gameplay successes.

The six affected assignments are 002, 028, 032, 046, 300 and 335. For each, the
native manifest identifies a real `level_clear` terminal, its exact engine event
is present, and the final recorded pig is destroyed. None of these six segments
is censored. Thus the discrepancy is not explained by the rule retaining
timeouts as failures. The evidence proves loss of a completed win in the
collector's post-capture outcome projection; it does not require changing the
physics, replaying a level, guessing success from pig counts, or treating an
arbitrary file ending as a terminal.

`scripts/issue_76_native_outcomes.py` publishes a separate source-bound audit
over all 350 assigned episodes and validates terminal-to-event bindings. It
does not replace any raw result, collection report, prepared tensor, cost, or
earlier scientific disposition. A stable stop, pig-removal event alone,
censored segment, missing terminal or technical failure is still not success.
The native-terminal tally is 6/350 (1.71%), not the original recorded 0/350.
This is fixed-action development evidence, not learned-policy performance or
an advancement result. The original false values remain visible alongside the
derived outcomes so the correction cannot be mistaken for the initial report.

Future live evaluation must use the same explicit engine-terminal projection
for every arm, stop its episode when the native `level_clear` terminal is
observed, and keep the interface lifecycle state as separate diagnostic data.
Only RGB, observation timing and executed actions enter learned planning; the
terminal event is an evaluation/stopping fact, never a counterfactual action
selection input. Freeze this rule before fresh access. No advancement margin,
failure rule for censored data, or previous unsupported comparison is changed.
