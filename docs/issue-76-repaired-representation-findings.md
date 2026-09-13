# Repaired shared perception: downstream refit complete, adaptation still not ready

The prospectively frozen `issue-76-repaired-representation-v1` run completed
all three shared history fits, six independent dynamics fits, six controllers,
and 600 assigned diagnostic records on September 13. It reused the three
balanced parser checkpoints without further optimization. This tests the
combined mixed-batch/balanced-perception intervention, not balancing alone.

Publication: `data/issue-76-repaired-representation/report.json`; the adjacent
`diagnostic-report.json.gz` preserves the complete 99,316,242-byte diagnostic
report, including per-transition work records. With `novphy` activated and
`env.sh` sourced, independent validation completed successfully:

```bash
python -m scripts.run_issue_76_repaired_representation --validate
```

All six arm/seed cells retain 50 assigned calibration and 50 assigned
model-selection members, with 45 available endpoints in each role. The 90%
endpoint-coverage checks pass, and no prediction exceptions were recorded.
Neither fact establishes prediction accuracy, gameplay competence, or readiness.
Unavailable assignments remain unavailable and are retained.

## Within-refit comparison

Mean recursive carrier MSE on the 45 available model-selection members per
seed, at the common 11,250-native-step / 4.5-second endpoint:

| Seed | Adaptive hybrid | Same hybrid, fixed 750 continuous | Same hybrid, fixed 750 micro | Independent pure, adaptive / fixed 750 continuous |
| --- | ---: | ---: | ---: | ---: |
| 760930001 | .083199 | .083199 | .059527 | .022779 |
| 760930002 | .199937 | .062391 | .047178 | .025261 |
| 760930003 | .081637 | .081637 | .053378 | .027986 |

Calibration selects fixed-750-micro as the lowest-carrier-MSE fixed hybrid
policy for every seed, and fixed-750-continuous for every independent pure
model. These are diagnostic selections, not established strongest gameplay
policies. Fixed micro improves carrier error within each hybrid checkpoint,
but remains worse than independent pure on that metric. Adaptive hybrid is
identical to its fixed continuous control for seeds 1/3 and worse for seed 2.

All available adaptive hybrid model-selection trajectories execute continuous
mode only. Seeds 1/3 each execute 675 transitions at horizon 750. Seed 2 executes
1,095 at horizon 50 and 602 at horizon 750. All pure adaptive cells execute
675 horizon-750 transitions. Horizon variation in seed 2 does not establish
useful adaptation, and there is no deployed symbolic-mode use in these rows.

## Prespecified physical comparison against the original refit

Carrier targets changed, so old-versus-new carrier MSE is not a physical-effect
estimate. The frozen paired engine-relative contrasts are retained individually
in `paired_physical_contrasts`, including failures and unavailable members.
Below are mean repaired-minus-original differences over the same 45 available
model-selection members; negative means lower error. Each pair of columns is
adaptive / fixed-750-continuous, with no selection among seeds or metrics.

| Seed / arm | Block-count MAE change | Pig-count MAE change | Presence MSE change |
| --- | ---: | ---: | ---: |
| 1 hybrid | -.599250 / -.554013 | -.694297 / -.047094 | -4.300143 / -.006759 |
| 1 pure | -.442982 / -.591567 | +.021914 / +.021914 | -207.628086 / -.056109 |
| 2 hybrid | -.913072 / -.989107 | -.012755 / -.013025 | +.036203 / -.073173 |
| 2 pure | -1.172249 / -.704439 | -.668643 / +.081344 | -2.116502 / -.051257 |
| 3 hybrid | -.813765 / -.825638 | -.016672 / -.016672 | -1.400023 / -.023020 |
| 3 pure | -.750544 / -.744795 | -.971196 / -.041726 | -881.855359 / -.046702 |

Block-count errors improve in every listed contrast. Changes are not uniformly
beneficial: seed-2 hybrid adaptive presence error increases, as do some pure
pig-count errors. Large adaptive presence improvements also reflect avoidance
of the original unstable short-horizon policies; they do not isolate perception
from the downstream refit or prove general physical validity.

Current adaptive hybrid block-count MAEs are .962833/.891075/.764013;
independent pure values are .979241/.872196/.827234. Adaptive hybrid presence
MSEs are .121390/.172680/.102465 versus pure .059688/.062808/.067153. Adaptive
hybrid pig-count MAEs are .076773/.009467/.005550 versus pure
.044137/.116591/.006582. These fieldwise differences do not establish action
ranking, winning, or a seed-robust hybrid advantage.

## Post-publication controller diagnosis, no optimization

A CPU audit loaded every saved training-only teacher shard and evaluated each
completed controller on its exact stored inputs, action, and remaining time.
There are 62,935 teacher rows per arm/seed; frames/segments are not independent
scenario replicates. Hybrid teacher symbolic-label counts are 0, 1, and 13:
seed 2 has one macro-750 label; seed 3 has ten macro-250 and three macro-50
labels. No seed has micro teacher labels, despite fixed micro's favorable
recursive diagnostic contrast. This distinguishes local teacher utility from
recursive fixed-policy accuracy; it does not identify the cause by itself.

| Seed | Hybrid exact teacher-label accuracy | Pure exact teacher-label accuracy |
| --- | ---: | ---: |
| 760930001 | .521014 | .542909 |
| 760930002 | .540494 | .547946 |
| 760930003 | .567077 | .585398 |

Hybrid training-input predictions contain 0/0/3 symbolic choices; all other
predictions are continuous. The three seed-3 symbolic predictions do not
transfer to any scored calibration/model-selection adaptive trajectory.
This is neither a meaningful symbolic-use result nor successful distillation.

A deterministic necessary-readiness check on the validated publication fails:

```python
import json
from pathlib import Path

report = json.loads(Path("data/issue-76-repaired-representation/report.json").read_text())
for seed in report["plan"]["seeds"]:
    rows = [r for r in report["rows"] if r["seed"] == seed and not r["pure"]
            and r["role"] == "model_selection" and r["available"]]
    symbolic = sum(n for r in rows for mode, n in
                   r["policies"]["adaptive"]["mode_counts"].items() if mode != "continuous")
    assert symbolic > 0, "adaptive hybrid executes zero symbolic transitions"
```

It fails at seed 760930001 with zero transitions; direct counts are zero for
all three seeds. This is only a necessary diagnostic check, not the full gate:
arbitrary mode variation would not make the candidate ready.

## Costs, limits, and next step

New recorded active work totals 3,561.025 seconds, counting each shared common
fit once, metadata preparation once, and all six predictor/controller/diagnostic
jobs. Shared common costs are 264.103/266.676/266.960 seconds; metadata
preparation is 13.360 seconds. Reused parser fitting adds 309.981 seconds, and
original preprocessing retains its separately recorded 5,861.401 active seconds
and 12,125.005 parallel child CPU seconds. Prior research/control costs remain
in the source-bound plans; these totals are not full project research cost,
end-to-end FLOPs, or matched deployment compute. All jobs completed within
their frozen ceilings. Each predictor applied 5,952 of 6,000 scheduled updates;
each controller applied 1,984 of 2,000, retaining the declared skipped cases.

The shared perception repair is insufficient for adaptive readiness. Before
any fresh access, distinguish local-versus-recursive teacher mismatch,
cost-versus-accuracy utility, dynamics training quality, and controller fitting
failure with a separately specified development diagnosis. Do not force
symbolic labels, select a favorable seed, or weaken the primary gate. A learned
gameplay test still requires its own frozen paired matrix, strongest eligible
controls/action prior, full compute accounting, and actual engine outcomes.

The legacy collection-success field remains zero in the unchanged parent
kernel. The authoritative separate native-terminal audit remains 6/350
fixed-policy collection wins, not learned-policy wins. No fresh or final data
was opened; no supported advancement disposition was produced; #64/#65 remain
unauthorized. All original negative results and the present failed-readiness
evidence are preserved.
