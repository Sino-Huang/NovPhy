# #123 G-fid: FourRoom Distractors fidelity check

Official code 97054ec; success % on the 50 released tasks; seeds 42/43/44.

## Tier E (released checkpoints): PASS

| model | seed 42 | seed 43 | seed 44 | mean ± SE | paper | band | in band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lewm | 40.0 | 50.0 | 30.0 | 40.0 ± 5.8 | 40.7 ± 6.8 | [27.1, 54.3] | True |
| hjepa_l2 | 86.0 | 72.0 | 92.0 | 83.3 ± 5.9 | 81.3 ± 6.7 | [67.9, 94.7] | True |
| hjepa_l3 | 100.0 | 92.0 | 96.0 | 96.0 ± 2.3 | 96.0 ± 1.1 | [92.8, 99.2] | True |

H-JEPA-3 − LeWM per seed {'42': 60.0, '43': 42.0, '44': 66.0}, mean 56.0 pp; ordering True, every seed ≥ 30: True

## Tier T (retrained with main_hjepa.py): PASS

| model | seed 42 | seed 43 | seed 44 | mean ± SE | paper | band | in band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lewm | 32.0 | 20.0 | 58.0 | 36.7 ± 11.2 | 40.7 ± 6.8 | [14.5, 66.9] | True |
| hjepa_l2 | 84.0 | 88.0 | 74.0 | 82.0 ± 4.2 | 81.3 ± 6.7 | [65.5, 97.1] | True |
| hjepa_l3 | 92.0 | 96.0 | 98.0 | 95.3 ± 1.8 | 96.0 ± 1.1 | [91.8, 100.2] | True |

H-JEPA-3 − LeWM per seed {'42': 60.0, '43': 76.0, '44': 40.00000000000001}, mean 58.7 pp; mean ≥ 30 with every seed positive: True

**G-fid: PASS**
