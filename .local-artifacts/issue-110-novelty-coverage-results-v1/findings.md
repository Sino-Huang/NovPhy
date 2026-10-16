# #110 results: contract-v2 coverage of all 45 NovPhy cells

`python -u -m scripts.run_novelty_coverage --validate`; plan frozen 2026-10-15T23:05:36Z. Guards: G1_isolation pass, G2_labels pass, G3_training pass, G4_scoring pass.

**Readiness: 45 / 45 cells supported.**

| cell | status | reasons | flags | eval levels | mixed levels | identifiability (k = 15) |
| --- | --- | --- | --- | --- | --- | --- |
| 0/single_force | supported | - | - | 64 | 26 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 0/multiple_forces | supported | - | rollout_error_only; no_mixed_verdict_evaluation_level | 64 | 0 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 0/rolling | supported | - | - | 64 | 33 | external-force-active: not_identifiable; force-on: not_identifiable |
| 0/falling | supported | - | - | 64 | 37 | external-force-active: identifiable; force-on: identifiable |
| 0/sliding | supported | - | - | 64 | 39 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 1/single_force | supported | - | - | 8 | 5 | not required |
| 1/multiple_forces | supported | - | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | not required |
| 1/rolling | supported | - | - | 8 | 4 | not required |
| 1/falling | supported | - | - | 8 | 3 | not required |
| 1/sliding | supported | - | - | 8 | 8 | not required |
| 2/single_force | supported | - | - | 8 | 1 | external-force-active: readiness_or_precision_insufficient; force-on: readiness_or_precision_insufficient |
| 2/multiple_forces | supported | - | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 2/rolling | supported | - | - | 7 | 4 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 2/falling | supported | - | - | 8 | 8 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 2/sliding | supported | - | - | 8 | 4 | external-force-active: not_identifiable; force-on: not_identifiable |
| 3/single_force | supported | - | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 3/multiple_forces | supported | - | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: not_identifiable; force-on: identifiable |
| 3/rolling | supported | - | - | 8 | 4 | external-force-active: not_identifiable; force-on: label_invalid |
| 3/falling | supported | - | - | 8 | 2 | external-force-active: identifiable; force-on: identifiable |
| 3/sliding | supported | - | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid; force-on: readiness_or_precision_insufficient |
| 4/single_force | supported | - | - | 8 | 8 | external-force-active: label_invalid |
| 4/multiple_forces | supported | - | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid |
| 4/rolling | supported | - | - | 8 | 8 | external-force-active: not_identifiable |
| 4/falling | supported | - | - | 8 | 2 | external-force-active: not_identifiable |
| 4/sliding | supported | - | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid |
| 5/single_force | supported | - | - | 8 | 2 | not required |
| 5/multiple_forces | supported | - | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | not required |
| 5/rolling | supported | - | - | 8 | 5 | not required |
| 5/falling | supported | - | - | 8 | 6 | not required |
| 5/sliding | supported | - | - | 8 | 4 | not required |
| 6/single_force | supported | - | - | 8 | 7 | gravity-inverted: identifiable |
| 6/multiple_forces | supported | - | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | gravity-inverted: identifiable |
| 6/rolling | supported | - | - | 8 | 3 | gravity-inverted: identifiable |
| 6/falling | supported | - | - | 8 | 6 | gravity-inverted: identifiable |
| 6/sliding | supported | - | - | 8 | 6 | gravity-inverted: readiness_or_precision_insufficient |
| 7/single_force | supported | - | - | 8 | 6 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 7/multiple_forces | supported | - | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: readiness_or_precision_insufficient; force-on: identifiable |
| 7/rolling | supported | - | - | 8 | 5 | external-force-active: not_identifiable; force-on: not_identifiable |
| 7/falling | supported | - | - | 8 | 8 | external-force-active: readiness_or_precision_insufficient; force-on: identifiable |
| 7/sliding | supported | - | - | 8 | 3 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 8/single_force | supported | - | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: not_identifiable; storm-active: identifiable |
| 8/multiple_forces | supported | - | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid; storm-active: readiness_or_precision_insufficient |
| 8/rolling | supported | - | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: readiness_or_precision_insufficient; storm-active: readiness_or_precision_insufficient |
| 8/falling | supported | - | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: readiness_or_precision_insufficient; storm-active: identifiable |
| 8/sliding | supported | - | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid; storm-active: identifiable |

## Rollout error at 225 frames (masked carrier MSE, level-clustered bootstrap)

Zero-shot (normal training) and few-shot (level adaptation) are separate conditions; never pooled.

| cell | condition | F-1-continuous | F-1-micro | F-1-macro | F-5-continuous | F-5-micro | F-5-macro | F-15-continuous | F-15-micro | F-15-macro | persistence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0/single_force | zero-shot | 0.1089 | 0.1109 | 0.1463 | 0.0377 | 0.0379 | 0.0427 | 0.0231 | 0.0231 | 0.0236 | 0.0929 |
| 0/multiple_forces | zero-shot | 0.1945 | 0.1981 | 0.2289 | 0.0938 | 0.0949 | 0.0963 | 0.0765 | 0.0769 | 0.0763 | 0.1296 |
| 0/rolling | zero-shot | 0.1833 | 0.1856 | 0.2176 | 0.0605 | 0.0585 | 0.0618 | 0.0411 | 0.0401 | 0.0411 | 0.1040 |
| 0/falling | zero-shot | 0.1941 | 0.1973 | 0.2285 | 0.0593 | 0.0569 | 0.0596 | 0.0384 | 0.0372 | 0.0381 | 0.1020 |
| 0/sliding | zero-shot | 0.1624 | 0.1657 | 0.1975 | 0.0480 | 0.0466 | 0.0500 | 0.0293 | 0.0288 | 0.0293 | 0.0964 |
| 1/single_force | zero-shot | 0.0923 | 0.0929 | 0.1335 | 0.0423 | 0.0429 | 0.0508 | 0.0340 | 0.0346 | 0.0355 | 0.1102 |
| 1/single_force | few-shot-1 | 0.1575 | 0.1538 | 0.1229 | 0.0758 | 0.0760 | 0.0725 | 0.0726 | 0.0731 | 0.0701 | 0.1320 |
| 1/multiple_forces | zero-shot | 0.2030 | 0.2079 | 0.2393 | 0.1083 | 0.1110 | 0.1150 | 0.0924 | 0.0936 | 0.0935 | 0.1447 |
| 1/multiple_forces | few-shot-1 | 0.1510 | 0.1479 | 0.1339 | 0.1095 | 0.1089 | 0.1098 | 0.1112 | 0.1115 | 0.1111 | 0.1603 |
| 1/rolling | zero-shot | 0.2085 | 0.2121 | 0.2497 | 0.0762 | 0.0766 | 0.0758 | 0.0540 | 0.0539 | 0.0538 | 0.1174 |
| 1/rolling | few-shot-1 | 0.1236 | 0.1217 | 0.1169 | 0.0616 | 0.0623 | 0.0656 | 0.0585 | 0.0595 | 0.0601 | 0.1189 |
| 1/falling | zero-shot | 0.2219 | 0.2267 | 0.2568 | 0.0931 | 0.0938 | 0.0919 | 0.0695 | 0.0695 | 0.0689 | 0.1264 |
| 1/falling | few-shot-1 | 0.1405 | 0.1405 | 0.1272 | 0.0787 | 0.0799 | 0.0802 | 0.0760 | 0.0773 | 0.0765 | 0.1323 |
| 1/sliding | zero-shot | 0.1489 | 0.1500 | 0.1770 | 0.0793 | 0.0797 | 0.0815 | 0.0602 | 0.0601 | 0.0600 | 0.1177 |
| 1/sliding | few-shot-1 | 0.1324 | 0.1322 | 0.1159 | 0.0662 | 0.0671 | 0.0669 | 0.0595 | 0.0603 | 0.0587 | 0.1205 |
| 2/single_force | zero-shot | 0.1402 | 0.1420 | 0.1756 | 0.0704 | 0.0712 | 0.0735 | 0.0504 | 0.0505 | 0.0501 | 0.1100 |
| 2/single_force | few-shot-2 | 0.0586 | 0.0614 | 0.1580 | 0.0260 | 0.0266 | 0.0327 | 0.0258 | 0.0260 | 0.0274 | 0.0991 |
| 2/multiple_forces | zero-shot | 0.2251 | 0.2299 | 0.2610 | 0.1167 | 0.1180 | 0.1171 | 0.0961 | 0.0962 | 0.0951 | 0.1427 |
| 2/multiple_forces | few-shot-2 | 0.0804 | 0.0832 | 0.1512 | 0.0482 | 0.0483 | 0.0528 | 0.0482 | 0.0482 | 0.0488 | 0.1187 |
| 2/rolling | zero-shot | 0.2204 | 0.2250 | 0.2476 | 0.0926 | 0.0904 | 0.0923 | 0.0699 | 0.0689 | 0.0692 | 0.1226 |
| 2/rolling | few-shot-2 | 0.1133 | 0.1209 | 0.2543 | 0.0430 | 0.0432 | 0.0519 | 0.0387 | 0.0399 | 0.0434 | 0.1008 |
| 2/falling | zero-shot | 0.1998 | 0.2037 | 0.2332 | 0.0673 | 0.0662 | 0.0660 | 0.0432 | 0.0422 | 0.0426 | 0.1043 |
| 2/falling | few-shot-2 | 0.0675 | 0.0692 | 0.1514 | 0.0176 | 0.0173 | 0.0193 | 0.0121 | 0.0120 | 0.0124 | 0.0817 |
| 2/sliding | zero-shot | 0.2003 | 0.2029 | 0.2309 | 0.0696 | 0.0678 | 0.0684 | 0.0463 | 0.0455 | 0.0456 | 0.1068 |
| 2/sliding | few-shot-2 | 0.0864 | 0.0905 | 0.1949 | 0.0327 | 0.0344 | 0.0400 | 0.0303 | 0.0332 | 0.0361 | 0.0972 |
| 3/single_force | zero-shot | 0.1254 | 0.1264 | 0.1642 | 0.0378 | 0.0363 | 0.0393 | 0.0133 | 0.0124 | 0.0131 | 0.0783 |
| 3/single_force | few-shot-3 | 0.1143 | 0.1218 | 0.1762 | 0.0322 | 0.0290 | 0.0651 | 0.0146 | 0.0143 | 0.0298 | 0.0810 |
| 3/multiple_forces | zero-shot | 0.2168 | 0.2208 | 0.2454 | 0.1086 | 0.1083 | 0.1092 | 0.0822 | 0.0819 | 0.0818 | 0.1278 |
| 3/multiple_forces | few-shot-3 | 0.1560 | 0.1614 | 0.1962 | 0.0889 | 0.0863 | 0.1007 | 0.0783 | 0.0774 | 0.0806 | 0.1302 |
| 3/rolling | zero-shot | 0.1977 | 0.2011 | 0.2277 | 0.0665 | 0.0615 | 0.0660 | 0.0428 | 0.0409 | 0.0422 | 0.1035 |
| 3/rolling | few-shot-3 | 0.1271 | 0.1265 | 0.1631 | 0.0460 | 0.0441 | 0.0643 | 0.0362 | 0.0360 | 0.0441 | 0.0999 |
| 3/falling | zero-shot | 0.2020 | 0.2033 | 0.2319 | 0.0751 | 0.0697 | 0.0745 | 0.0525 | 0.0500 | 0.0521 | 0.1127 |
| 3/falling | few-shot-3 | 0.1234 | 0.1231 | 0.1600 | 0.0426 | 0.0411 | 0.0631 | 0.0303 | 0.0303 | 0.0398 | 0.0912 |
| 3/sliding | zero-shot | 0.1937 | 0.1991 | 0.2330 | 0.0475 | 0.0445 | 0.0477 | 0.0201 | 0.0192 | 0.0202 | 0.0862 |
| 3/sliding | few-shot-3 | 0.1150 | 0.1182 | 0.1617 | 0.0406 | 0.0389 | 0.0523 | 0.0271 | 0.0272 | 0.0288 | 0.0913 |
| 4/single_force | zero-shot | 0.1508 | 0.1518 | 0.1795 | 0.0669 | 0.0672 | 0.0684 | 0.0464 | 0.0460 | 0.0463 | 0.1081 |
| 4/single_force | few-shot-4 | 0.0684 | 0.0699 | 0.1136 | 0.0433 | 0.0433 | 0.0505 | 0.0413 | 0.0408 | 0.0416 | 0.1049 |
| 4/multiple_forces | zero-shot | 0.2343 | 0.2392 | 0.2685 | 0.1185 | 0.1201 | 0.1175 | 0.0958 | 0.0961 | 0.0947 | 0.1450 |
| 4/multiple_forces | few-shot-4 | 0.0957 | 0.0971 | 0.1365 | 0.0668 | 0.0678 | 0.0713 | 0.0646 | 0.0652 | 0.0646 | 0.1245 |
| 4/rolling | zero-shot | 0.2067 | 0.2099 | 0.2347 | 0.0781 | 0.0750 | 0.0782 | 0.0532 | 0.0516 | 0.0531 | 0.1089 |
| 4/rolling | few-shot-4 | 0.0811 | 0.0797 | 0.0938 | 0.0403 | 0.0404 | 0.0392 | 0.0359 | 0.0359 | 0.0329 | 0.1005 |
| 4/falling | zero-shot | 0.2082 | 0.2103 | 0.2367 | 0.0807 | 0.0775 | 0.0807 | 0.0559 | 0.0545 | 0.0557 | 0.1131 |
| 4/falling | few-shot-4 | 0.0724 | 0.0696 | 0.0844 | 0.0273 | 0.0269 | 0.0291 | 0.0198 | 0.0199 | 0.0189 | 0.0836 |
| 4/sliding | zero-shot | 0.2038 | 0.2104 | 0.2382 | 0.0597 | 0.0575 | 0.0594 | 0.0332 | 0.0324 | 0.0329 | 0.0957 |
| 4/sliding | few-shot-4 | 0.0637 | 0.0626 | 0.0802 | 0.0223 | 0.0227 | 0.0227 | 0.0151 | 0.0156 | 0.0129 | 0.0788 |
| 5/single_force | zero-shot | 0.0900 | 0.0963 | 0.1291 | 0.0318 | 0.0320 | 0.0363 | 0.0264 | 0.0266 | 0.0271 | 0.0975 |
| 5/single_force | few-shot-5 | 0.1382 | 0.1320 | 0.1975 | 0.0608 | 0.0627 | 0.0704 | 0.0588 | 0.0608 | 0.0594 | 0.1251 |
| 5/multiple_forces | zero-shot | 0.1993 | 0.2027 | 0.2310 | 0.0968 | 0.0971 | 0.0983 | 0.0817 | 0.0818 | 0.0813 | 0.1318 |
| 5/multiple_forces | few-shot-5 | 0.1789 | 0.1761 | 0.2394 | 0.1079 | 0.1084 | 0.1190 | 0.0923 | 0.0913 | 0.0924 | 0.1367 |
| 5/rolling | zero-shot | 0.1876 | 0.1939 | 0.2287 | 0.0506 | 0.0509 | 0.0527 | 0.0315 | 0.0314 | 0.0314 | 0.0982 |
| 5/rolling | few-shot-5 | 0.1385 | 0.1322 | 0.1962 | 0.0538 | 0.0511 | 0.0685 | 0.0357 | 0.0325 | 0.0382 | 0.0925 |
| 5/falling | zero-shot | 0.1955 | 0.2004 | 0.2302 | 0.0537 | 0.0531 | 0.0525 | 0.0334 | 0.0328 | 0.0324 | 0.0959 |
| 5/falling | few-shot-5 | 0.1463 | 0.1400 | 0.2028 | 0.0590 | 0.0559 | 0.0745 | 0.0376 | 0.0352 | 0.0399 | 0.0917 |
| 5/sliding | zero-shot | 0.1231 | 0.1284 | 0.1496 | 0.0559 | 0.0564 | 0.0615 | 0.0492 | 0.0494 | 0.0502 | 0.1177 |
| 5/sliding | few-shot-5 | 0.1367 | 0.1326 | 0.1891 | 0.0662 | 0.0668 | 0.0766 | 0.0557 | 0.0564 | 0.0580 | 0.1142 |
| 6/single_force | zero-shot | 0.1319 | 0.1344 | 0.1683 | 0.0659 | 0.0669 | 0.0709 | 0.0552 | 0.0555 | 0.0560 | 0.1270 |
| 6/single_force | few-shot-6 | 0.0412 | 0.0413 | 0.0572 | 0.0285 | 0.0279 | 0.0270 | 0.0291 | 0.0281 | 0.0224 | 0.0943 |
| 6/multiple_forces | zero-shot | 0.2335 | 0.2390 | 0.2712 | 0.1261 | 0.1282 | 0.1284 | 0.1120 | 0.1131 | 0.1113 | 0.1653 |
| 6/multiple_forces | few-shot-6 | 0.0837 | 0.0845 | 0.0935 | 0.0644 | 0.0652 | 0.0614 | 0.0630 | 0.0638 | 0.0565 | 0.1156 |
| 6/rolling | zero-shot | 0.2334 | 0.2348 | 0.2576 | 0.1344 | 0.1349 | 0.1354 | 0.1207 | 0.1213 | 0.1202 | 0.1707 |
| 6/rolling | few-shot-6 | 0.0908 | 0.0903 | 0.0982 | 0.0637 | 0.0638 | 0.0609 | 0.0614 | 0.0615 | 0.0571 | 0.1241 |
| 6/falling | zero-shot | 0.2271 | 0.2331 | 0.2601 | 0.1066 | 0.1080 | 0.1070 | 0.0919 | 0.0925 | 0.0910 | 0.1524 |
| 6/falling | few-shot-6 | 0.0577 | 0.0572 | 0.0689 | 0.0319 | 0.0318 | 0.0314 | 0.0331 | 0.0334 | 0.0272 | 0.0965 |
| 6/sliding | zero-shot | 0.2317 | 0.2355 | 0.2570 | 0.1210 | 0.1218 | 0.1211 | 0.1067 | 0.1073 | 0.1059 | 0.1632 |
| 6/sliding | few-shot-6 | 0.0658 | 0.0650 | 0.0774 | 0.0329 | 0.0329 | 0.0325 | 0.0294 | 0.0308 | 0.0246 | 0.0943 |
| 7/single_force | zero-shot | 0.1197 | 0.1224 | 0.1509 | 0.0382 | 0.0376 | 0.0396 | 0.0146 | 0.0139 | 0.0139 | 0.0787 |
| 7/single_force | few-shot-7 | 2.0991 | 1.5929 | 1.2169 | 0.1080 | 0.1043 | 0.0948 | 0.0482 | 0.0516 | 0.0568 | 0.1004 |
| 7/multiple_forces | zero-shot | 0.1959 | 0.1999 | 0.2278 | 0.0839 | 0.0845 | 0.0853 | 0.0569 | 0.0569 | 0.0562 | 0.1109 |
| 7/multiple_forces | few-shot-7 | 2.0560 | 1.6277 | 1.1494 | 0.1682 | 0.1611 | 0.1420 | 0.0977 | 0.1050 | 0.1071 | 0.1323 |
| 7/rolling | zero-shot | 0.1980 | 0.2011 | 0.2286 | 0.0661 | 0.0631 | 0.0654 | 0.0405 | 0.0391 | 0.0397 | 0.1018 |
| 7/rolling | few-shot-7 | 2.3957 | 1.8236 | 1.3467 | 0.1298 | 0.1190 | 0.1001 | 0.0432 | 0.0494 | 0.0447 | 0.0914 |
| 7/falling | zero-shot | 0.1940 | 0.1973 | 0.2242 | 0.0604 | 0.0551 | 0.0599 | 0.0360 | 0.0341 | 0.0353 | 0.0974 |
| 7/falling | few-shot-7 | 2.3346 | 1.7889 | 1.3130 | 0.1437 | 0.1296 | 0.1118 | 0.0680 | 0.0614 | 0.0712 | 0.0958 |
| 7/sliding | zero-shot | 0.1989 | 0.2058 | 0.2337 | 0.0483 | 0.0451 | 0.0479 | 0.0208 | 0.0197 | 0.0207 | 0.0865 |
| 7/sliding | few-shot-7 | 2.5075 | 1.8986 | 1.3458 | 0.1342 | 0.1219 | 0.1020 | 0.0457 | 0.0513 | 0.0468 | 0.0923 |
| 8/single_force | zero-shot | 0.1362 | 0.1410 | 0.1693 | 0.0693 | 0.0710 | 0.0744 | 0.0564 | 0.0568 | 0.0566 | 0.1131 |
| 8/single_force | few-shot-8 | 0.0421 | 0.0431 | 0.0444 | 0.0305 | 0.0308 | 0.0315 | 0.0319 | 0.0321 | 0.0323 | 0.1052 |
| 8/multiple_forces | zero-shot | 0.1747 | 0.1796 | 0.2015 | 0.1221 | 0.1233 | 0.1279 | 0.1136 | 0.1139 | 0.1145 | 0.1641 |
| 8/multiple_forces | few-shot-8 | 0.0766 | 0.0782 | 0.0786 | 0.0524 | 0.0527 | 0.0520 | 0.0515 | 0.0516 | 0.0511 | 0.1229 |
| 8/rolling | zero-shot | 0.1186 | 0.1185 | 0.1499 | 0.0580 | 0.0580 | 0.0599 | 0.0465 | 0.0466 | 0.0464 | 0.1078 |
| 8/rolling | few-shot-8 | 0.0369 | 0.0377 | 0.0404 | 0.0244 | 0.0244 | 0.0258 | 0.0257 | 0.0256 | 0.0257 | 0.1026 |
| 8/falling | zero-shot | 0.1862 | 0.1912 | 0.2278 | 0.0574 | 0.0573 | 0.0584 | 0.0381 | 0.0379 | 0.0377 | 0.1003 |
| 8/falling | few-shot-8 | 0.0515 | 0.0512 | 0.0549 | 0.0268 | 0.0266 | 0.0282 | 0.0229 | 0.0228 | 0.0227 | 0.0942 |
| 8/sliding | zero-shot | 0.1190 | 0.1183 | 0.1510 | 0.0640 | 0.0645 | 0.0673 | 0.0548 | 0.0550 | 0.0553 | 0.1206 |
| 8/sliding | few-shot-8 | 0.0414 | 0.0425 | 0.0449 | 0.0357 | 0.0362 | 0.0374 | 0.0364 | 0.0366 | 0.0363 | 0.1076 |

Full tables (15 / 60 / 225 frames, intervals, position MSE, novel - normal contrasts, C-F-* rows): `rollout-error.csv` and `compute.json`.

## Static fit (80 templates)

v2: 80 / 80 templates fit on every captured level; v1: 19 / 80.

## Fan labeler repair

- fit/type010201: legacy disagree (median -1.37e-16), repaired agree (median 100.0, precision 0.995, recall 0.974)
- fit/type010202: legacy disagree (median 0.0), repaired agree (median 100.0, precision 0.996, recall 0.976)
- fit/type010203: legacy disagree (median -100.0), repaired agree (median 100.0, precision 0.996, recall 0.954)
- fit/type010204: legacy disagree (median 0.0), repaired agree (median 100.0, precision 0.995, recall 0.863)
- fit/type010205: legacy disagree (median 0.0), repaired agree (median 100.0, precision 0.989, recall 0.903)
- smoke/type010201: legacy disagree (median -1.92e-17), repaired not_exercised (median n/a, precision n/a, recall n/a)
- smoke/type010202: legacy disagree (median -50.0), repaired agree (median 100.0, precision 1.000, recall 0.990)
- smoke/type010203: legacy disagree (median 0.0), repaired agree (median 100.0, precision 1.000, recall 0.987)
- smoke/type010204: legacy disagree (median -100.0), repaired agree (median 100.0, precision 1.000, recall 0.982)
- smoke/type010205: legacy not_exercised (median n/a), repaired not_exercised (median n/a, precision n/a, recall n/a)

## Members

| condition | family | seed | usable | skips | excursion | GPU s |
| --- | --- | --- | --- | --- | --- | --- |
| zero-shot | hybrid | 20260908 | True | 0 | False | 1260 |
| zero-shot | hybrid | 20260909 | True | 0 | False | 1279 |
| zero-shot | hybrid | 20260910 | True | 0 | False | 1268 |
| zero-shot | continuous | 20260908 | True | 0 | False | 996 |
| zero-shot | continuous | 20260909 | True | 0 | False | 1036 |
| zero-shot | continuous | 20260910 | True | 0 | False | 1031 |
| few-shot-1 | hybrid | 20260908 | True | 0 | False | 317 |
| few-shot-1 | hybrid | 20260909 | True | 0 | False | 323 |
| few-shot-1 | hybrid | 20260910 | True | 0 | False | 318 |
| few-shot-1 | continuous | 20260908 | True | 0 | False | 245 |
| few-shot-1 | continuous | 20260909 | True | 0 | False | 264 |
| few-shot-1 | continuous | 20260910 | True | 0 | False | 268 |
| few-shot-2 | hybrid | 20260908 | True | 0 | False | 286 |
| few-shot-2 | hybrid | 20260909 | True | 0 | False | 468 |
| few-shot-2 | hybrid | 20260910 | True | 0 | False | 284 |
| few-shot-2 | continuous | 20260908 | True | 0 | False | 256 |
| few-shot-2 | continuous | 20260909 | True | 0 | False | 265 |
| few-shot-2 | continuous | 20260910 | True | 0 | False | 239 |
| few-shot-3 | hybrid | 20260908 | True | 0 | False | 284 |
| few-shot-3 | hybrid | 20260909 | True | 0 | False | 290 |
| few-shot-3 | hybrid | 20260910 | True | 0 | False | 287 |
| few-shot-3 | continuous | 20260908 | True | 0 | False | 239 |
| few-shot-3 | continuous | 20260909 | True | 0 | False | 249 |
| few-shot-3 | continuous | 20260910 | True | 0 | False | 254 |
| few-shot-4 | hybrid | 20260908 | True | 0 | False | 286 |
| few-shot-4 | hybrid | 20260909 | True | 0 | False | 314 |
| few-shot-4 | hybrid | 20260910 | True | 0 | False | 282 |
| few-shot-4 | continuous | 20260908 | False | 0 | True | 255 |
| few-shot-4 | continuous | 20260909 | True | 0 | False | 252 |
| few-shot-4 | continuous | 20260910 | True | 0 | False | 238 |
| few-shot-5 | hybrid | 20260908 | True | 0 | False | 284 |
| few-shot-5 | hybrid | 20260909 | True | 0 | False | 290 |
| few-shot-5 | hybrid | 20260910 | True | 0 | False | 285 |
| few-shot-5 | continuous | 20260908 | True | 0 | False | 240 |
| few-shot-5 | continuous | 20260909 | True | 0 | False | 250 |
| few-shot-5 | continuous | 20260910 | True | 0 | False | 255 |
| few-shot-6 | hybrid | 20260908 | True | 0 | False | 286 |
| few-shot-6 | hybrid | 20260909 | True | 0 | False | 313 |
| few-shot-6 | hybrid | 20260910 | True | 0 | False | 284 |
| few-shot-6 | continuous | 20260908 | True | 0 | False | 255 |
| few-shot-6 | continuous | 20260909 | True | 0 | False | 251 |
| few-shot-6 | continuous | 20260910 | True | 0 | False | 238 |
| few-shot-7 | hybrid | 20260908 | True | 0 | False | 284 |
| few-shot-7 | hybrid | 20260909 | True | 0 | False | 290 |
| few-shot-7 | hybrid | 20260910 | True | 0 | False | 286 |
| few-shot-7 | continuous | 20260908 | True | 0 | False | 240 |
| few-shot-7 | continuous | 20260909 | True | 0 | False | 250 |
| few-shot-7 | continuous | 20260910 | True | 0 | False | 255 |
| few-shot-8 | hybrid | 20260908 | True | 0 | False | 286 |
| few-shot-8 | hybrid | 20260909 | True | 0 | False | 313 |
| few-shot-8 | hybrid | 20260910 | True | 0 | False | 284 |
| few-shot-8 | continuous | 20260908 | True | 0 | False | 256 |
| few-shot-8 | continuous | 20260909 | True | 0 | False | 250 |
| few-shot-8 | continuous | 20260910 | True | 0 | False | 238 |

## Encoder training (`encoder-training.json`, read from the checkpoints)

- zero-shot: 12 epochs, mean loss 10.49831 (epoch 1) → 7.64830 (epoch 12).
- few-shot-1: 2000 updates from zero-shot, loss (mean of 100 updates) 14.67307 → 6.08210.
- few-shot-2: 2000 updates from zero-shot, loss (mean of 100 updates) 12.50681 → 5.98685.
- few-shot-3: 2000 updates from zero-shot, loss (mean of 100 updates) 9.73934 → 4.15984.
- few-shot-4: 2000 updates from zero-shot, loss (mean of 100 updates) 14.02166 → 8.06459.
- few-shot-5: 2000 updates from zero-shot, loss (mean of 100 updates) 8.29411 → 5.03005.
- few-shot-6: 2000 updates from zero-shot, loss (mean of 100 updates) 10.93934 → 5.45027.
- few-shot-7: 2000 updates from zero-shot, loss (mean of 100 updates) 14.51560 → 7.33351.
- few-shot-8: 2000 updates from zero-shot, loss (mean of 100 updates) 6.89766 → 4.25960.

## Scoring-policy coverage (phase-A review)

G4 checks the scoring policy of the 1278 rollout records; 0 of the 60 identifiability probe records carry one. Note: the identifiability probes (MLP probes trained and scored in-process by --probe) ran without deterministic_scoring() and their records carry no scoring policy; the frozen G4 check iterates the rollout records only, so the plan's G4 text ('every record under deterministic_scoring') does not hold for the probe records. The probe models were not retained, so the verdicts cannot be re-scored without retraining, which a post-freeze change may not do; they are reported as recorded. A fresh process importing this runner runs with matmul TF32 off (as SCORING_POLICY) and cuDNN deterministic off / cuDNN TF32 on; the probes use no cuDNN operation. Borderline verdicts are not guaranteed to reproduce. G4 and every token are unchanged.

Compute: 35444 GPU s, 37337 process wall s, 0 engine s.
