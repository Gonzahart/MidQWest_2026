# Fresh seed-32 test of the frozen routers (Sean, PCHome, 2026-11-22)

Data: relay-BP-1 and relay-BP-5 on the same 10,000 pairs per rate, seed 32, `sweep.py --workers 20`, run 10:08-10:37 CST.
Both tables were committed before this data existed (frozen table 1463749 at 04:29, GPU-priced table a4414c7 at 05:33). Nothing was refit.
Times come from a 20-worker run: compare them within a row only. They are not final timing.

## p = 0.005

| Policy | Escalated | Failed pairs | Mean ms | p99 ms |
|---|---:|---:|---:|---:|
| relay-BP-1 | 0% | 970 | 716 | 7,688 |
| relay-BP-5 | 100% | 808 | 1,792 | 10,350 |
| Frozen table | 10.8% | 863 | 1,303 | 14,950 |
| Simple rule, same escalations | 10.8% | 876 | 1,767 | 15,538 |
| Coin flip at equal mean time | 54.6% | 881.6 (95% range 859-904) | 1,303 | - |
| Candidate (price 0.04 + likelihood pick) | - | 859 | 1,171 | 14,895 |
| GPU-priced table (CPU replay) | 60.1% | 801 | 1,881 | 15,305 |

Paired, table only fails / other only fails: frozen vs relay-BP-1 39/146 (p = 1e-15); frozen vs simple rule 115/128 (p = 0.44); candidate vs frozen 36 fixed / 32 broken (p = 0.72); GPU-priced vs relay-BP-5 92/99 (p = 0.66).

## p = 0.004

| Policy | Escalated | Failed pairs | Mean ms | p99 ms |
|---|---:|---:|---:|---:|
| relay-BP-1 | 0% | 141 | 113 | 2,034 |
| relay-BP-5 | 100% | 104 | 393 | 6,538 |
| Frozen table | 4.1% | 119 | 183 | 3,011 |
| Simple rule, same escalations | 4.1% | 115 | 301 | 7,973 |
| Coin flip at equal mean time | 25.0% | 131.8 (95% range 123-140) | 183 | - |
| Candidate | - | 118 | 181 | 2,937 |
| GPU-priced table (CPU replay) | 40.1% | 106 | 336 | 4,860 |

Paired: frozen vs relay-BP-1 4/26 (p = 6e-5); frozen vs simple rule 37/33 (p = 0.72); GPU-priced vs relay-BP-5 21/19 (p = 0.88).

## Verdict

- Confirmed: the frozen table beats relay-BP-1 by a wide margin and beats a coin flip at equal time (clearly at p = 0.004, at the edge of the range at p = 0.005), and it uses less time than the simple rule for the same number of escalations.
- Not confirmed: the table is not more accurate than the simple rule at equal escalations.
- Not confirmed: the GPU-priced table beating relay-BP-5. It ties (801 vs 808, 106 vs 104). Pooled with seed 10: 185/220, p = 0.09. Claim "matches relay-BP-5 accuracy while sending 40-60% of shots to the strong pass", not "beats it".
- Candidate: about 10% less mean time at p = 0.005, with accuracy unchanged within noise.

Reproduce: `python truth_table_router.py --table "results/07_truth_table/table/<table>.json" results/06_router/fresh_s32/relay1_...npz results/06_router/fresh_s32/relay5_...npz`; candidate: `python results/06_router/fresh_s32/score_candidate_s32.py` from the kit folder.
