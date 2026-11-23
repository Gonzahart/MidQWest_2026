# Relay Lab accuracy and latency experiment

This candidate makes the six-bit router more selective and gives it a way to keep a good first correction. For zero-loss circuits at p = 0.004 and 0.005, the price of escalation increases from 0.035 to 0.04 expected failures avoided per second. After escalation, if only the first decoder converged, keep its correction. If both converged, take the correction with higher log likelihood under the kit's independent error priors. Ties and cases where neither converged use the second result, as in the original router. At p = 0.003 the original policy is used.

## Matched replay

| Noise rate | Seed | Pairs | Failed pairs, original → candidate | Mean ms per pair, original → candidate | p99 ms per pair, original → candidate |
|---|---|---:|---:|---:|---:|
| 0.004 | 4 | 10,000 | 117 → 114 | 272.4 → 267.5 | 4,577 → 4,445 |
| 0.004 | 16 | 1,000 | 10 → 10 | 63.8 → 63.8 | 1,001 → 1,001 |
| 0.005 | 10 | 10,000 | 837 → 818 | 1,830.1 → 1,671.2 | 19,323 → 19,312 |
| 0.005 | 16 | 1,000 | 71 → 64 | 561.2 → 487.9 | 6,366 → 6,359 |

The calibration script compares eight prices on seed 2, chooses among prices with lower mean time and no increase in failed pairs, then freezes the choice before evaluating the other seeds. The original six-bit table was already trained. These replay packs had previously been used to study the baseline. Compare times within each row because worker configurations differ. Timing charges both passes whenever a shot escalates and excludes policy evaluation overhead. The strong decoder's state can evolve differently when called only on selected shots.

On seed 10 at p = 0.005, changing the price alone gives 821 failures. Comparing the corrections brings that to 818. Relative to the original router, the complete candidate fixes 36 pairs and introduces 17 failures. Mean time improves more than p99.

## Fresh-shot check

At p = 0.005, X seed 923 and Z seed 924, both policies failed on the same 6 of 128 pairs. The original escalated 27 basis shots and averaged 508.6 ms per pair. The candidate escalated 26 and averaged 478.3 ms. Their p99 values were 6,777.3 and 6,769.8 ms respectively. This small run verifies the live implementation and gives one timing observation. More fresh pairs are needed to establish an accuracy gain.

## Reproduce

Use the team kit checkout at revision `db2b907` and extract the original judge packet. In the kit's Linux or WSL Python environment, set `KIT` to the kit directory and `JUDGE` to the extracted judge directory. From this experiment folder run the following commands.

```bash
python replay_compare.py "$KIT/results" "$JUDGE"
python live_compare.py --kit "$KIT" --judge "$JUDGE" --p 0.005 --seed 923 --shots 128
```

The second command samples fresh shots with the kit, warms up each policy, alternates which one runs first and reports measured latency including routing and correction comparison. It scores a pair as failed if either its X or Z memory experiment fails. Change the seed or increase the shot count for an independent evaluation. The scripts import the frozen six-bit policy from the supplied judge packet.
