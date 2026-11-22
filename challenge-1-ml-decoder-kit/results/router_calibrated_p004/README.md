# Calibrated Relay-BP router at p = 0.004

No loss. Test data is the checked-in BlueQubit cache of 10,000 paired X/Z
shots (seed 2). Thresholds were fixed from a separate 5,000-pair Relay-BP-1
calibration run on an Apple M5 (seed 4). The baseline and replay times came
from the BlueQubit test cache. They exclude inference and are not the final
single-worker measurements.

The failure-label models used the checked-in 20,000-pair BlueQubit seed-1
Relay-BP-1 training run at p = 0.005. The rescue-label models used the first
10,000 pairs of a separately sampled 20,000-pair seed-1 batch on Apple M5,
with both decoders evaluated on the same local shots. The fallback ran only
on the 965 basis-level Relay-BP-1 failures and rescued 365. The sampled data
and fallback call history differ between the local run and a full BlueQubit
Relay-BP-5 sweep. The models also use different training sample counts, so
the replay does not isolate the effect of changing the label.

| Policy | Actual escalation | Failed pairs | Mean ms | p99 ms |
|---|---:|---:|---:|---:|
| Relay-BP-1 | 0% | 147 | 156 | 3,131 |
| Relay-BP-5 | — | 96 | 505 | 9,393 |
| Failure-label boosted | 2.02% | 122 | 322 | 10,840 |
| Simple rule matched to failure-label boosted | 2.02% | 125 | 335 | 11,114 |
| Random matched to failure-label boosted | 2.02% | 143 | 168 | 3,535 |
| Rescue-label logistic | 2.09% | 125 | 320 | 10,871 |
| Simple rule matched to rescue-label logistic | 2.09% | 123 | 338 | 11,114 |
| Random matched to rescue-label logistic | 2.09% | 146 | 164 | 3,191 |
| Rescue-label boosted | 2.08% | 127 | 308 | 10,101 |
| Simple rule matched to rescue-label boosted | 2.08% | 121 | 337 | 11,114 |
| Random matched to rescue-label boosted | 2.08% | 146 | 169 | 3,352 |

The selected failure-label boosted model had the fewest failed pairs among
the learned policies near 2%. It won 9 discordant pairs against the
matched simple rule and lost 6 (exact paired p = 0.607). Its advantage over
one matched-random draw was 41 wins to 20 losses (exploratory p = 0.00985).
The selected router cut mean replay latency by 36% versus Relay-BP-5, with
26 more failed pairs and higher p99. It is a candidate for a live
tradeoff measurement. Loss-aware decoding remains the headline result.

## Single-worker handoff

Use the packaged model from this trusted branch. It was exported with
scikit-learn 1.9.1; use the same version for loading. Seed 7 is fresh for
this selected policy. Rerun both baselines with that seed if publishing
a matched live comparison.

```bash
python benchmark_timing.py --p 0.004 --p-loss 0 --shots 1000 --seed 7 \
  --decoders relay1+H relay+H live-relay \
  --live-model results/router_calibrated_p004/selected_boosted.joblib \
  --out final_timing_p004_seed7.csv
```

The replay CSVs and plots cover both budgets. The CSVs include the exact
achieved rates. The archived decisions for the selected model include the learned,
matched simple, and matched random masks.
