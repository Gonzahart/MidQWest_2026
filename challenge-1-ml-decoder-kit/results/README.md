# Results

Merge any sweep output with `python sweep.py summarize results/*.npz`.

## evaluate_p004_loss005_2000shots (Gonzalo's Mac, Sat 4:30 PM)
p = 0.004, p_loss = 0.005, 2000 shots per basis, same shots for every decoder. `+H` = loss flags as extra rows of H.

| Decoder | Fail X, Z | LER / cycle | Mean ms | p99 ms | To stage 2 |
|---|---|---|---|---|---|
| bposd | 104, 88 | 8.1e-3 ± 0.6e-3 | 429 | 1304 | |
| bposd+H | 64, 68 | 5.6e-3 ± 0.5e-3 | 903 | 3086 | |
| relay (relay-BP-5) | 24, 20 | 1.85e-3 ± 0.28e-3 | 474 | 4804 | |
| relay+H | 22, 21 | 1.81e-3 ± 0.28e-3 | 916 | 8377 | |
| bp>relay+H | 22, 15 | 1.55e-3 ± 0.26e-3 | 910 | 8853 | 68.7% |

Takeaways
- 0.5% loss roughly doubles relay-BP's error rate vs the kit's no-loss 8.7e-4 at p = 0.004.
- Flags-as-rows helps BP-OSD (1.4x) but not relay-BP, and doubles decode time: a weak loss-aware baseline.
- Plain BP converges on only ~31% of lossy shots, so a BP-first stage saves little time under loss.

## Reweighted loss priors (Gonzalo's Mac, Sat 5:15 PM) — `sweep.py --loss-mode reweight`
p = 0.004, p_loss = 0.005, 2000 shots per basis, seed 1 (same shots as above). Files: `*_l0.005_*_n2000_*.npz`.

| Decoder | Loss info | Fail X, Z | LER / cycle | Mean ms | p99 ms |
|---|---|---|---|---|---|
| BP-OSD | ignored | 104, 88 | 8.1e-3 ± 0.6e-3 | 391 | 1394 |
| BP-OSD | reweighted priors | 61, 66 | 5.4e-3 ± 0.5e-3 | 412 | 1303 |
| relay-BP-5 | ignored | 28, 20 | 2.0e-3 ± 0.3e-3 | 414 | 4252 |
| relay-BP-5 | reweighted priors | 13, 11 | **1.0e-3 ± 0.2e-3** | 346 | 4827 |

Takeaways
- Reweighting halves relay-BP-5's error rate under 0.5% loss and brings it close to the kit's no-loss value (8.7e-4 at p = 0.004), at no extra mean latency.
- BP-OSD improves 1.5x. Reweighting beats flags-as-rows on both accuracy and speed.
- This is the "naive loss-aware" baseline that any ML loss component must beat.

## Held-out Relay-BP-1 routing pilot

Physical error rate 0.005, stock Gross circuits without loss. Training used 300
X/Z pairs (X seed 41, Z seed 42) and produced 11 basis shots where Relay-BP-5
rescued Relay-BP-1. The held-out set used 200 pairs (X seed 2, Z seed 3).
Both decoders ran on the same shots. Timings were gathered on an arm64 Mac
with eight worker processes and single-threaded BLAS.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python relay_router_pilot.py \
  --workers 8 --out /tmp/relay_router_pilot_p005
```

The logistic model uses Relay-BP-1 syndrome weight, iterations, convergence,
and two posterior confidence summaries. The simple rule ranks nonconvergence
first and longer runs next. The learned, simple, and random policies each
route exactly 20% of shots in each basis, using an offline ranking budget fixed
before evaluation. Replay timing sums decoder calls and excludes feature
extraction and inference. Live Relay-BP outcomes can vary with call history.

On the held-out split, the learned and simple rules each failed 12 pairs, with
one pair won by each rule and an exact paired p-value of 1. Learned routing
reduced mean replay latency from 815 ms to 618 ms, with p99 of 5,633 ms for both.
Relay-BP-5 alone also failed 12 pairs, with 714 ms mean and 3,335 ms p99.
The pilot gives no evidence of an accuracy gain over the simple rule. The mean
latency difference needs a larger held-out sample and live timing.

[Summary CSV](relay_router_pilot_p005.csv) · [Pilot cache](relay_router_pilot_p005.npz)

## Loss prior fix check (Gonzalo's Mac, Sat 6:05 PM)
Same shots as above (seed 1, 2000 shots, p = 0.004, p_loss = 0.005) with the corrected prior P(column | loss) = 1/2
(was 1/4). Files kept locally in `results_v2/`.

| Decoder | Prior 1/4 | Prior 1/2 (fixed) |
|---|---|---|
| BP-OSD, reweighted | 5.4e-3 ± 0.5e-3 | 5.5e-3 ± 0.5e-3 |
| relay-BP-5, reweighted | 1.0e-3 ± 0.2e-3 | 1.1e-3 ± 0.2e-3 |

The fix changes nothing measurable: what matters is marking where loss happened (unfired heralds -> ~0,
fired -> large prior), not the exact value.

## Larger held-out Relay-BP-1 routing replay

Physical error rate 0.005, no loss, 2,000 training pairs and 2,000 held-out
pairs. Train X/Z seeds were 41/42; held-out X/Z seeds were 2/3. This scales up
the pilot with the same seeds and features. Training had 75 basis shots where
Relay-BP-5 rescued Relay-BP-1. Both learned and simple rules used a fixed 20%
fallback rate in each basis. The replay ran on an arm64 Mac with 10 workers.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python relay_router_pilot.py --p 0.005 --train-shots 2000 --test-shots 2000 \
  --train-seed 41 --test-seed 2 --workers 10 --fallback-rate 0.20 \
  --out /tmp/relay_router_scaleup_p005
```

| Policy | Failed pairs | LER / cycle | Mean ms | p99 ms |
|---|---:|---:|---:|---:|
| Relay-BP-1 | 173 | 7.51e-3 | 374 | 3,866 |
| Relay-BP-5 | 137 | 5.90e-3 | 890 | 4,759 |
| Simple rule | 144 | 6.21e-3 | 1,038 | 7,826 |
| Learned | 152 | 6.57e-3 | 956 | 7,561 |
| Matched random | 176 | 7.65e-3 | 560 | 7,032 |

The learned router saved 7.8% mean replay time against the simple rule and
failed eight additional pairs. The learned rule won four discordant pairs
and the simple rule won 12 (exact paired p = 0.0768). Relay-BP-5 alone had
fewer failures, lower mean latency, and lower p99 than the learned router in
this run. These results do not support expanding the current policy to a
live three-rate benchmark. Replay latency excludes feature extraction and
inference.

[Summary CSV](relay_router_scaleup_p005.csv) · [Replay cache](relay_router_scaleup_p005.npz)

The pilot script also performs a **live learned-routing pass** on the held-out shots. It
calibrates one fallback threshold per basis from training scores, then runs Relay-BP-1,
feature extraction, model inference, and conditional Relay-BP-5 calls in sequence. New
pilot outputs include this row in the CSV and a latency-versus-error plot next to the
requested output prefix. The live row is the end-to-end measurement; cached replay rows
remain useful for comparing fixed routing decisions but exclude feature and inference time.

## Task #8 single-worker timing smoke test

30 matched X/Z pairs at p = 0.005, no loss, seed 2 on an Apple M5 with one
decoder worker and numerical library threads set to one. Each policy ran in
a fresh process. The small sample verifies the timing and memory pipeline.
Peak worker RSS includes the runtime, sampled data, and decoder state.

| Policy | Failed pairs | Mean ms | p99 ms | Peak worker RSS (MiB) |
|---|---:|---:|---:|---:|
| Relay-BP-1 | 9 | 597.2 | 2,139.5 | 117.2 |
| Relay-BP-5 | 5 | 891.8 | 2,379.4 | 117.0 |
| Convergence-only | 6 | 903.6 | 2,172.1 | 119.5 |

[Full smoke-test CSV](single_worker_timing_smoke_p005.csv) ·
[Run instructions](../README.md#single-worker-timing-and-resource-benchmark)

The final 1,000–2,000-pair comparison is pending the agreed benchmark machine and a lower-budget trained router. Baseline timing can proceed independently.

## Loss prior check on Sean's x86 PC (Sat 6:57 PM)

At p = 0.004 and p_loss = 0.005, all five settings used the same 2,000 X/Z pairs on this PC (seed 1). The Mac and PC sampled different shots with the same seed. Each paired comparison uses shots from a single machine.

| Decoder | Loss info | Fail X, Z | LER / cycle | Mean ms | p99 ms |
|---|---|---|---|---|---|
| BP-OSD | ignored | 97, 99 | 8.2e-3 ± 0.6e-3 | 265 | 681 |
| BP-OSD | reweighted, prior 1/4 | 62, 54 | 4.9e-3 ± 0.5e-3 | 297 | 884 |
| BP-OSD | reweighted, P(col \| loss) | 61, 57 | 4.8e-3 ± 0.5e-3 | 300 | 884 |
| relay-BP-5 | ignored | 28, 34 | 2.6e-3 ± 0.3e-3 | 523 | 6576 |
| relay-BP-5 | reweighted, P(col \| loss) | 11, 12 | **9.6e-4 ± 2.0e-4** | 345 | 5368 |

The two-sided exact sign test counts pairs where only one setting failed.

| Decoder | Settings compared (first vs second) | Only first failed | Only second failed | Exact test p |
|---|---|---:|---:|---:|
| BP-OSD | Reweighted vs ignored | 51 | 127 | 1.15e-8 |
| relay-BP-5 | Reweighted vs ignored | 8 | 47 | 8.07e-8 |
| BP-OSD | Prior 1/4 vs conditional prior | 51 | 50 | 1 |

On the PC, reweighting reduced relay-BP-5's logical error rate per cycle by a factor of 2.7. The Mac also showed a reduction, about 2×. For BP-OSD, the paired comparison found no detectable difference between the two fired-herald priors at this noise level.

Mean relay-BP-5 time fell from 523 to 345 ms on the PC. Eight workers ran concurrently, so these timings are approximate. Sean keeps the per-shot files (`results/t3_*`) on the PC because their run metadata matches the Mac files and `summarize` would merge them.
