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

## Loss sweep, relay-BP-5 (Sat 9:15 PM) — headline result
Logical error rate per cycle, seed 2. p = 0.003: 20,000 pairs per point (BlueQubit). p = 0.004 / 0.005: 2,000 pairs
per point (Gonzalo's Mac). 0% loss: 10,000-pair no-loss runs (BlueQubit). Figure: `loss_sweep_relay5.png`.

| p | Loss | Loss ignored | Loss-reweighted | Gain |
|---|---|---|---|---|
| 0.003 | 0% | 8.3e-5 | 8.3e-5 | - |
| 0.003 | 0.2% | 7.1e-5 ± 1.7e-5 | 6.3e-5 ± 1.6e-5 | 1.1x (n.s.) |
| 0.003 | 0.5% | 1.92e-4 ± 0.28e-4 | 1.08e-4 ± 0.21e-4 | 1.8x |
| 0.003 | 1.0% | 4.97e-4 ± 0.46e-4 | 1.04e-4 ± 0.21e-4 | **4.8x** |
| 0.004 | 0.2% | 9.2e-4 | 5.9e-4 | 1.6x |
| 0.004 | 0.5% | 1.81e-3 | 1.05e-3 | 1.7x |
| 0.004 | 1.0% | 3.92e-3 | 1.51e-3 | 2.6x |
| 0.005 | 0.2% | 8.97e-3 | 7.19e-3 | 1.25x |
| 0.005 | 0.5% | 1.27e-2 | 7.24e-3 | 1.75x |
| 0.005 | 1.0% | 2.26e-2 | 1.03e-2 | 2.2x |

The gain is largest where ordinary noise is lowest: at p = 0.003 and 1% loss, ignoring loss makes the decoder 6x worse
than no loss; with reweighted priors it is only 1.25x worse. Mean decode times stay in the same range (p = 0.003: 183-208 ms
reweighted vs 165-310 ms ignored, 20 parallel workers; not final timing).

## Loss sweep, BP-OSD (Gonzalo's Mac, finished 9:10 PM)
Seed 2, 2,000 pairs per point, 8 workers. "Fixed / broken" = pairs where only the loss-ignoring / only the reweighted decoder failed.

| p | Loss | Ignored | Reweighted | Gain | Fixed / broken |
|---|---|---|---|---|---|
| 0.003 | 0.2% | 5.9e-4 | 4.2e-4 | 1.4x (n.s.) | 9 / 5 |
| 0.003 | 0.5% | 1.39e-3 | 5.0e-4 | 2.8x | 28 / 7 |
| 0.003 | 1.0% | 3.09e-3 | 4.6e-4 | **6.7x** | 69 / 7 |
| 0.004 | 0.2% | 4.97e-3 | 4.75e-3 | 1.05x (n.s.) | 52 / 47 |
| 0.004 | 0.5% | 8.93e-3 | 4.57e-3 | 2.0x | 135 / 38 |
| 0.004 | 1.0% | 1.70e-2 | 6.34e-3 | 2.7x | 280 / 55 |
| 0.005 | 0.2% | 2.63e-2 | 2.39e-2 | 1.1x | 214 / 170 |
| 0.005 | 0.5% | 3.67e-2 | 2.57e-2 | 1.4x | 320 / 134 |
| 0.005 | 1.0% | 5.68e-2 | 3.02e-2 | 1.9x | 502 / 108 |

- Reweighting helps BP-OSD too, with the same pattern as relay-BP-5: bigger gains at more loss and lower p. At p = 0.003,
  1% loss, reweighted BP-OSD (4.6e-4) is back at its no-loss value (4.8e-4).
- **Timing caveat:** reweighted BP-OSD was slower in these runs (mean +20-50%, p99 up to 2.7x), likely because OSD
  runs more often with sharp priors. Relay-BP-5 showed no consistent slowdown. Check in the single-worker timing.

## Herald control: is the gain from the flags themselves? (Sat ~11:45 PM)
relay-BP-5, p = 0.004, 1% loss, seed 5 (fresh), 1,000 pairs per arm, all arms on one machine (2-core cloud VM)
and the same sampled shots. `reweight`, `blind` and `shuffled` use the same projected graph (kit-size H) and
the same per-shot relay rebuild; they differ only in the priors:
- `reweight`: fired flags -> prior/p_loss (≈ 1/2), unfired -> ~0 (our method)
- `blind`: unconditional DEM priors, herald identity ignored (controls for the graph change and the rebuild)
- `shuffled`: our rule, but with the flag pattern of a different shot (k -> k + 500)
- `unaware`: the kit-style DEM with heralds dropped (the "loss ignored" baseline in the sweep)

| Arm | Failed pairs | LER/cycle | Arm-only / reweight-only failures | Exact paired p | Mean ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|
| reweight | 8 | 6.7e-4 | – | – | 386 | 5,981 |
| unaware | 42 | 3.6e-3 | 36 / 2 | 5e-9 | 828 | 6,078 |
| shuffled | 59 | 5.1e-3 | 53 / 2 | 9e-14 | 1,398 | 8,660 |
| blind | 89 | 7.7e-3 | 84 / 3 | 1e-21 | 2,290 | 11,659 |

The graph change does not explain the gain: on the same graph without flag information (`blind`) the decoder
is *worse* than the loss-ignoring baseline, and the right flags take it from 89 to 8 failed pairs. Flags from
the wrong shot (`shuffled`) also hurt, so the gain needs each shot's own flags. Timing is from a slow shared
VM: compare arms with each other only. Reproduce:
`python sweep.py run --decoder relay5 --p 0.004 --p-loss 0.01 --loss-mode {reweight,unaware,shuffled,blind} --shots 1000 --seed 5`

## Plot provenance fix
`plot_loss.py` now keeps one run per point (largest requested shot count, single host), refuses incomplete or
overlapping shards, and writes the host, input files and skipped files into the CSV next to the PNG.
Regenerated `loss_sweep_relay5.png/.csv` and `loss_sweep_bposd.png/.csv`: every number matches the tables
above, except the p = 0.003 no-loss point now uses the 50k run (5.3e-5 instead of 8.3e-5 from 10k).

## Why the fired-herald prior is 1/2: injection check (Sun ~12:30 AM)
`python check_herald_prior.py > results/herald_prior_check.txt`. For data qubits 72 and 150, lost in cycle 0, 5
and 11, in both memory bases, we injected I, X, Y and Z by hand at the erasure location in a noiseless copy of
the circuit and compared the flipped check detectors and logicals with the decoder's DEM columns on that herald.
- X memory: I and X do nothing (no X-check or X-logical flip). Y and Z flip the same 3 checks and the same
  logicals. So the four equally likely branches collapse into "nothing" (prob 1/2) and one column (prob 1/2).
  Z memory: the same with I+Z and X+Y. Every case matched the DEM exactly (column, logicals, mapping).
- Every one of the 1,728 heralds (12 cycles x 144 data qubits) has exactly one column with check support, with
  prior/p_loss = 0.499-0.500. So for a fired herald the merged channel has only one relevant branch; the
  "independent mechanisms" approximation does not matter there.
- The projection onto check rows (Reweighted) drops 1,728 herald-only columns per basis; none flips a logical.
- Sizes: with heralds, H is 2,664 rows (936 check + 1,728 herald) x 12,240 columns, 39,312 nonzeros. Projected
  (what reweight decodes): 936 rows x 10,512 columns. Kit no-loss problem: 936 x 8,784.

## Router vs random decoder choice at equal mean time (Sun ~2 AM)
"Coin flip" = for each X/Z pair choose relay-BP-1 or relay-BP-5 at random *before* decoding, with the probability set so
the mean time equals the router's (2,000 random draws; mean and 95% range of failed pairs). This is a fairer baseline
than escalating random shots, which pays relay-BP-1 first. Replay times, 20-worker caches: compare within a row only.

| p | Test data | Router (locked policy) | Esc. | Router failed pairs / mean ms / p99 | Coin flip at same mean | Simple rule, same esc. |
|---|---|---|---:|---|---|---|
| 0.003 | seed 2, 50k pairs, seed-4 calibration | boosted 2% | 1.3% | 47 / 54 / – | 52.4 (48–57), p99 468 | 48 |
| 0.004 | seed 2, 10k (Sena's selection) | boosted 2% | 2.0% | 122 / 322 / 10,840 | 122.8 (113–133), p99 6,089 | 125 |
| 0.004 | seed 4 validation, 10k | logistic | 7.4% | 98 / 469 / 11,430 | 97.4 (88–106), p99 9,456 | 88 (16%) |
| **0.005** | **seed 10, fresh, 10k; rule fixed in advance** | **logistic 2%** | **2.2%** | **851 / 1,441 / 19,233** | **903.5 (884–924), p99 10,378** | **847** |

Reading: at p = 0.005 escalation beats random decoder choice by ~50 failed pairs (~6%) at equal mean time, on fresh
data with the policy fixed beforehand; at 0.003 it is marginally better; at 0.004 it ties. The learned model ties the
simple rule (not converged first, then most iterations) everywhere, and p99 is always worse than random choice, because
escalated shots are the slowest ones and pay for both decoders. Note: boosted with seed-4 calibration at p = 0.005
escalated 0% (threshold above every test score), so the pre-registered logistic policy is the one reported.
CSVs: `router_fresh_s10_p0.005.csv`, `router_calibrated_s2_p0.003.csv`. Coin flip: `python router_coinflip.py <relay1 file> <relay5 file> <router mean ms>`.

## Leg cap + OSD-0 fallback (`relay1capK`, BlueQubit batch 4, Sun ~2:15 AM)
Seed 10, 10k pairs, all runs on one machine with 20 workers (compare within the table only). Relay-BP-1 with at most K
relay legs; if unconverged, OSD-0 on relay's own posteriors.

| p | Decoder | Failed pairs | Mean ms | p99 ms | Max ms | vs relay-BP-1 |
|---|---|---:|---:|---:|---:|---|
| 0.004 | relay-BP-1 | 140 | 166 | 3,254 | 9,737 | — |
| | cap 200 + OSD-0 | 146 | 152 | 3,235 | 7,043 | +6 (p = 0.38) |
| | cap 100 + OSD-0 | 162 | 127 | 2,838 | 4,027 | +22 (p = 0.006) |
| | relay-BP-5 | 102 | 564 | 9,609 | 18,045 | −38 |
| 0.005 | relay-BP-1 | 938 | 1,043 | 9,704 | 19,265 | — |
| | cap 200 + OSD-0 | 1,005 | 865 | 6,916 | 13,292 | +67 (p = 0.003) |
| | cap 100 + OSD-0 | 1,178 | 627 | 3,866 | 7,255 | +240 |
| | relay-BP-5 | 817 | 2,447 | 12,908 | 19,228 | −121 |

A cap is a speed/accuracy dial, not a free lunch: cap 200 is nearly free at p = 0.004, but tighter caps lose accuracy
fast because OSD-0 is a weak fallback. Relay-BP's random memory draws carry over between calls, so capped runs were
decoded for real, not simulated from caches.

## ML early exit after relay-BP's first leg (negative result)
`exit_data.py` / `exit_router.py`, p = 0.005, seed 1, 10k pairs (Mac), split in halves for train/test. On the 36% of
basis shots the first leg leaves unconverged (98% of decode time), continuing relay-BP fails 13% and OSD-0 fails 48%;
only 149 of 7,141 are OSD-0-right / relay-wrong. Classifier AUROC 0.62 for "keep going helps", and the simple rule
(continue the shots with the fewest unsatisfied checks) beats it; any exit rate that saves ~25% mean time costs ~25%
more failed pairs. Dropped: OSD-0 is too weak a fallback for an exit to pay.

## Net-benefit ("economic") router (Sean's question, Sun ~2:45 AM)
`econ_router.py`. Label per basis shot = int(relay-BP-1 wrong) − int(relay-BP-5 wrong) (+1 rescue, −1 regression).
Gradient-boosted regressor on relay-BP-1's features; escalate when predicted benefit is above a threshold fixed on a
calibration set. Train seed 2 (relay-BP-1 and relay-BP-5 on the SAME shots), thresholds seed 4, test seed 10, p = 0.005.

| Policy | Escalated | Failed pairs | Mean ms | Random decoder choice at same mean |
|---|---:|---:|---:|---:|
| relay-BP-1 alone | 0% | 938 | 1,043 | — |
| net benefit, 2% budget | 2.2% | **837** (fixes 102, breaks 1 vs relay-BP-1) | 1,434 | 904 |
| simple rule, 2% | 2.1% | 850 | 1,437 | 904 |
| earlier logistic router (locked) | 2.2% | 851 | 1,441 | 903 |
| net benefit, 10% budget | 8.5% | **816** (= relay-BP-5, paired p = 1) | ~2,030 | 853 |
| relay-BP-5 alone | 100% | 817 | 2,447 | — |

- Predicting net benefit (including the 211 basis shots where relay-BP-5 is wrong and relay-BP-1 right) beats predicting
  "relay-BP-1 wrong": relay-BP-5 accuracy at ~17% less mean time.
- Adding a predicted cost (benefit / cost ranking) made it worse: it favours cheap, low-benefit shots.
- p99 stays ~19 s (escalated shots pay both decoders).
- Caveats: seed 10 was used once before (earlier router, not this model); model trained on another machine.
  Fresh confirmation at all three p on seed 14 is queued on BlueQubit.
- Correction: the seed-1 relay-BP-1 (20k) and relay-BP-5 (10k) caches are NOT the same shots (different shot counts
  draw different samples; 1.75% of syndromes match), so rescue labels built from them were wrong. Always pair files
  with the same seed, shot count and machine.
