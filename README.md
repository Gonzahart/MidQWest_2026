# MidQWest 2026 — Infleqtion Challenge 1: ML-assisted decoding of the Gross code

Team repo for the MidQWest 2026 hackathon (UIC, Nov 21–22). Team: Sena Ozcan, Gonzalo Hartstein, Sean Guirola,
Lena Rotteveel, Kahnishga Solaidurairaj.

We decode the [[144,12,12]] Gross code (12-cycle circuit-level X/Z memory, Stim, p = 0.003 / 0.004 / 0.005) and improve
the kit baselines (BP-OSD, relay-BP) in three ways: **(1) atom-loss flags as per-shot decoder priors**, **(2) a learned
router that decides when relay-BP-1's answer needs relay-BP-5**, and **(3) relay-BP on a consumer GPU**. Every
comparison is paired on the same simulated shots; every ML test was fixed in advance and run once on fresh seeds.

**Slides:** [Faster_more_accurate_decoding_of_the_Gross_code_v2.pdf](Faster_more_accurate_decoding_of_the_Gross_code_v2.pdf)

## Approach in one paragraph

Each shot's syndrome (936 parity checks per basis) is decoded by relay-BP. **Loss-aware priors:** when a data atom is lost,
the hardware raises a herald flag; a lost atom leaves I/X/Y/Z with equal odds, so in one memory basis the flagged error
column has probability exactly 1/2, and unflagged loss columns ~0. We set those priors per shot on the unchanged check
matrix (rebuild ~5 ms). **Learned routing (the ML part):** relay-BP-1 decodes every shot; a gradient-boosted model
(monotone-constrained) reads 7 features of that decode (+3 flag features with loss) and predicts the net benefit of
escalating (relay-BP-5 fixes − breaks); shots above a threshold fixed on a separate calibration seed are re-decoded with
relay-BP-5. A 6-bit truth-table version of the router fits one FPGA lookup table. **GPU:** the same relay-BP-5 runs
through NVIDIA CUDA-Q QEC on an RTX 3060 (fp32, batch 1).

## Results at a glance

A trial (X/Z pair) fails if either memory fails; LER = per-cycle logical error rate `1-(1-P)^(1/12)`.
Full tables, caveats and file lists: [`challenge-1-ml-decoder-kit/results/README.md`](challenge-1-ml-decoder-kit/results/README.md).

| Result | Number | Where |
|---|---|---|
| Loss flags as priors (fired flag → prior 1/2, unfired → ~0), relay-BP-5, 1% loss | LER 4.8× / 2.6× / 2.2× lower at p = 0.003 / 0.004 / 0.005; BP-OSD 6.7× / 2.7× / 1.9× | `results/loss_sweep_relay5.png`, `loss_sweep_bposd.png` |
| ...and faster (single worker, 500 pairs, seed 6) | mean 112→62, 547→261, 1,759→1,146 ms per pair | `results/timing/` |
| Control: gain comes from each shot's own flags (p = 0.004, 1k pairs) | real flags 8 · no loss info 42 · flags from another shot 59 · flags ignored on the same graph 89 failed pairs | results README, "Herald control" |
| The 1/2 prior is exact | Pauli injection at all 1,728 flag positions; one check-supported column each, prior/p_loss 0.499–0.500 | `check_herald_prior.py`, `results/herald_prior_check.txt` |
| Flags beat 5× compute (p = 0.004, 1% loss, 4k pairs) | relay-BP-1 + flags 87 vs relay-BP-5 without flags 211 failed pairs | `results/flags_vs_compute/` |
| Robust to imperfect flags (p = 0.004, 1% loss) | 3.4× perfect · 3.2× with 20% false flags · 2.2× with 20% missed | `results/robust/` |
| Net-benefit router vs simple rule at equal mean time, p = 0.005, fresh seeds | +44 [17, 72] (seed 14), +42 [15, 68] (seed 32), +30 [12, 48] with 1% loss (seed 26); ties at p = 0.004 and 0.003 | `results/overnight/`, `econ_router.py` |
| 6-bit truth-table router (FPGA-LUT sized), fresh seed 32 | simple-rule accuracy at 27–40% less mean time | `results/fresh_s32/`, `truth_table_router.py` |
| GPU relay-BP-5 (RTX 3060, fp32, batch 1, same shots) | mean / p99 25 / 315 ms vs CPU 206 / 3,287 ms at p = 0.004; 87 / 486 vs 905 / 4,698 ms at p = 0.005; accuracy within noise | `results/Sean GPU-relay/` |
| Official single-worker CPU latency (seed 16) | BP-OSD, relay-BP-1, relay-BP-5 at all three p | `results/timing/final timing/` |

**Comparison plots (deliverable):**
[loss flags vs baseline, LER at p = 0.003/0.004/0.005](challenge-1-ml-decoder-kit/results/figures/fig_loss.png) ·
[router vs simple rule, random choice and relay-BP baselines](challenge-1-ml-decoder-kit/results/figures/fig_router.png) ·
[GPU vs CPU relay-BP-5 worst-case latency](challenge-1-ml-decoder-kit/results/figures/fig_gpu.png) ·
[loss sweep, relay-BP-5](challenge-1-ml-decoder-kit/results/loss_sweep_relay5.png) ·
[loss sweep, BP-OSD](challenge-1-ml-decoder-kit/results/loss_sweep_bposd.png)

Limits: loss model = prompt, perfect-location heralds on data qubits at cycle start; ML router helps only at p = 0.005;
router p99 is worse than relay-BP-5 alone (escalated shots pay both decoders). Closest prior work: Murri,
arXiv 2610.38283 (loss-aware relay-BP on the Gross code for delayed, correlated loss; no latency reported).

## Quick start

```bash
git clone https://github.com/Gonzahart/MidQWest_2026
cd MidQWest_2026/challenge-1-ml-decoder-kit
python -m venv .venv && source .venv/bin/activate          # Python 3.11 or 3.12
pip install -r requirements.txt scikit-learn pytest
# relay-bp builds from source (needs Rust, rustup.rs); on Linux x86-64 + Python 3.11 use the prebuilt wheel:
#   pip install wheels/relay_bp-0.2.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
python -m pytest test_kit.py                                # kit sanity tests
python sweep.py run --decoder relay5 --p 0.004 --p-loss 0.01 --loss-mode reweight --shots 200 --seed 99 --workers 4
python sweep.py summarize results/relay5_p0.004_l0.01_reweight_s99_n200_shard0of1.npz
```

## Reproduce the headline results

From `challenge-1-ml-decoder-kit/` with the quick-start environment active.
`sweep.py run` writes per-shot `.npz` files (failures, times, iterations, features); `sweep.py summarize` prints tables.

```bash
# Baselines and loss sweep (flags ignored vs used), any p / loss rate
python sweep.py run --decoder relay5 --p 0.004 --p-loss 0.01 --loss-mode unaware  --shots 2000 --seed 2
python sweep.py run --decoder relay5 --p 0.004 --p-loss 0.01 --loss-mode reweight --shots 2000 --seed 2
python sweep.py run --decoder bposd  --p 0.004 --p-loss 0.01 --loss-mode reweight --shots 2000 --seed 2
python sweep.py summarize results/*.npz
python plot_loss.py results/relay5_*.npz --out results/loss_sweep_relay5.png

# Herald control (blind / shuffled flags) and the 1/2-prior injection check
for m in reweight unaware shuffled blind; do
  python sweep.py run --decoder relay5 --p 0.004 --p-loss 0.01 --loss-mode $m --shots 1000 --seed 5; done
python check_herald_prior.py > results/herald_prior_check.txt

# Flag robustness (prior value, missed and false flags)
bash run_robustness.sh

# Net-benefit router: train seed 2, thresholds seed 4, test once on a fresh seed (same shots for relay1/relay5)
for d in relay1 relay5; do python sweep.py run --decoder $d --p 0.005 --shots 10000 --seed 14; done
python econ_router.py --monotone \
  --train-relay1 results/relay1_p0.005_l0.0_unaware_s2_n10000_shard0of1.npz \
  --train-relay5 results/relay5_p0.005_l0.0_unaware_s2_n10000_shard0of1.npz \
  --calibration results/relay1_p0.005_l0.0_unaware_s4_n20000_shard0of1.npz \
  --test-relay1  results/relay1_p0.005_l0.0_unaware_s14_n10000_shard0of1.npz \
  --test-relay5  results/relay5_p0.005_l0.0_unaware_s14_n10000_shard0of1.npz --out results/router_s14_p0.005
# the whole pre-registered test (all p, with and without loss): bash run_overnight.sh

# Official single-worker latency (one machine, one thread, nothing else running)
for p in 0.003 0.004 0.005; do for d in relay1 relay5 bposd; do
  OMP_NUM_THREADS=1 python sweep.py run --decoder $d --p $p --shots 1000 --seed 16 --workers 1 --out results/final_timing; done; done
# (our committed run of this is in results/timing/final timing/)
```

Seeds used (never tune on a test seed): 1/2 train · 4 thresholds · 5 herald control · 6 loss timing · 8 GPU check ·
10, 14, 32 fresh router tests · 12 robustness · 16 final timing · 22/24/26 loss-router train/thresholds/test.
Files from different machines or worker counts draw different shots: pair only matching files.

## Layout

| Path | What it is |
|---|---|
| `challenge-1-ml-decoder-kit/` | Infleqtion's decoder kit (baselines, circuits) plus our code |
| `challenge-1-ml-decoder-kit/loss_circuit.py` | Adds heralded atom loss (Stim `HERALDED_ERASE` + flag detectors) to the Gross code circuits |
| `challenge-1-ml-decoder-kit/sweep.py` | Main experiment runner: decoders × p × loss mode (unaware / reweight / blind / shuffled / noisy flags), per-shot `.npz` output |
| `challenge-1-ml-decoder-kit/econ_router.py` | Net-benefit router: train, set thresholds, test against the simple rule and random choice at equal time |
| `challenge-1-ml-decoder-kit/truth_table_router.py` | Frozen 6-bit truth-table router |
| `challenge-1-ml-decoder-kit/check_herald_prior.py` | Pauli-injection proof of the 1/2 prior |
| `challenge-1-ml-decoder-kit/results/` | All results; `results/README.md` explains every table |
| `challenge-1-ml-decoder-kit/evaluate.py` | Early comparison script (decoders with and without loss flags) |
| `docs/TEAM_GUIDE.md` | Team guide (snapshot of the live doc) |
| `docs/context/` | Planning and research notes: track analysis, Infleqtion insights, challenge plan, decoder literature |
| `docs/reference/` | Challenge README, Sean's handoff and research report, Fowler's surface code paper |
| `prep/` | Friday-night prep: surface code threshold baseline with Stim + PyMatching |

## Reproduce the frozen truth-table routers

From `challenge-1-ml-decoder-kit/` with the quick-start environment active,
replay both tables on the committed no-loss seed-16 run at p = 0.005.
The two inputs are matched Relay-BP-1 and Relay-BP-5 per-shot archives from
the same host, with 1,000 X/Z pairs and one worker.

```bash
timing="results/timing/final timing"
suffix="p0.005_l0.0_unaware_s16_n1000_shard0of1.npz"
for table in truth_table_router_frozen truth_table_router_gpu_cost_frozen; do
  python truth_table_router.py \
    "$timing/relay1_$suffix" "$timing/relay5_$suffix" \
    --table "results/truth table/$table.json" \
    --out "results/repro/${table}_p0.005_s16.csv"
done
```

Pass `--table` explicitly because the script's default path uses an underscore
and the repository folder uses a space. The frozen policies are the
[CPU-priced table](challenge-1-ml-decoder-kit/results/truth%20table/truth_table_router_frozen.json)
and the
[GPU-priced table](challenge-1-ml-decoder-kit/results/truth%20table/truth_table_router_gpu_cost_frozen.json).
To score another held-out run, pass its matched Relay-BP-1 and Relay-BP-5
archives and choose a new CSV output path. The replay prints paired failure
comparisons and writes logical error rate, escalation rate, and latency to CSV.

Both replay commands use the decode times recorded in the input archives,
including when scoring the GPU-priced policy. The separate
[GPU cost projection](challenge-1-ml-decoder-kit/results/truth%20table/gpu_cost_projection.py)
estimates GPU fallback timing by scaling the recorded Relay-BP-5 times.

| Configuration | CPU | Host memory | GPU memory |
|---|---|---|---|
| Single-worker Relay-BP benchmarks | 1 CPU core | ~120 MiB peak worker RSS in the CPU smoke test | ~22 MiB additional device memory observed in the exploratory GPU probe |

[CPU timing and memory smoke test](challenge-1-ml-decoder-kit/results/README.md#task-8-single-worker-timing-smoke-test) · [GPU timing and memory notes](challenge-1-ml-decoder-kit/results/Sean%20GPU-relay/README.md#results)
