# Results

One folder per experiment, in the order of the talk. Per-shot files are `sweep.py` outputs named
`{decoder}_p{p}_l{loss}_{mode}_s{seed}_n{shots}_shard{i}of{n}.npz` (failures, decode seconds, iterations and router
features per X/Z shot); `python sweep.py summarize <files>` prints a table. The full chronological record, with every
table, caveat and correction from the hackathon, is in [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md).

Timing rule: only one-worker, one-thread runs on one machine are latency claims (`05_timing/`, `08_gpu/`). Everything
else was run with many parallel workers: compare times within a table only. Files from different machines, shot counts
or worker counts contain different shots: pair only matching files.

| Folder | What | Headline | Produced by |
|---|---|---|---|
| [`01_baselines/`](01_baselines) | Kit baselines (BP-OSD, relay-BP) without loss; early sweeps and BlueQubit timing | relay-BP-5 ≈ 3× fewer failures than BP-OSD at p = 0.5% | `baseline.py`, `sweep.py`, `benchmark_timing.py` |
| [`02_loss_sweep/`](02_loss_sweep) | Loss flags ignored (`unaware`) vs used (`reweight`), relay-BP-5 and BP-OSD, p = 0.003–0.005, loss 0.2–1% | up to 4.8× (relay-BP-5) and 6.7× (BP-OSD) lower LER at 1% loss | `sweep.py`, `plot_loss.py` |
| [`03_herald_control/`](03_herald_control) | Real vs blind vs shuffled flags on the same graph (seed 5); Pauli-injection check of the ½ prior | 8 vs 42 / 59 / 89 failed pairs | `sweep.py --loss-mode blind/shuffled`, `check_herald_prior.py` |
| [`04_robustness/`](04_robustness) | Prior value 0.2–0.5, missed and false flags (seed 12) | 3.2× with 20% false flags, 2.2× with 20% missed | `run_robustness.sh` |
| [`05_timing/`](05_timing) | One-worker timing: loss-aware vs unaware (seed 6) and the final baseline table (`final_timing/`, seed 16) | flags: 2.4–4.1× fewer errors, 35–52% less time | `sweep.py --workers 1` |
| [`06_router/`](06_router) | Learned routing relay-BP-1 → relay-BP-5 | +44 / +42 failed pairs per 10k vs the simple rule at p = 0.5% (fresh seeds 14 / 32), +30 with loss (seed 26) | `econ_router.py`, `run_overnight.sh`, `train_router.py` |
| ↳ `caches/` | No-loss relay-BP-1 / relay-BP-5 per-shot files: train (1, 2), thresholds (4), tests (10, 14); leg-cap runs (`relay1capK`) | | `sweep.py` |
| ↳ `preregistered_s14_s26/` | The pre-registered test: tables per p, with and without loss, plus run logs | | `run_overnight.sh` |
| ↳ `loss_router/` | 1% loss caches for the loss-aware router (seeds 22 / 24 / 26) | | `sweep.py --loss-mode reweight` |
| ↳ `fresh_s32/` | Sean's fresh seed-32 caches and scores (truth table and net-benefit router) | | `truth_table_router.py`, `econ_router.py` |
| ↳ `calibrated_p004/`, `replays/` | Earlier router work: calibrated logistic/boosted models, replay curves, pilots | | `train_router.py`, `relay_router_pilot.py` |
| [`07_truth_table/`](07_truth_table) | Frozen 6-bit truth-table routers (`table/`), RelayLab candidate and judge files | same accuracy as the simple rule at 27–40% less time | `truth_table_router.py` |
| [`08_gpu/`](08_gpu) | relay-BP-5 on an RTX 3060 via CUDA-Q QEC vs CPU, fp64 / fp32 / two lanes (seed 8) | fp32: ~10× lower mean and p99, accuracy within noise | `08_gpu/gpu_check.py` |
| [`figures/`](figures) | Slide figures (loss, router, GPU) | | matplotlib |

Seeds: 1/2 train · 4 thresholds · 5 herald control · 6 loss timing · 8 GPU · 10, 14, 32 fresh router tests ·
12 robustness · 16 final timing · 22/24/26 loss-router train/thresholds/test.
