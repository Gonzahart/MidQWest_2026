# MidQWest 2026 — Infleqtion Challenge 1: ML-assisted decoding of the Gross code

Team repo for the MidQWest 2026 hackathon (UIC, Nov 21–22). We are building a two-stage decoder for the
[[144,12,12]] Gross code: fast BP decodes every shot, a small classifier sends hard shots to relay-BP, and
atom-loss flags become decoder priors.

**Start with the team guide:** [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md). It has setup, status, the task board, compute plan,
rules for results and gate times.

## Layout

| Path | What it is |
|---|---|
| `challenge-1-ml-decoder-kit/` | Infleqtion's decoder kit (baselines, circuits) plus our code |
| `challenge-1-ml-decoder-kit/loss_circuit.py` | Adds heralded atom loss (Stim `HERALDED_ERASE` + flag detectors) to the Gross code circuits |
| `challenge-1-ml-decoder-kit/evaluate.py` | Compares decoders with and without loss flags, and a BP → relay-BP two-stage decoder |
| `docs/TEAM_GUIDE.md` | Team guide (snapshot of the live doc) |
| `docs/context/` | Planning and research notes: track analysis, Infleqtion insights, challenge plan, decoder literature |
| `docs/reference/` | Challenge README, Sean's handoff and research report, Fowler's surface code paper |
| `prep/` | Friday-night prep: surface code threshold baseline with Stim + PyMatching |

## Quick start

```bash
mkdir -p ~/midqwest && cd ~/midqwest
uv venv --python 3.12 && source .venv/bin/activate
git clone https://github.com/Gonzahart/MidQWest_2026
cd MidQWest_2026/challenge-1-ml-decoder-kit
uv pip install -r requirements.txt pytest torch scikit-learn matplotlib   # relay-bp needs Rust: rustup.rs
python -m pytest test_kit.py
python baseline.py --p 0.004 --shots 500
python evaluate.py --p 0.004 --p-loss 0.005 --shots 200
```

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
