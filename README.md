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
