# MidQWest 2026 – Team Guide (Infleqtion Challenge 1)

Snapshot of the working guide from Nov 21, 2026, 4:20 PM.

## Start here

We are building a two-stage decoder for the [[144,12,12]] Gross code. Fast BP decodes every shot, a small classifier sends hard shots to relay-BP, and atom-loss flags become decoder priors.

**Our claim:** relay-BP-5 accuracy at lower average latency, plus measurable gains from loss information. Every claim is made against the matching baseline (see Rules).

**The challenge in brief** ([Infleqtion repo](https://github.com/Infleqtion/MidQWest-2026)):

- Build an ML-assisted decoder that improves BP-OSD / relay-BP in accuracy, speed, or handling of qubit loss.
- Required sweep: physical error rate p = 0.003, 0.004, 0.005.
- Required metrics: logical error rate, decoding latency, decoder resources.
- Deliverables: decoder + data-generation + evaluation code on GitHub, at least one plot vs baselines, a short demo.
- Deadline: code freeze **2:00 PM Sunday**, demos 2–4 PM.

**How to pick up a task:** read Status, take an unowned row in the Task board, put your name in it, and post in the group chat. Ask Sean on anything QEC.

## Setup

You are set up when the kit tests pass and `baseline.py` prints a BP-OSD line. Python 3.12+ is required; relay-BP needs Rust.

```bash
# 1. Python env (uv downloads Python for you)
curl -LsSf https://astral.sh/uv/install.sh | sh      # then open a new terminal
mkdir -p ~/midqwest && cd ~/midqwest
uv venv --python 3.12 && source .venv/bin/activate

# 2. Team repo (kit + our code + docs)
git clone https://github.com/Gonzahart/MidQWest_2026
cd MidQWest_2026/challenge-1-ml-decoder-kit

# 3. Rust for relay-BP (skip on laptops if it fails; only compute machines need relay-BP)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# 4. Packages + tests
uv pip install -r requirements.txt pytest torch scikit-learn matplotlib
python -m pytest test_kit.py
python baseline.py --p 0.004 --shots 500
```

- On zsh, don't paste the `# ...` comments on the same line as a command; zsh treats them as arguments.
- Our added files: `loss_circuit.py` (heralded loss) and `evaluate.py` (decoder comparison). Both are in the team repo, with all context docs under docs/.
- Post your CPU core count and GPU (if any) in the group chat.

## Status

Loss simulation and the three-rate baseline sweep are complete. The per-shot export, router reporting harness, and two held-out Relay-BP routing replays are on main.

**Done**

- Environment on Gonzalo's Mac (Python 3.12, Stim, ldpc, relay-BP built).
- `loss_circuit.py`: at the start of every cycle each of the 144 data qubits is lost with probability `p_loss`, detected and replaced (Stim `HERALDED_ERASE`). Each loss flag is a `DETECTOR` with coordinate 2. Verified identical to the kit's problem at `p_loss = 0`. `lossy_dem_matrices(circuit, basis, use_heralds)` builds H, L and priors.
- `evaluate.py`: compares `bposd`, `relay`, the same with loss flags as extra rows of H (`+H`), and `bp>relay+H` (plain BP first, relay-BP only if BP doesn't converge). Reports logical error rate per cycle, mean and p99 ms, share escalated; appends to `results.csv`.
- Library checks: `relay_bp` has no `update_channel_probs`, so per-shot priors mean rebuilding the decoder (about 5 ms, fine). It does expose `decode_detailed` (iterations, success, posteriors) and `explicit_gammas`.

**First result** (only 200 shots, p = 0.004, 1% loss; treat as rough):

| Decoder | Logical error / cycle | Mean ms | p99 ms |
| --- | --- | --- | --- |
| bposd | 2.2e-2 | 392 | 893 |
| bposd+H | 5.1e-3 | 611 | 1438 |
| relay | 3.4e-3 | 833 | 6743 |
| relay+H | 2.1e-3 | 1091 | 8068 |
| bp>relay+H | 2.1e-3 | 1041 | 7814 |

Loss flags cut BP-OSD's error rate about 4×. Under 1% loss plain BP converges on only 27% of shots, so the router's savings depend strongly on the loss rate.

**Sena’s completed work**

- Added the per-shot NPZ export with failure flags, decode times, and separate first-stage and fallback iteration counts. [Evaluator](../challenge-1-ml-decoder-kit/evaluate.py)
- Built and smoke-tested the cached router comparison harness. It aligns shot IDs and produces a CSV and latency-versus-error plot. [Report harness](../challenge-1-ml-decoder-kit/router_report.py)
- Ran the 200-pair pilot and 2,000-pair held-out scale-up at p = 0.005 without loss. At a matched 20% fallback rate, the learned router was 7.8% faster on mean replay latency than the simple rule, with 152 vs. 144 failed pairs. Relay-BP-5 alone had fewer failures and lower latency than either router. [Replay results](../challenge-1-ml-decoder-kit/results/README.md#larger-held-out-relay-bp-1-routing-replay)

## Task board

Each row is owned by one person; put your name in the Owner cell when you take it.

| # | Task | Owner | Machine | Due | Done when | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Team GitHub repo with kit + `loss_circuit.py` + `evaluate.py`; per-shot results format (`.npz`: fail, seconds, iterations); README skeleton | Integration | laptop | 4:30 PM | Everyone can clone and run tests | Per-shot export done (Sena) |
| 2 | No-loss baseline sweep: BP-OSD, relay-BP-5 (kit default), relay-BP-1 (`stop_nconv=1`) at p = 0.003 / 0.004 / 0.005, 10k shots, per-shot results cached | Compute | 3060 box CPUs + BlueQubit | start now, done ~10 PM | p = 0.004 matches the kit README table within error | Done |
| 3 | Herald → error-column map + per-shot prior reweighting (fired ≈ 1/4, unfired 0) for BP-OSD (`update_channel_probs`) and relay-BP (rebuild per shot) | Sean (QEC lead) | laptop | 6 PM | Reweighted BP-OSD decodes a lossy shot | |
| 4 | Loss sweep: p_loss = 0, 0.002, 0.005, 0.01 × {loss-unaware, flags as H rows, reweighted priors} | Sean → Compute | 3060 box / BlueQubit | start ~7 PM | Cached results + one plot | |
| 5 | Stage-1 min-sum BP logging features at iterations 10/20/40/80 (unsatisfied checks, LLR sign flips, mean abs LLR, converged) + OSD-0 disagreement; 20k training shots at p = 0.005, seed 1 | Stage-1 BP | laptop | 10 PM | Features saved to `.npz` | |
| 6 | Router: labels (not converged by 80, or logically wrong); logistic regression then boosted trees; compare to "converged" rule, OSD-0 threshold, random routing | Router | laptop | 4 AM | AUROC on a held-out seed for all | Held-out Relay-BP routing replay done (Sena). Full task criteria remain open. |
| 7 | `HierarchicalDecoder.decode(syndrome)` plugging into `decoders.score`; end-to-end run with a dummy router; plot scripts | Integration | laptop | 10 PM | One results row + one plot | Comparison harness and smoke-test plot done (Sena). Live integration remains open. |
| 8 | Final timing: one machine, single-threaded, about 1–2k shots per point | Compute | 3060 box | after 8 AM | Mean + p99 table | Timing harness, 30-pair baseline smoke, and live-router adapter smoke done (Sena). Final 1,000–2,000-pair runs pending benchmark machine access. The timing handoff branch includes the live router and trained model. |
| 9 | Slides and talk: outline at 10 PM, fill after the 8 AM freeze, rehearse twice | Integration leads, all help | — | 1 PM Sun | Rehearsed | |
| S | Stretch (after 4 AM only): learned relay-BP memory strengths via `explicit_gammas`; learned loss-prior network; bb72 code | whoever is ahead | 3060 GPU / BlueQubit GPU | — | — | |
## Compute plan

Compute, not ideas, is our bottleneck: the kit's BP-OSD sweep at p = 0.005 alone took 58 minutes on 8 cores, and relay-BP is slower.

| Machine | Use it for | Notes |
| --- | --- | --- |
| 3060 box (SSH) | Always-on baseline and loss sweeps on its CPU cores; final single-threaded timing | Run everything in `tmux` so jobs survive SSH drops. GPU only for stretch goals |
| BlueQubit cloud (each member has $30) | Extra sweep boxes: a LARGE CPU notebook ($1.5/hour, about 20 hours per $30). GPU A100 notebooks ($4/hour) only for stretch training | Launch from app.bluequbit.io → Hybrid. Run `nproc` first; if the LARGE box has few cores it isn't worth it. The hackathon page says LARGE machines have a 6+ hour limit, so check the usage page |
| Laptops | Development, router training, small runs | Idle laptops can take a seed range of an accuracy sweep; error counts merge afterwards |

- Accuracy sweeps can be split across machines by seed; logical error counts add up.
- Timing numbers must all come from **one machine, one run, single-threaded**.
- BlueQubit's simulators (statevector, MPS, Pauli-path) don't help here; Stim is far faster for QEC sampling. We use BlueQubit only as rented computers.

## Rules for results

A claim counts only next to its matching baseline.

| Claim | Compare against |
| --- | --- |
| Accuracy | BP-OSD (kit settings), relay-BP-1, relay-BP-5 |
| Speed | Relay-BP-5 (relay-BP-1 already stops early on easy shots) |
| Router is useful | "Accept if BP converged" rule; OSD-0-disagreement threshold; random routing at the same escalation rate |
| Loss helps | Loss-unaware decoder **and** naive loss-aware priors; ML gets credit only for the gap over naive priors |

- Same shots and seeds for every decoder; train seed ≠ test seed.
- Train the router at p = 0.005, test at all three noise levels.
- A shot fails if any of the 12 logical observables is wrong. Report X, Z and pairs with `decoders.per_cycle_rate`, with error bars.
- Cache per-shot results; never re-run a sweep to make a plot.
- State the loss model in the talk: a lost atom is detected and replaced every cycle, and only data qubits are lost.
- Quote numbers from papers only if someone opened the paper; say "reported" for preprints.

## Gates

Each gate is a 10-minute stand-up: every owner says done, blocked or cut, and Sean decides.

1. **4 PM Sat** — setup passes everywhere; baseline sweep started.
2. **10 PM Sat (CP1)** — baselines match the kit table; stage-1 features saved; loss circuit decodes with reweighted priors.
3. **4 AM Sun (CP2)** — router beats the "converged" rule on a held-out seed. If not, the pitch becomes early exit plus loss-awareness (decided now, not at 4 AM).
4. **8 AM Sun** — feature freeze: only bug fixes, plots and slides after this.
5. **2 PM Sun** — repo pushed, talk rehearsed twice, submitted.

## Key context

Enough vocabulary to follow the stand-ups; Sean's 30-minute primer covers the rest.

| Term | Meaning here |
| --- | --- |
| Gross code | The [[144,12,12]] bivariate bicycle qLDPC code: 12 logical qubits in 144 data qubits, distance 12 |
| Detector / syndrome | Parity checks that flip when an error happens; the decoder's input (936 bits per basis in the kit) |
| H e = s | H marks which detectors each error mechanism flips; the decoder finds an error e that explains the syndrome s |
| Logical failure | The predicted correction gets any of the 12 logical observables wrong |
| BP / min-sum | Fast iterative message passing on the Tanner graph; may not converge |
| BP-OSD | BP plus an expensive ordered-statistics cleanup when BP fails |
| Relay-BP-S | Chains of BP runs with memory; stops after S solutions. Kit default S = 5 |
| Herald / erasure | A loss flag: we know *where* a qubit was lost, not what error it left |

**Reading list** (in order of usefulness):

- [Infleqtion challenge README](reference/INFLEQTION_CHALLENGE_README.md) and the kit README
- Relay-BP paper, [arXiv:2506.01779](https://arxiv.org/abs/2506.01779)
- Sean's research report [ML-decoders-for-Gross-code.pdf](reference/ML-decoders-for-Gross-code.pdf) (why a router, not a neural decoder)
- Sean's handoff [MidQWest-2026-Team-Handoff.pdf](reference/MidQWest-2026-Team-Handoff.pdf)
- Stim gate reference for [`HERALDED_ERASE`](https://github.com/quantumlib/Stim/blob/main/doc/gates.md)
- MIT transformer decoder on BB codes, [arXiv:2504.13043](https://arxiv.org/abs/2504.13043) (why we don't build an end-to-end neural decoder)
- GNN failure prediction on BB codes, [arXiv:2610.00504](https://arxiv.org/abs/2610.00504) (evidence a learned router works on this code)
- Our notes: [decoder literature](context/decoder-literature.md), [challenge plan](context/challenge1-plan.md), [Infleqtion insights](context/infleqtion-insights.md), [track analysis](context/track-analysis-and-prep.md)
