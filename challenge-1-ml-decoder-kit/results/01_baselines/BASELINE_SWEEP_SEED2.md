# No-loss baseline sweep, seed 2: results and handoff

Sat Nov 21, 2026, 5:40 PM · Sean · task 2 on the board (no-loss baseline sweep)

**Short version:** the baselines match the kit (one point needs a recheck), so CP1's baseline gate passes. "Plain BP first" saves time against relay-BP-5 only at low noise and never against relay-BP-1. The router worth building sits between relay-BP-1 and relay-BP-5. A perfect version of it beats relay-BP-5 on accuracy and is 1.4–4.4× faster.

## Files in this folder

| File | What it is | Open with |
|---|---|---|
| `baseline_sweep_seed2.xlsx` | Full workbook: **Read me** (settings, definitions, caveats), **Baselines** (4 decoders × 3 p, compared to the kit README), **Routing** (two-stage policies replayed from cached shots). Error rates and error bars are live formulas | Excel, LibreOffice, Google Sheets |
| `baseline_sweep_seed2_baselines.csv` | The Baselines tab as plain values | GitHub (renders as a table) |
| `baseline_sweep_seed2_routing.csv` | The Routing tab as plain values | GitHub |

The per-shot `.npz` files are on Sean's PC (`results/<decoder>_p<p>_seed2.npz`) and are not committed, because they don't use the team `sweep.py` format (no `shot_idx`, `host`, `stim_version`). See "Next steps" for how to get shots `router_report.py` can read.

## What was run

| Setting | Value |
|---|---|
| Code, experiment | [[144,12,12]] Gross code, X and Z memory, 12 noisy cycles, **no loss** |
| Noise | p = 0.003, 0.004, 0.005 |
| Shots | 10,000 per basis; shot *k* of X paired with shot *k* of Z |
| Seed | **2** (X sampled with seed 2, Z with seed 3). Router training must not use seed 2 |
| Decoders | `bposd` (kit settings), `relay5` (kit relay-BP, `stop_nconv=5`), `relay1` (`stop_nconv=1`), `bp` (plain min-sum, 80 iterations, the BP stage of BP-OSD) |
| Machine | Sean's home PC, Windows, Python 3.12, 8 parallel worker processes |
| Scripts | Sean's local `sweep.py` (decodes and caches per-shot `failed`, `seconds`, `converged`) and `hybrid.py` (replays two-stage policies from the cache). Not the team `sweep.py`; not in the repo yet |

A pair fails if either basis gets any of the 12 logical observables wrong. LER per cycle $= 1-(1-P)^{1/12}$ with $P$ = failed pairs / 10,000, as in `decoders.per_cycle_rate`. Times are one X decode plus one Z decode.

## Result 1: baselines match the kit (sheet: Baselines)

| p | Decoder | Failures X, Z | LER / cycle | Mean ms | p99 ms | Kit README failures X, Z | Match |
|---|---|---|---|---|---|---|---|
| 0.003 | BP-OSD | 40, 48 | 7.4e-4 ± 0.8e-4 | 89 | 536 | 26, 30 | **check** |
| 0.003 | Relay-BP-1 | 7, 10 | 1.4e-4 ± 0.3e-4 | 14 | 127 | — | |
| 0.003 | Relay-BP-5 | 9, 4 | 1.1e-4 ± 0.3e-4 | 70 | 506 | 3, 2 | yes |
| 0.004 | BP-OSD | 253, 279 | 4.5e-3 ± 0.2e-3 | 238 | 776 | 253, 269 | yes |
| 0.004 | Relay-BP-1 | 60, 60 | 1.01e-3 ± 0.09e-3 | 69 | 1307 | — | |
| 0.004 | Relay-BP-5 | 42, 54 | 8.0e-4 ± 0.8e-4 | 365 | 6431 | 45, 60 | yes |
| 0.005 | BP-OSD | 1304, 1302 | 2.30e-2 ± 0.05e-2 | 389 | 744 | 1302, 1286 | yes |
| 0.005 | Relay-BP-1 | 512, 486 | 8.4e-3 ± 0.3e-3 | 472 | 4547 | — | |
| 0.005 | Relay-BP-5 | 434, 375 | 6.8e-3 ± 0.2e-3 | 1136 | 5774 | 411, 414 | yes |

Plain BP alone fails 21% / 54% / 86% of pairs at p = 0.003 / 0.004 / 0.005. That's expected: it gives up when it doesn't converge.

**Open issue: BP-OSD at p = 0.003.** 88 failed pairs where the kit's 100k-shot run predicts about 56, roughly 4σ high. Every other point matches. Rerun this one point with another seed before CP1 to rule out a fluke or a Windows `ldpc` build difference.

## Result 2: speed-up from "plain BP first" (sheet: Routing)

"Plain BP → X": accept plain BP's answer when it converges, otherwise run X.

| p | BP accepts | Plain BP → relay-BP-5 vs relay-BP-5 | Plain BP → relay-BP-1 vs relay-BP-1 |
|---|---|---|---|
| 0.003 | 85% of shots | 48 vs 70 ms, **1.46× faster** | 23 vs 14 ms, slower |
| 0.004 | 62% | 329 vs 365 ms, 1.11× faster | 85 vs 69 ms, slower |
| 0.005 | 32% | 1136 vs 1136 ms, no gain | 498 vs 472 ms, slower |

- **Converged BP was never logically wrong**: 0 of about 36,000 converged shots across all three p. So accepting converged BP costs no accuracy, and the "accept if BP converged" rule is already lossless.
- **Relay-BP-1 is faster than the plain-BP stage itself** (14 vs 15 ms at p = 0.003). Relay-BP is Rust; `ldpc` BP is not. So a BP-first stage can never beat relay-BP-1 on speed.
- Per the Rules table, speed claims are judged against relay-BP-5 with relay-BP-1 next to it. Relay-BP-1 already gives 2.4–5.3× over relay-BP-5 with no ML, so **"faster than relay-BP-5" is not an ML result on its own.**

## Result 3: headroom for a relay-BP-1 → relay-BP-5 router (sheet: Routing, yellow rows)

Policy: run relay-BP-1 on every shot; when the router doesn't trust the answer, rerun relay-BP-5 from scratch. The **perfect router** escalates exactly the basis shots relay-BP-1 got wrong. It's an upper bound, not something we can claim.

| p | Relay-BP-1 LER | Relay-BP-5 LER | Perfect router LER | Perfect router mean ms | Speed-up vs relay-BP-5 | Basis shots escalated |
|---|---|---|---|---|---|---|
| 0.003 | 1.4e-4 | 1.1e-4 | 5.0e-5 | 16 | 4.4× | 17 of 20,000 (0.1%) |
| 0.004 | 1.01e-3 | 8.0e-4 | 5.2e-4 | 110 | 3.3× | 120 of 20,000 (0.6%) |
| 0.005 | 8.4e-3 | 6.8e-3 | 5.3e-3 | 806 | 1.4× | 998 of 20,000 (5%) |

- The perfect router beats relay-BP-5 on accuracy because **relay-BP-5 is not a superset of relay-BP-1**. It gets some shots wrong that relay-BP-1 gets right (34 basis shots at p = 0.004), so combining them helps.
- **Relay-BP-1's mistakes are almost all "found a solution, but the wrong one"**: 15 of 17 at p = 0.003, 90 of 120 at p = 0.004. "Did it converge?" can't catch them. The router needs features of the solution itself: its weight or likelihood under the priors, iterations used, which leg converged (`relay_bp` `decode_detailed`).
- The target is narrow: escalate under ~1% of shots at p = 0.003–0.004 and still catch most of relay-BP-1's errors. A router that escalates 5–10% is still worth reporting if it beats random routing at the same rate.

## What this changes

1. **Router (task 6): build relay-BP-1 → router → relay-BP-5**, not BP → router → relay-BP. Labels: relay-BP-1 logically wrong on that basis shot. Features: relay-BP-1 output (solution weight / log-likelihood, iterations, leg), syndrome weight, optionally plain-BP features. Sena's `router_report.py` already compares relay-BP-1, relay-BP-5 and random routing, so plug into it.
2. **Stage-1 BP features (task 5):** plain-BP iteration features are still useful router inputs, but plain BP is no longer the first stage of the decoder.
3. **CP2 gate (4 AM):** "router beats the converged rule" is too easy, since that rule is already lossless. Proposed bar: on a held-out seed, **lower LER than relay-BP-1 and lower mean time than relay-BP-5, and better than random routing at the same escalation rate.**
4. **Pitch:** "relay-BP-5-level accuracy (or better) at close to relay-BP-1 cost", plus loss-awareness. The fallback pitch (early exit + loss) is already backed by Result 2 at p = 0.003.

## Next steps

| What | Who | Notes |
|---|---|---|
| Recheck BP-OSD at p = 0.003 with `--seed 3` | Compute | One point, ~5 min |
| Router training data at p = 0.005, **seed 1**, with relay-BP-1 `decode_detailed` output per shot | Router / Stage-1 | Never seed 2 (this test set) |
| Rerun relay-BP-1 and relay-BP-5 at seed 2 through the team `sweep.py` (`--seed 2`) so `router_report.py` can read them, or convert these caches | Compute / Sean | ~1 h for both decoders at all p on 8 cores |
| Fix loss reweighting in team `sweep.py`: fired-herald columns get prior **0.5**, not 0.25 (Z and Y merge into one column in X memory, so the column fires with probability 1/2 given a loss) | Sean (task 3) | One line in `Reweighted.priors_for` |
| Change the team `sweep.py` default `--seed` from 1 to a test seed | Integration | Seed 1 collides with router training data |
| Final single-threaded timing (task 8) | Compute | These timings had 8 parallel workers: ratios are fair, absolute ms are not final |

## Caveats

- Timings come from 8 parallel workers on one machine; quote ratios, not absolute latency, until task 8.
- Relay-BP at p = 0.003 has fewer than 20 failures per decoder, so its error bars are wide.
- "Perfect router" rows are an upper bound computed from known labels. Never present them as a decoder result.
- Replayed policies add cached decode times and leave out router inference time. Relay-BP results can depend on call history, so final router claims need a live run (see the kit README section "Compare routing policies from cached shots").
