# Review brief: MidQWest 2026, Infleqtion Challenge 1 (for an outside reviewer)

Sat Nov 21, 2026, 9 PM CT. Code freeze Sun 2 PM CT. Repo: https://github.com/Gonzahart/MidQWest_2026

**What we want from you:** concrete, testable suggestions that could improve our results before the freeze
(~12 hours left, laptops + one 20-core cloud CPU box + one RTX 3060). Please rank them by expected gain per hour
of work, and flag anything in our method a QEC expert judge would attack.

## The challenge (Infleqtion, "Qubits & Gradients", Challenge 1)

Build an ML-assisted decoder for the [[144,12,12]] Gross code (bivariate bicycle qLDPC) that improves on BP-OSD
and relay-BP in accuracy, speed or handling of qubit loss. Required: sweep p = 0.003 / 0.004 / 0.005; report
logical error rate (per cycle), decoding latency and resources; code + data generation + evaluation on GitHub;
at least one plot; short demo. Full text: `docs/reference/INFLEQTION_CHALLENGE_README.md`. Kit:
`challenge-1-ml-decoder-kit/` (Stim circuits for X and Z memory, 12 noisy cycles, circuit-level noise; baselines
BP-OSD with OSD-CS order 10, relay-BP with the relay-BP paper's settings, stop_nconv = 5).

## What we built

| File | What it does |
|---|---|
| `loss_circuit.py` | Adds heralded atom loss: at each cycle start every data qubit is erased with prob p_loss (Stim `HERALDED_ERASE`), herald = extra detector (coord 2). Verified identical to the kit at p_loss = 0 |
| `sweep.py` | Shardable sweeps with per-shot results (.npz): fail[X,Z], seconds, iterations, relay router features. `--loss-mode unaware / rows / reweight` |
| `sweep.py` `Reweighted` | **Loss-reweighted priors:** per shot, columns caused by a fired herald get prior P(column \| loss) (= 1/2), columns of unfired heralds ~0, others unchanged; decode on the check rows only (H keeps kit size). BP-OSD via `update_channel_probs`, relay-BP rebuilt per shot (~5 ms) |
| `train_router.py` | Relay-BP-1 -> router -> relay-BP-5, replayed from cached decodes. Logistic regression + gradient-boosted trees on relay-BP-1 output features (converged, solution weight, log-likelihood under priors, syndrome weight, posterior \|LLR\| mean and p10, iterations). Options: `--calibrate` (thresholds from a separate seed at the test p), `--train-relay5` (rescue labels), random routing matched to the actual escalation count |
| `relay_router_pilot.py`, `router_report.py`, `benchmark_timing.py`, `decoders.HierarchicalDecoder` | Teammates' live router (feature extraction + inference inside decode) and single-worker timing/memory harness |
| `plot_loss.py` | Loss-sweep figure |

Seeds: 1 = router training, 2 = final test, 3 = threshold calibration, 4 = validation. Stim draws different shots
from the same seed on different CPUs, so paired comparisons always use files from one machine (`host` field).

## Results so far (details: `challenge-1-ml-decoder-kit/results/README.md`, `results/*.csv`)

**Baselines reproduce the kit** (10k pairs): relay-BP-5 at p = 0.004: 8.0e-4 per cycle (kit 8.7e-4 +- 0.9e-4);
BP-OSD at p = 0.003: 4.8e-4 (kit 4.7e-4).

**Result 1, loss-reweighted priors (our headline).** Relay-BP-5, p = 0.004, 2k pairs/point, seed 2:

| Loss rate | Ignored | Reweighted | Gain |
|---|---|---|---|
| 0.2% | 9.2e-4 | 5.9e-4 | 1.6x |
| 0.5% | 1.81e-3 | 1.05e-3 | 1.7x |
| 1.0% | 3.92e-3 | 1.51e-3 | 2.6x |

p = 0.005: 1.25x / 1.75x / 2.2x. Independent PC run at 0.5% loss: 2.6e-3 -> 9.6e-4, paired sign test 47 vs 8 shots,
p = 8e-8. "Flags as extra rows of H" helped BP-OSD 1.4x, relay-BP not at all, and doubled decode time.
Figure: `results/loss_sweep_relay5.png`.

**Result 2, router (partial).** Trained on 20k pairs at p = 0.005 (~2,000 relay-BP-1 errors), tested on 10k
seed-2 pairs at p = 0.004 (replay times, 10 parallel workers, exclude inference):

| Policy | Escalated | Failed pairs / 10k | Mean ms/pair |
|---|---|---|---|
| Relay-BP-1 | 0% | 147 | 156 |
| Relay-BP-5 | - | 96 | 505 |
| Logistic router | 0.8% | 129 | 256 |
| Random, same count | ~1% | ~147 | ~160 |
| Perfect router (bound) | 0.76% | 69 | 225 |

AUROC 0.94-0.97, but a hand rule (non-converged first, then iterations) ranks almost as well. Thresholds from
p = 0.005 under-escalate at lower p (fixed with `--calibrate`). Many relay-BP-1 failures are also relay-BP-5
failures (rescue labels being added). Even a perfect router can't beat 225 ms at p = 0.004 because escalated
shots pay for both decoders. Earlier replay at a fixed 20% escalation lost to relay-BP-5 on both axes.

## Known limitations

- Loss model: lost atom detected and refilled every cycle, data qubits only, left maximally mixed. Real loss
  persists across gates and also hits ancillas.
- p = 0.003 has few failures (19 relay-BP-1 pairs in 10k); larger runs are in progress.
- All latency so far is from parallel workers; final single-worker timing is pending (Linux/WSL, one idle machine).
- Relay-BP is stateful across calls, so replayed and live results can differ slightly.

## Questions we'd most like answered

1. Cheap ways to make the router escalate the *right* shots (relay-BP-1 wrong AND relay-BP-5 right): better
   features from relay_bp's `decode_detailed` (posterior_ratios, iterations, success)? Escalate to more relay legs
   instead of a full relay-BP-5 restart?
2. Is there a faster second stage than relay-BP-5 for escalated shots (e.g. warm-starting relay from relay-BP-1's
   posteriors, or OSD-0 on relay-BP-1's posteriors)?
3. Can an ML component beat our analytic loss reweighting (the bar any learned loss model must clear)? E.g.
   learned per-shot prior correction, or learned relay memory strengths (`explicit_gammas`)?
4. Anything wrong or attackable in the loss model, the reweighting (P(column | loss) = 1/2), or the evaluation?
