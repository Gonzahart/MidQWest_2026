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
