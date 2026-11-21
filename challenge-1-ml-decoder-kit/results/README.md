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

## Reweighted priors, first check (cloud test box, 300 shots, p = 0.004, p_loss = 0.005)
BP-OSD: unaware 1.15e-2 · rows 7.5e-3 (555 ms) · **reweight 5.7e-3 (415 ms)**. Larger run pending.
