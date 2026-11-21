# Infleqtion Challenge 1 – team plan & status (Sat 3:15 PM)

The live task board is in [../TEAM_GUIDE.md](../TEAM_GUIDE.md) / the live doc. This is the earlier working plan.

Team plan = Sean's handoff (two-stage decoder: fast BP → router → relay-BP; loss flags → BP priors). Claim: relay-BP-5 accuracy at lower mean latency + gains from loss info, each vs the right baseline.

## Gates (clock time)
4 PM setup + sweep started · 10 PM CP1 · 4 AM CP2 (fallback pitch: early exit + loss-awareness) · 8 AM feature freeze · 2 PM Sun submit.

## Done
- Env on Gonzalo's Mac (Py 3.12, stim, ldpc, relay-bp built with Rust).
- `loss_circuit.py`: HERALDED_ERASE + flag DETECTOR (coord 2) on data qubits each cycle; verified identical to kit at p_loss=0; `lossy_dem_matrices(..., use_heralds)`.
- `evaluate.py`: bposd, relay, +H (heralds as H rows), bp>relay+H; LER, mean/p99 ms, % escalated, results.csv.
- First result (200 shots, p=0.004, p_loss=0.01): heralds cut BP-OSD LER ~4x; plain BP converges on only 27% of shots under 1% loss.
- Checked: relay_bp has no update_channel_probs → rebuild decoder per shot (~5 ms, fine); has decode_detailed (iterations, success, posteriors) and explicit_gammas.

## Implementation notes
- Stim fuses consecutive `DETECTOR` instructions with equal arguments when built with `Circuit.append`, so `loss_circuit.add_loss` builds the circuit as text.
- Circuits with `HERALDED_ERASE` need `detector_error_model(approximate_disjoint_errors=True)`; use `lossy_dem_matrices`, not the kit's `dem_matrices`.
- A flattened circuit gives a DEM with a different line count than the REPEAT version, but the same merged mechanisms (checked column by column).

## Rules
Same shots/seeds; train seed ≠ test seed; router trained at p=0.005, tested at all three; report X, Z and pairs with per_cycle_rate + error bars; cache everything; timing single-threaded on one machine.
