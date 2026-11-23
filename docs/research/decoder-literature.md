# ML decoders for qLDPC / Gross code – literature notes (Sat Nov 21, 3 PM)

> **Snapshot (Sat Nov 21, 3 PM CT)**, written right after the challenge was revealed. Final numbers: [root README](../../README.md).


See also Sean's fuller report: [ML-decoders-for-Gross-code.pdf](ML-decoders-for-Gross-code.pdf).

| Approach | Key paper | What it shows | Fit for us (24h, laptop) |
|---|---|---|---|
| End-to-end transformer decoder | Blue, Avlani, He, Ziyin, Chuang (MIT), arXiv 2504.13043, Quantum 2026 | ~5x lower LER than BP-OSD on [[72,12,6]]; on [[144,12,12]] (our code) *worse* LER, but steadier & faster runtime | Low – the experts didn't beat baselines on our exact code |
| GNN learned message passing (Astra) | Maan & Paler, arXiv 2408.07038, npj QI 2025; code github.com/arshpreetmaan/astra | Beats BP-OSD (code capacity); Astra+OSD calls OSD less often → faster | Medium – open code, but circuit-level Gross code training is heavy |
| Neural BP (learned weights) | Liu & Poulin 1811.07835; Miao et al. 2212.10245 | Few trainable weights per iteration/check improve BP convergence | High – small, CPU-trainable, drops into our stage 1 |
| Neural failure prediction / post-selection | Carty, Noszko, Roffe, Bondesan, arXiv 2610.00504 | GNN on syndromes + BP posteriors predicts decoder failure; on [[144,12,12]] keeping 90% of shots cuts LER ~740x | High – this is exactly the confidence router |
| AI predecoder | NVIDIA Ising Decoding (Chamberland et al. 2604.12841) | 3D CNN predecoder + global decoder; surface/color codes | Inspiration; CNN locality doesn't transfer to qLDPC |
| Loss-aware decoding | Baranes et al. (delayed erasure) 2502.20558; Perrin et al. 2412.07841; "High-rate qLDPC processors" 2607.28795; mixture decoding for correlated loss 2610.38283 | Use loss locations: rebuild decoding graph per shot or reweight priors ("Pauli envelope"); qLDPC + realistic loss flagged as open problem | High – our heralds-as-detectors = erasure decoding; per-shot prior reweighting is a faster variant |

## Recommended design
Stage 1: loss-aware BP (per-shot loss-reweighted priors, optionally learned BP weights) → Router: small model on syndrome + BP features (+ #heralds) predicts "trust stage 1?" → Stage 2: relay-BP (or BP-OSD).
Report: LER/cycle, mean & p99 latency, % escalated, model params/inference time, across p = 0.003/0.004/0.005 and a p_loss sweep.

## BlueQubit
Simulators (statevector/MPS/Pauli-path) are not useful for QEC sampling (Stim is far faster). Use: Hybrid notebooks as extra compute — LARGE CPU ($1.5/h) for relay-BP/BP-OSD sweeps; GPU A100 ($4/h) or H100 ($8/h) notebooks exist for stretch training. Each team member has $30 in credits.
