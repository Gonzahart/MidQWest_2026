# Infleqtion track – insights (Sat Nov 21, 11 AM, before the challenge was revealed)

## What Infleqtion is working on (= what judges will care about)
- **AI-accelerated QEC:** integration with NVIDIA Ising Decoding – a GPU AI *predecoder* that cleans the syndrome stream before a standard decoder. Metrics: logical error rate first, then latency/throughput (real-time loop; ~60 µs readout target).
- **Neutral-atom-specific noise:** leakage (looks like |0>/|1> at readout) and atom loss (detectable with a 2nd measurement → ternary readout). Detected loss/leakage = **erasure** (you know *where*, not *what*). Their pipeline: flag loss/leakage → feed as erasure info → train predecoder on it → simpler problem downstream. Tool: open-source Leaky / `leakysim`.
- **qLDPC codes:** HGPS [[98,18,4]] code with NVIDIA CUDA-Q; resource-superstaq (UChicago, Fred Chong).
- **Contextual ML (CML):** quantum-inspired ML that widens context for long sequences, runs on GPUs. Won Army xTechScalable AI (SAPIENT: multimodal sensor fusion, Boltzmann-machine-based, spoofing detection).

## Past hackathons
Little public record of Infleqtion student-hackathon winners. Big Q Hackathon (Chicago 2023): winning technical team used Infleqtion's platform on an industry use case.

## Angles likely to stand out
1. Erasure/loss-aware decoding, compared against a loss-unaware decoder and a simple loss-aware baseline.
2. Report accuracy *and* latency against the right baselines.
3. Demo: one clear plot, honest numbers, tie back to neutral-atom hardware (Sqale).
