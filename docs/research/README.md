# Research

These are the research reports and notes we wrote or commissioned during the hackathon. Most are snapshots from before the final results. For the numbers we stand behind, see the [root README](../../README.md).

| File | What it is | By |
|---|---|---|
| [`prior-work-and-novelty.md`](prior-work-and-novelty.md) | The closest prior work (Murri 2026, Baranes et al. 2026), how our loss-aware decoding differs from it, and the novelty claim we made | Team |
| [`ML-decoders-for-Gross-code.pdf`](ML-decoders-for-Gross-code.pdf) | Why we built a router around BP-family decoders rather than a neural decoder: the hardware and data that published neural decoders needed | Sean Guirola (AI-assisted) |
| [`decoder-literature.md`](decoder-literature.md) | Literature notes on ML decoders for qLDPC codes, written right after the challenge was revealed | Team (AI-assisted) |
| [`decoder-upgrade-ideas.md`](decoder-upgrade-ideas.md) | Ranked list of decoder upgrades that fit a 4–6 hour window (utility-aware router, gamma schedules, posterior-guided fallback) | AI-assisted review |
| [`ml-ideas-gross-code-decoding.md`](ml-ideas-gross-code-decoding.md) | Overnight review plus a 400-shot local benchmark of capped relay-BP with an OSD or LSD fallback | AI-assisted review |
| [`ml-ideas-benchmark/`](ml-ideas-benchmark/) | Scripts and raw outputs for that benchmark (`python docs/research/ml-ideas-benchmark/bp_postprocessing_bench_full.py p shots n_bposd num_sets`) | — |

Other people's papers are linked, not copied. The key ones are listed in [`prior-work-and-novelty.md`](prior-work-and-novelty.md#background-papers).
