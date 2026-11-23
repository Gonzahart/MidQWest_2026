# Fast, Actionable ML-Assisted Decoder Upgrades for the Gross [[144,12,12]] Code

> **Snapshot (Nov 22, 2026).** AI-assisted literature review commissioned by the team mid-hackathon; it shaped the net-benefit router and the gamma/fallback experiments. Expected gains here are hypotheses from the literature, not our measurements. Final numbers: [root README](../../README.md).


**MidQWest 2026 — Infleqtion Challenge 1**  
**Research date:** 2026-11-22  
**Setting:** 12-cycle Stim circuit-level X/Z memory, p=0.003/0.004/0.005, 936×8,784 detector-error-model check matrix per basis. CPU-first, 20 cores, RTX 3060 available. Baselines: ldpc BP-OSD (OSD-CS-10) and Relay-BP-1/5.

**Evidence standard:** Published findings are distinguished from speculative performance forecasts. All expected gains are hypotheses to test, not measured improvements.

## Executive ranking

The highest-value work in a 4–6 hour window is **not** to train a new neural decoder. The literature favors exploiting Relay-BP memory-strength tunability, using its posteriors, and optimizing ML routing for *incremental benefit per additional millisecond* rather than fast-decoder failure.

| Rank | Idea | Feasibility | Expected value | Main risk |
|---|---|---|---|---|
| 1 | Utility-aware router: predict incremental value of Relay-BP-5 | Excellent | Medium–high | Gain may exist only at p=.005 |
| 2 | Tune/learn small portfolio of Relay-BP gamma schedules | Excellent | Medium–high | Noisy sweeps, limited complementarity |
| 3 | Posterior-guided cheap fallback: OSD-0 or localized solve | Good | Medium–high | Integration and latency |
| 4 | Residual logical classifier on Relay-BP outputs | Excellent | Medium | Weak signal |
| 5 | Calibrated loss priors from flags, cycle and syndrome | Excellent | Low–medium extra gain, high narrative value | 1/2 prior may already be optimal |
| 6 | Ambiguity/LSD confidence as router feature | Fair | Medium | New integration |
| 7 | Full transformer/GNN/diffusion decoder | Poor | Long-term | Training beyond deadline |

The directly relevant circuit-level BB transformer study used a 19M-parameter model and approximately 3.8×10^8 training examples for [[144,12,12]]; its initial accuracy was roughly comparable to BP-OSD-3 in p≈.003–.007, not a decisive improvement over Relay-BP. [Blue et al., arXiv:2504.13043](https://arxiv.org/abs/2504.13043). Astra is promising but its reported BB results are code-capacity, not this circuit-level DEM task. [arXiv:2408.07038](https://arxiv.org/abs/2408.07038), [code](https://github.com/arshpreetmaan/astra).

## 1. Utility-aware router (highest priority)

**Research basis:** Adaptive decoder selection work emphasizes that a policy optimizing failure prediction can fail to improve the error/latency frontier when expensive shots are precisely those selected for escalation. [Adaptive decoder selection for quantum LDPC codes is effectively one-dimensional (Zenodo, 2026)](https://zenodo.org/records/21466148). Relay-BP's structure and latency behavior are described in [Müller et al., arXiv:2506.01779](https://arxiv.org/abs/2506.01779).

**Idea:** AUROC≈.95 for “BP-1 fails” does not imply benefit over a compute-matched random BP-1/BP-5 mixture. Relabel paired samples as incremental success: y=1 when BP-1 is wrong and BP-5 is correct; penalize reverse flips. Train a second predictor for extra BP-5 wall time. Rank escalation by estimated logical-error improvement / extra milliseconds or directly optimize empirical L = failure + λ·latency. Optimize p99 separately; mean-cost optimization alone may worsen tails.

**Implementation:** Reuse existing paired-shot table; add bp1_correct, bp5_correct, bp5_time, bp1_time, delta_correct = bp1_fail - bp5_fail. Fit HistGradientBoostingClassifier and cost regression on the existing BP-1 features (convergence, weight, log-likelihood, syndrome weight, posterior confidence quantiles, iterations). Threshold score; plot LER vs *measured* mean latency and LER vs p99. Optionally estimate conditional expected benefit via separate classifiers for BP-1→BP-5 rescue and harm.

**Compute/time:** 30–90 minutes CPU once paired labels exist. **Expected gain:** 5–15% fewer failures at matched mean time for p=.005 is *speculative*, likely zero or smaller at p=.004. **Risk:** medium; if salvageable shots are universally slow, the achievable frontier may match random mixing. **Fair comparison:** paired same-shot outcomes; random escalation at equal escalation fraction *and* separately at equal mean wall time; fresh seeds; bootstrap paired error deltas. Plot oracle escalation upper bound.

## 2. Small portfolio of optimized Relay-BP gamma schedules

**Paper:** [Müller et al., “Improved belief propagation is sufficient for real-time decoding of quantum memory,” arXiv:2506.01779](https://arxiv.org/abs/2506.01779). [Rust/Python code](https://github.com/zphy/relay_bp).

**Idea:** Relay-BP performance is sensitive to memory strengths. For Gross, the paper reports a strong gamma-dependent performance landscape; a published choice used a first-leg gamma near 0.125 and later strengths roughly over [-0.24, 0.66]. Authors used extensive sweeps and identify better memory-strength choice as open. Relaying outperformed independent restarts in a reported comparison (approximately 7×10^-6 versus 1.4×10^-5 logical error at p=.003, with lower mean iterations), but that is *not* a prediction for this challenge.

**Implementation:** Random/Latin-hypercube search over gamma0, explicit_gammas or interval center/width, pre_iter, set_max_iter, alpha and num_sets, separately by basis and p. Start 20–50 candidates on 20k–50k fixed validation shots, promote top 5–8 to ≥200k shots. Select 3 schedules based on *complementarity of per-shot failures*, not simply marginal accuracy. Train a tiny schedule selector using syndrome and early Relay posterior summaries, or choose best fixed schedule if routing overhead erases benefit. Account for per-shot decoder rebuild (~5 ms).

**Compute/time:** 2–4 h on 20 cores if existing shot-parallel pipeline is efficient; otherwise reduce grid. **Expected gain:** 10–30% relative residual failure reduction is an optimistic *search target*, not reported ML performance. **Risk:** low–medium, multiple-comparison overfitting. **Fair comparison:** freeze all parameters before fresh-seed test; same-shot comparisons; same-cost random schedule choice; show fixed-best, routed, and oracle portfolio.

## 3. Relay-posterior-guided OSD-0 / localized fallback

**Papers/code:** [Hillmann et al., “Localized statistics decoding,” arXiv:2406.18655](https://arxiv.org/abs/2406.18655); [ldpc package](https://github.com/quantumgizmos/ldpc); [Wolanski & Barber, “Ambiguity Clustering,” arXiv:2406.14527](https://arxiv.org/abs/2406.14527). Ambiguity Clustering reports up to 27× lower runtime than BP-OSD at matched accuracy in a BB circuit-level benchmark; hardware and setup differ from yours.

**Idea:** Relay-BP's posterior ratios already provide a reliability ordering even when the hard decision fails. A cheap OSD-0-like GF(2) solve or BP+LSD fallback on only ambiguous shots might recover errors at lower incremental cost than full BP-5. LSD was designed to replace expensive global OSD with localized reliability-guided inversion; published BB benchmarks show BP+LSD comparable to BP+OSD.

**Implementation:** Convert posterior ratios to stable LLR magnitudes, sort DEM columns by reliability, construct independent pivots and solve H e=s over GF(2). *Do not* mistake a syndrome-consistent correction for logical correctness; validate observable flips. A safer deadline path is to benchmark existing ldpc.BpLsdDecoder or its OSD-0 as a fallback, with input priors matched to your DEM. Trigger only on Relay-BP-1 ambiguity/nonconvergence; measure construction and Python/Rust overhead. Avoid writing a general dense elimination routine if integration takes >1 h.

**Compute/time:** 2–5 h. **Expected gain:** potentially useful on hard tail, magnitude unknown. **Risk:** medium; 936×8,784 rank/ordering overhead or library mismatch can overwhelm speed. **Fair comparison:** identical Relay-BP-1 outputs, identical shot sets; BP-5 fallback and random escalation matched on measured mean time and p99.

## 4. Residual logical classifier (rather than full decoder)

**Paper:** [Blue et al., arXiv:2504.13043](https://arxiv.org/abs/2504.13043); [code](https://github.com/jblue1/ml_bb_decoding).

**Idea:** Predict the residual logical observable class conditioned on Relay-BP's correction, not all 8,784 error variables. This makes ML an inexpensive residual decision layer. It is inspired by logical-class neural decoding but is not a published demonstration of gains on Relay-BP for this exact matrix.

**Implementation:** Train 12 binary classifiers (or a multi-output model) on paired syndrome/observable truth and BP-1 outputs: syndrome bits (936), weight, posterior confidence quantiles, counts of low-|LLR| columns, cycle-local aggregates, convergence and iterations. First try regularized logistic regression / gradient boosting; optionally a small two-layer MLP on RTX 3060. Use out-of-sample calibration; apply predicted observable correction only if validated against the memory-experiment output semantics.

**Compute/time:** 1–3 h. **Expected gain:** highly uncertain; 10–20% relative BP-1 failure reduction would be excellent but is not supported by direct literature. **Risk:** medium–high; logical failures may be too rare or poorly separable. **Fair comparison:** same-shot BP-1, residual classifier, and syndrome-only classifier; fresh test seeds and equal inference-cost accounting.

## 5. Calibrate loss-conditioned DEM priors

**Papers:** [Pecorari & Pupillo, “Quantum LDPC codes for erasure-biased atomic quantum processors,” arXiv:2502.20189](https://arxiv.org/abs/2502.20189); [Perrin, Jandura & Pupillo, “Quantum Error Correction resilient against Atom Loss,” Quantum (2025)](https://quantum-journal.org/papers/q-2025-10-13-1884/); [QuEra fault-tolerant neutral-atom architecture, Nature](https://doi.org/10.1038/s41586-025-09848-5).

**Idea:** Known loss locations are established useful decoder side information. Perrin et al. report nearly three orders of magnitude improvement over naive loss treatment in a surface-code setting, and Pecorari/Pupillo benchmark BB codes under erasure-biased circuit-level noise. QuEra reports erasure-informed and ML decoding. **Novelty caution:** no exact Gross/Relay-BP demonstration was found using cycle-resolved heralded-loss flags to reweight the corresponding merged Stim DEM column, but this is a negative search, not proof of first publication. Say “we are not aware of prior Gross/Relay-BP results with per-shot heralded-loss-conditioned DEM priors,” *not* “first loss-aware BB decoder.”

**Implementation:** Fit shared calibrated probability model P(e_j=1 | fired flag, cycle, basis, p, local syndrome, nearby flags, column type), e.g. regularized logistic regression or isotonic calibration for grouped categories. Preserve 1/2 for fired erasure if symmetry dictates it; test whether learned adjustments matter only for unfired flags or imperfect flagging. Compare flags alone, syndrome alone, shuffled flags, learned priors, and analytic priors.

**Compute/time:** 1–2 h. **Expected gain:** small additional improvement if 1/2 is physically exact; potentially stronger if detector merging and flag imperfections produce heterogeneous conditional channels. **Risk:** low. **Fair comparison:** paired flags/syndromes; loss/no-loss strata; fresh seeds; report your established 2–5× result separately from *incremental* ML gain.

## 6. Ambiguity / cluster statistics for confidence routing

**Paper/code:** [Lee, English & Bartlett, “Efficient Post-Selection for General Quantum LDPC Codes,” arXiv:2510.05795](https://arxiv.org/abs/2510.05795); [implementation](https://github.com/seokhyung-lee/ldpc-post-selection).

**Idea:** The authors show cluster structure and aggregated LLR confidence can be better indicators of difficult qLDPC shots than correction weight, including Gross-code benchmarks. Post-selection is not permitted as an accuracy trick in your task; use its confidence metrics solely as features for adaptive compute allocation.

**Implementation:** If BP+LSD exposes clusters, record count, max size, low-confidence mass and cluster-aggregated LLR. Otherwise threshold |Relay LLR| and compute connected components in the induced sparse Tanner subgraph with SciPy. Append these to the utility router and measure net incremental inference time.

**Compute/time:** 1–3 h. **Expected gain:** more plausible for confidence calibration and p99-aware routing than direct raw LER reduction. **Risk:** low for feature-only version, medium for new decoder integration. **Fair comparison:** ablate base router vs base+clusters, random-mix matched mean latency, fresh test seeds.

## 7. Full GNN / transformer / diffusion decoder — future work only

**Papers/code:** [Astra, arXiv:2408.07038](https://arxiv.org/abs/2408.07038), [code](https://github.com/arshpreetmaan/astra); [Blue et al., circuit-level BB transformer, arXiv:2504.13043](https://arxiv.org/abs/2504.13043), [code](https://github.com/jblue1/ml_bb_decoding); [masked diffusion qLDPC decoder, arXiv:2509.22347](https://arxiv.org/abs/2509.22347).

**Idea:** Astra reports scalable graph message passing and BB code-capacity gains; Blue et al. demonstrate circuit-level BB transformers but require massive training data; masked diffusion reports strong BB circuit-level accuracy and worst-case latency relative to some baselines. These are not apples-to-apples comparisons against your 12-cycle DEM Relay-BP-5, and none is a credible from-scratch CPU training project in under 2 h.

**Implementation sketch for later:** Tanner-graph GNN or masked diffusion model conditioned on syndrome, loss flags and Relay posteriors, predicting logical residuals rather than full correction; train on millions of simulated shots with GPU batching and cross-p/basis validation. **Compute/time:** days or longer, not this sprint. **Expected gain:** unquantified here. **Risk:** high. **Fair comparison:** full circuit-level same-shot benchmark against tuned Relay-BP, BP-OSD, and compute-matched hybrid.

## Learned BP / Relay-BP: feasibility assessment

Earlier neural BP work with trainable message weights and overcomplete checks is real: [arXiv:2212.10245](https://arxiv.org/abs/2212.10245), [arXiv:2308.08208](https://arxiv.org/abs/2308.08208). A Harvard thesis explored neural BP on BB codes and encountered difficulty transferring surface-code improvements: [Gong thesis](https://dash.harvard.edu/items/78dcef0b-782d-43e9-bf9b-a630fc7c8057). These settings do not establish that a 936×8,784 circuit-level neural-BP model can be trained effectively on 20 CPU cores in under 2 h. Low-dimensional gamma tuning is the much stronger practical bet.

Relay-BP memory updates take the form Λ_j(t)=(1−γ_j)Λ_j(0)+γ_j M_j(t−1). The API exposes gamma0, pre_iter, num_sets, set_max_iter, gamma_dist_interval, explicit_gammas, stop_nconv and alpha, but no warm-start/channel-update interface. Therefore any per-shot prior/schedule proposal must account for the ~5 ms decoder rebuild.

Correlated XYZ Relay-BP is another literature-supported direction but demands a much larger decoding matrix (roughly 2k×71k versus ~1k×9k) and is not a deadline-friendly CPU/Python task. [arXiv:2506.01779](https://arxiv.org/abs/2506.01779).

## Neutral-atom context and careful novelty statement

The existing loss-aware result is strong: at 1% heralded loss, your per-shot DEM reweighting reportedly cuts Relay-BP-5 failures by 2–5× and shuffled/control flags show the gain is due to side information. This is your team’s observed result, not a literature result.

A defensible submission sentence: “Loss-aware decoding and erasure-biased BB-code studies already exist; our implementation directly injects cycle-resolved heralded-loss flags into shot-specific Relay-BP priors for merged Gross-code detector-error mechanisms. We are not aware of a prior Gross/Relay-BP evaluation of this exact mechanism.”

Related neutral-atom context: [Infleqtion logical-qubit work (2025)](https://www.nature.com/articles/s41534-025-01095-w) and [Atom Computing toric-code QEC (2026)](https://atom-computing.com/wp-content/uploads/2026/06/AtomComputing_QuantumErrorCorrectionwithToricCode.pdf). Neither is evidence of an identical Gross/Relay-BP method.

## Fair evaluation protocol

1. **Paired shots:** all candidate decoders see the same syndrome, observable truth, loss flags and physical realization. Split by random seed before tuning; keep a completely untouched fresh-seed test set.
2. **Logical error per cycle:** use one declared conversion from 12-cycle experiment failure probability consistently across all models. Do not confuse experiment-level and per-cycle LER.
3. **Random controls:** match both escalation fraction and independently *measured mean end-to-end wall-clock latency*. The latter is the meaningful compute-matched control. Include decoder construction (~5 ms), Python/Rust overhead and fallback setup.
4. **Uncertainty:** paired bootstrap confidence intervals for error-rate differences, separate p=.003/.004/.005 and X/Z; report denominators and number of observed failures.
5. **Latency/resource:** mean, median, p95, p99, CPU-seconds per million shots, peak RSS, and GPU use if applicable. Also record p99 vs LER, not only mean vs LER.
6. **Pareto and oracle:** plot LER versus mean measured latency for all router thresholds, with random mixing as a continuous curve. Add the oracle that escalates only shots on which BP-5 uniquely succeeds; it bounds possible benefit from better ML.
7. **No leakage:** use training seeds for fitting, validation seeds for thresholds/schedule selection, and fresh held-out seeds for final comparisons. For gamma search, account for selection bias from many candidate sweeps.

## Recommended 4–6 hour sprint

**Top 1: Utility-aware routing.** Relabel existing BP-1/BP-5 paired results for incremental rescue/harm, fit a runtime model, and optimize error improvement per millisecond. Stop early if oracle/cost frontier proves random mixing cannot be beaten.

**Top 2: Gamma-portfolio search.** Start a parallel coarse sweep immediately; optimize fixed schedule first, then try 3-way portfolio selection only if joint failure patterns show complementarity. Report parameter tuning as the result if ML selection fails.

**Top 3: Posterior-guided cheap fallback.** Benchmark ldpc BP+LSD/OSD-0 on hard BP-1 shots against BP-5, including construction overhead. If integration stalls for >1 h, replace it with inexpensive ambiguity-cluster features in the existing router.

**Future-work-only:** GNN/transformer/masked-diffusion residual logical decoding conditioned on Relay posteriors and heralded flags. Strong research direction, not a 4–6 h deliverable.

## Source index

- Müller et al., Relay-BP: https://arxiv.org/abs/2506.01779 — https://github.com/zphy/relay_bp
- Blue et al., circuit-level BB ML: https://arxiv.org/abs/2504.13043 — https://github.com/jblue1/ml_bb_decoding
- Astra GNN: https://arxiv.org/abs/2408.07038 — https://github.com/arshpreetmaan/astra
- LSD: https://arxiv.org/abs/2406.18655 — https://github.com/quantumgizmos/ldpc
- Ambiguity Clustering: https://arxiv.org/abs/2406.14527
- Erasure-biased BB: https://arxiv.org/abs/2502.20189
- Loss-aware surface code: https://quantum-journal.org/papers/q-2025-10-13-1884/
- qLDPC cluster confidence: https://arxiv.org/abs/2510.05795 — https://github.com/seokhyung-lee/ldpc-post-selection
- Diffusion decoder: https://arxiv.org/abs/2509.22347
- Neural BP: https://arxiv.org/abs/2212.10245 and https://arxiv.org/abs/2308.08208
- Adaptive decoder selection (Zenodo): https://zenodo.org/records/21466148
- QuEra neutral atom: https://doi.org/10.1038/s41586-025-09848-5
- Infleqtion neutral atom: https://www.nature.com/articles/s41534-025-01095-w
