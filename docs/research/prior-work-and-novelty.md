# Prior work and what is new here

Written Sun Nov 22, 2026 (before the final presentation) to pin down which claims we could make. We read the closest paper (Murri) in full. The other papers are cited from the reports in this folder.

## Closest prior work

**Murri, *Maximally-Matched Mixture Decoding for Correlated Atom Loss in Quantum LDPC Codes*, [arXiv:2610.38283](https://arxiv.org/abs/2610.38283) (Oct 2026).** Per-shot loss-aware relay-BP on the [[144,12,12]] Gross code. Its loss model is harder than ours: loss happens at gates, lost atoms drop out of later gates, losses come in correlated pairs, and a loss is only detected at readout or teleportation. It builds a per-readout mixture error model over the likely loss scenarios (maximal matchings), and reports up to ~25× lower logical error than loss-unaware decoding and ~2× lower than the uniform mixture of Baranes et al. Mostly 2 rounds at p = 0.006–0.008. No latency is reported.

**Baranes et al., *Leveraging qubit loss detection in fault-tolerant quantum algorithms*, Physical Review X 16, 011002 (2026).** The delayed-erasure decoder that Murri compares against. It treats loss events as uncorrelated.

| | Murri (mixture decoding) | Ours |
|---|---|---|
| Loss model | Delayed, at gates, correlated (realistic) | Prompt herald, data qubits at cycle start (idealized) |
| Per-shot work | Build a new mixture error model | Change priors on the fixed error model (~5 ms rebuild) |
| Decoders | relay-BP | relay-BP and BP-OSD, plus a GPU relay-BP path |
| Setting | Mostly 2 rounds, p = 0.006–0.008 | 12 cycles, p = 0.003–0.005 |
| Latency | Not reported | Measured on one worker: 1.5–2.1× lower mean time |
| Checks | — | Real vs blind vs shuffled flags; Pauli-injection proof of the ½ prior |

## Our novelty statement

Loss-aware decoding of the Gross code already exists for delayed, correlated loss (Baranes et al. 2026; Murri 2026). We study the prompt-herald regime. We show that the exact per-shot prior is ½ on a single merged error column, and apply it to relay-BP and BP-OSD without changing the decoding graph. We measure that it lowers latency as well as failures, and our controls show that the gain comes from each shot's own flags.

We do **not** claim to be the first loss-aware relay-BP decoder for the Gross code. The prompt-erasure model is the main caveat.

## Background papers

- Gross code: Bravyi et al., *High-threshold and low-overhead fault-tolerant quantum memory*, [arXiv:2308.07915](https://arxiv.org/abs/2308.07915).
- Relay-BP: Müller et al., *Improved belief propagation is sufficient for real-time decoding of quantum memory*, [arXiv:2506.01779](https://arxiv.org/abs/2506.01779).
- Surface code introduction (used for Friday-night prep): Fowler, *Surface code quantum computation* (2025). Search for it on [arXiv](https://arxiv.org/search/?query=Fowler+Surface+code+quantum+computation&searchtype=all). For the classic introduction, see Fowler et al., *Surface codes: Towards practical large-scale quantum computation*, [arXiv:1208.0928](https://arxiv.org/abs/1208.0928).

## Future work we see

- Delayed and correlated loss (Murri's regime), using our cheap prior update in place of a full mixture model.
- A learned model from flags and syndrome to priors, for imperfect heralds.
- Richer ambiguity features for the router.
