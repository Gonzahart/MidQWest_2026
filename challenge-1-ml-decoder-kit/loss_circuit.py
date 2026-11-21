"""Add heralded qubit loss to the kit's Gross code memory circuits.

Model (neutral-atom loss with a loss-detection measurement, treated as an erasure):
at the start of every syndrome cycle each of the 144 data qubits is lost with probability
p_loss. A lost atom is detected and replaced, which leaves a random Pauli on that qubit
(a maximally mixed state) plus a herald bit that says *where* it happened, not *what*.
Stim's HERALDED_ERASE does exactly this.

Every herald bit becomes a DETECTOR with coordinate 2 (the kit uses 0 for X checks and 1 for
Z checks). A decoder can then use loss information simply by keeping coordinate-2 detectors
as extra rows of H: an error mechanism caused by an erasure flips its herald detector.

Inserting new measurements shifts the rec[-k] lookbacks of every later detector, so the
circuit is flattened and every lookback is rewritten against the new measurement record.

  python loss_circuit.py --basis X --p 0.004 --p-loss 0.01 --out lossy_X.stim
"""
import argparse

import numpy as np
import stim

from circuit import (CHECK_COORDINATE, GROSS_CIRCUITS, _incidence_matrix, _merge_duplicates, _row_numbers,
                     load_circuit)

HERALD_COORDINATE = 2
DATA_QUBITS = range(72, 216)     # left and right data qubits of the kit's Gross code layout
X_CHECKS = set(range(72))        # an RX on these opens every syndrome cycle


def add_loss(circuit, p_loss, data_qubits=DATA_QUBITS):
    """Flattened copy of circuit with a heralded erasure of every data qubit at each cycle start.

    Built as text: stim.Circuit.append fuses consecutive DETECTORs with equal arguments into one.
    """
    lines, old_to_new, n_old, n_new = [], [], 0, 0
    for inst in circuit.flattened():
        name, targets = inst.name, inst.targets_copy()
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            args = ",".join(repr(a) for a in inst.gate_args_copy())
            recs = " ".join(f"rec[{old_to_new[n_old + t.value] - n_new}]" for t in targets)
            lines.append(f"{name}({args}) {recs}")
            continue
        lines.append(str(inst))
        produced = _measurements(inst)
        old_to_new.extend(range(n_new, n_new + produced))
        n_old, n_new = n_old + produced, n_new + produced
        if p_loss > 0 and name == "RX" and {t.value for t in targets} == X_CHECKS:
            n = len(data_qubits)
            lines.append(f"HERALDED_ERASE({p_loss}) " + " ".join(map(str, data_qubits)))
            lines.extend(f"DETECTOR({HERALD_COORDINATE}) rec[{k - n}]" for k in range(n))
            n_new += n
    return stim.Circuit("\n".join(lines))


def _measurements(inst):
    """Number of measurement-record bits one flattened instruction writes."""
    data = stim.gate_data(inst.name)
    if not data.produces_measurements:
        return 0
    return len(inst.targets_copy()) // (2 if data.is_two_qubit_gate else 1)


def lossy_circuit(basis, p, p_loss):
    """The kit's Gross code memory circuit at noise p with heralded loss p_loss."""
    return add_loss(load_circuit(GROSS_CIRCUITS[basis], p), p_loss)


def detector_ids_with_heralds(circuit, basis, use_heralds):
    """Detectors on checks of the memory basis, plus the loss heralds if use_heralds."""
    wanted = {CHECK_COORDINATE[basis]} | ({HERALD_COORDINATE} if use_heralds else set())
    coords = circuit.get_detector_coordinates()
    return np.array([d for d in range(circuit.num_detectors) if coords[d] and coords[d][0] in wanted])


def lossy_dem_matrices(circuit, basis, use_heralds):
    """H, L, priors like circuit.dem_matrices, for a circuit with HERALDED_ERASE.

    With use_heralds the loss heralds are extra rows of H (after the check detectors), so a
    decoder sees where loss happened. Without them the same errors look like extra Pauli noise.
    """
    det_ids = detector_ids_with_heralds(circuit, basis, use_heralds)
    dem = circuit.detector_error_model(approximate_disjoint_errors=True).flattened()
    errors = _merge_duplicates(dem, _row_numbers(det_ids))
    detectors, observables = zip(*errors, strict=True)
    H = _incidence_matrix(detectors, len(det_ids))
    L = _incidence_matrix(observables, circuit.num_observables)
    return H, L.tocsr(), np.array(list(errors.values())), det_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--p-loss", type=float, default=0.01)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    circuit = lossy_circuit(args.basis, args.p, args.p_loss)
    circuit.to_file(args.out)
    print(args.out, f"{circuit.num_qubits} qubits, {circuit.num_detectors} detectors, "
          f"{circuit.num_measurements} measurements")


if __name__ == "__main__":
    main()
