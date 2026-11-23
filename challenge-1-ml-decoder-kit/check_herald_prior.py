"""Check of the fired-herald prior (1/2) and the herald -> column mapping, by direct Pauli injection.
For one lost data qubit in an early, middle and final cycle, and each memory basis:
  1. inject I, X, Y, Z deterministically at the erasure location in a noiseless copy of the circuit and record
     which decoded-basis check detectors and which logical observables flip;
  2. compare with the DEM columns that contain that qubit's herald row (what the decoder uses);
  3. report prior/p_loss of each such column (the reweight rule clips this at 0.5).
Also: global count of DEM columns dropped by the check-row projection that still flip an observable."""
"""
  python check_herald_prior.py > results/03_herald_control/herald_prior_check.txt
"""
import sys, numpy as np, stim
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '.')
from loss_circuit import lossy_circuit, lossy_dem_matrices, HERALD_COORDINATE, DATA_QUBITS
from circuit import CHECK_COORDINATE

P, PL = 0.004, 0.01
NOISE = ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2")

def injected(circuit, cycle, qubit, pauli):
    """Noiseless circuit with one deterministic Pauli right after the chosen erasure; heralds kept (p=0)."""
    lines, k = [], 0
    for inst in circuit.flattened():
        if inst.name in NOISE:
            continue
        if inst.name == "HERALDED_ERASE":
            lines.append("HERALDED_ERASE(0) " + " ".join(str(t.value) for t in inst.targets_copy()))
            if k == cycle and pauli != "I":
                lines.append(f"{pauli}_ERROR(1) {qubit}")
            k += 1
            continue
        if inst.name.startswith("M") and inst.gate_args_copy():   # noisy measurements -> noiseless
            lines.append(f"{inst.name} " + " ".join(str(t) if not hasattr(t,'value') else _tgt(t) for t in inst.targets_copy()))
            continue
        lines.append(str(inst))
    return stim.Circuit("\n".join(lines)), k

def _tgt(t):
    return ("!" if t.is_inverted_result_target else "") + str(t.value)

report = []
bad_total = 0
for basis in "XZ":
    circ = lossy_circuit(basis, P, PL)
    H, L, priors, det_ids = lossy_dem_matrices(circ, basis, use_heralds=True)
    H, L = H.tocsc(), L.tocsc()
    coords = circ.get_detector_coordinates()
    is_herald = np.array([coords[d][0] == HERALD_COORDINATE for d in det_ids])
    check_rows = np.flatnonzero(~is_herald)
    heralds_all = [d for d in range(circ.num_detectors) if coords[d] and coords[d][0] == HERALD_COORDINATE]
    nq = len(DATA_QUBITS)
    # global projection check (what Reweighted does): columns with no check rows are dropped
    check_support = np.asarray(H[check_rows].sum(axis=0)).ravel() > 0
    lnz = np.asarray(L.sum(axis=0)).ravel() > 0
    dropped_logical = int((~check_support & lnz).sum())
    bad_total += dropped_logical
    print(f"\n=== {basis} memory: H with heralds {H.shape[0]} rows ({(~is_herald).sum()} check + {is_herald.sum()} herald) x {H.shape[1]} cols, "
          f"nnz {H.nnz}; projected to check rows: {check_support.sum()} cols kept; dropped cols that flip a logical: {dropped_logical}")
    Hh = H[np.flatnonzero(is_herald)][:, np.flatnonzero(check_support)]
    per_herald = np.diff(Hh.tocsr().indptr)
    kept_ratio = priors[np.flatnonzero(check_support)][np.asarray(Hh.sum(axis=0)).ravel() > 0] / PL
    print(f"every herald: kept columns per herald min {per_herald.min()} max {per_herald.max()}; "
          f"herald columns prior/p_loss range {kept_ratio.min():.3f}-{kept_ratio.max():.3f}")
    det_pos = {d: i for i, d in enumerate(det_ids)}
    for cycle in (0, 5, 11):
        for qubit in (72, 150):
            hdet = heralds_all[cycle * nq + (qubit - 72)]
            row = det_pos[hdet]
            cols = H[row].nonzero()[1] if False else np.flatnonzero(H[[row]].toarray().ravel())
            dem_cols = {}
            for c in cols:
                sig = tuple(det_ids[check_rows[np.flatnonzero(H[check_rows, c].toarray().ravel())]])
                obs = tuple(L[:, c].nonzero()[0])
                dem_cols[(sig, obs)] = priors[c]
            inj = {}
            for pauli in "IXYZ":
                ic, nk = injected(circ, cycle, qubit, pauli)
                det, obs = ic.compile_detector_sampler().sample(1, separate_observables=True)
                dets = np.flatnonzero(det[0])
                checks = tuple(d for d in dets if coords[d][0] == CHECK_COORDINATE[basis])
                inj.setdefault((checks, tuple(np.flatnonzero(obs[0]))), []).append(pauli)
            # expected merged columns from injection: classes with any check or observable effect
            ok = True
            lines = []
            for key, paulis in inj.items():
                eff = key != ((), ())
                in_dem = key in dem_cols
                ratio = dem_cols[key] / PL if in_dem else None
                if eff and not in_dem: ok = False
                lines.append(f"{'+'.join(paulis):5s} -> {len(key[0]):2d} checks, obs {[int(o) for o in key[1]][:3]}{'...' if len(key[1])>3 else ''}: "
                             + (f"DEM column prior/p_loss = {ratio:.3f}" if in_dem else ("no column (no effect, dropped)" if not eff else "MISSING IN DEM")))
            extra = [k for k in dem_cols if k not in inj]
            if extra: ok = False
            print(f"cycle {cycle:2d} qubit {qubit}: {'OK' if ok else 'MISMATCH'}  ({len(cols)} DEM columns on this herald)")
            for l in lines: print("    " + l)
            if extra: print("    extra DEM columns not produced by injection:", extra)
print("\nall dropped-column logical checks clean" if bad_total == 0 else f"\nWARNING: {bad_total} dropped columns flip a logical")
