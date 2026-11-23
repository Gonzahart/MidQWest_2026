"""Compare early indicators of slow decoding using matched zero-loss p=0.005 archives."""
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("kit", type=Path, help="Kit directory with its circuits and results")
args = parser.parse_args()
sys.path.insert(0, str(args.kit.resolve()))
from circuit import sample
from loss_circuit import lossy_circuit, lossy_dem_matrices


def observations(seed):
    suffix = f"_p0.005_l0.0_unaware_s{seed}_n10000_shard0of1.npz"
    results = args.kit / "results"
    weights, densities = [], []
    with np.load(results / ("relay1" + suffix)) as fast, np.load(results / ("relay5" + suffix)) as strong:
        for key in ("seed", "shot_idx", "basis", "host", "workers", "shots", "p", "p_loss", "stim_version"):
            np.testing.assert_array_equal(fast[key], strong[key])
        np.testing.assert_array_equal(fast["shot_idx"], np.arange(10000))
        for b, basis in enumerate("XZ"):
            circuit = lossy_circuit(basis, 0.005, 0)
            H, _, _, ids = lossy_dem_matrices(circuit, basis, False)
            det, _ = sample(circuit, 10000, seed + b)
            active = sparse.csr_matrix(det[:, ids], dtype=np.int32)
            weight = np.asarray(active.sum(axis=1)).ravel()
            pos = list(fast["feature_names"]).index("syndrome_weight")
            np.testing.assert_array_equal(weight, fast["features"][:, b, pos])
            adjacent = H.astype(np.int32) @ H.T
            adjacent.setdiag(0)
            adjacent.eliminate_zeros()
            adjacent.data[:] = 1
            double_edges = np.asarray((active @ adjacent).multiply(active).sum(axis=1)).ravel()
            weights.append(weight)
            densities.append(double_edges / (2 * np.maximum(weight, 1)))
        gain = fast["fail"].astype(int) - strong["fail"].astype(int)
        return (np.stack(weights, axis=1).ravel(), np.stack(densities, axis=1).ravel(),
                fast["seconds"].ravel(), gain.ravel())


w, density, seconds, _ = observations(2)
test_w, test_density, test_seconds, test_gain = observations(10)
cuts = np.quantile(w, np.arange(1, 10) / 10)
wbin = np.searchsorted(cuts, w)
test_wbin = np.searchsorted(cuts, test_w)
target = np.log1p(1000 * seconds)
for geometry in (False, True):
    group, test_group = 4*wbin, 4*test_wbin
    if geometry:
        for j in np.unique(wbin):
            cuts = np.quantile(density[wbin == j], [.25, .5, .75])
            group[wbin == j] += np.searchsorted(cuts, density[wbin == j])
            test_group[test_wbin == j] += np.searchsorted(cuts, test_density[test_wbin == j])
    scores = np.full(len(test_seconds), target.mean())
    for k in np.unique(group):
        members = group == k
        scores[test_group == k] = (target[members].sum() + 30*target.mean()) / (members.sum() + 30)
    tie = np.random.default_rng(47).random(len(scores))
    selected = np.lexsort((tie, scores))[-2000:]
    slow = test_seconds >= np.quantile(test_seconds, .9)
    print("weight + geometry" if geometry else "weight alone",
          f"selected={len(selected)}, slow hits={slow[selected].sum()}, net basis rescues={test_gain[selected].sum()}")
