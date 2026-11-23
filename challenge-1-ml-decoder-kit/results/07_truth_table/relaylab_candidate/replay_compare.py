"""Replay a cheaper six-bit routing policy with a likelihood tie-break on escalations."""
import argparse
from pathlib import Path
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path, help="Path to the team's kit results directory")
parser.add_argument("judge", type=Path, help="Path to the extracted RelayLab-judge folder")
args = parser.parse_args()
sys.path.insert(0, str(args.judge.resolve()))
import judge


def evaluate(folder, p, seed, n, prices):
    suffix = f"_p{p}_l0.0_unaware_s{seed}_n{n}_shard0of1.npz"
    with np.load(folder / ("relay1" + suffix)) as f, np.load(folder / ("relay5" + suffix)) as s:
        for name in ("host", "shot_idx", "p", "p_loss", "loss_mode", "seed", "cycles",
                     "basis", "workers", "shots", "stim_version", "feature_names"):
            np.testing.assert_array_equal(f[name], s[name], err_msg=name)
        names = list(f["feature_names"])
        fast, strong = f["features"], s["features"]
        def col(data, name):
            return data[..., names.index(name)]
        a, b, c, d, e = judge.POLICY["cut"]
        conv = col(fast, "converged") > 0
        strong_conv = col(strong, "converged") > 0
        bits = (~conv, col(fast, "syndrome_weight") > a, f["iterations"] > b,
                f["iterations"] > c, col(fast, "posterior_abs_mean") < d,
                col(fast, "posterior_abs_p10") < e)
        keys = sum(bit.astype(int) << (5-i) for i, bit in enumerate(bits))
        prefer_fast = conv & (~strong_conv | (col(fast, "loglik") > col(strong, "loglik")))
        ff, sf, ft, st = f["fail"], s["fail"], f["seconds"], s["seconds"]
        def score(escalate, retain):
            fails = np.where(escalate & np.logical_not(retain), sf, ff).any(axis=1)
            ms = 1000*(ft + escalate*st).sum(axis=1)
            return fails, ms
        baseline = score(judge.TABLE[keys], False)
        policies = []
        for price in prices:
            table = np.array([cell["n"] >= 12 and cell["rescue"] - cell["regress"] >
                              price*cell["cost"]/1000 for cell in judge.POLICY["cells"]])
            policies.append((score(table[keys], prefer_fast), score(table[keys], False)))
        return baseline, policies


prices = np.array([0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.09, 0.12])
for p in judge.RATES:
    (bf, bt), candidates = evaluate(args.results, p, 2, 10000, prices)
    # At calibration require strictly faster average replay; then minimize failures.
    acceptable = [(f.sum(), t.mean(), i) for i, ((f, t), _) in enumerate(candidates)
                  if t.mean() < bt.mean() and f.sum() <= bf.sum()]
    if not acceptable:
        print(f"p={p}: no qualifying candidate on calibration")
        continue
    price = prices[min(acceptable)[-1]]
    print(f"p={p} price frozen at {price} from seed 2")
    seeds = {0.003: [], 0.004: [(4, 10000)], 0.005: [(10, 10000)]}[p]
    for seed, size in seeds + [(16, 1000)]:
        folder = args.judge / "inputs" if seed == 16 else args.results
        (old, before), [((new, after), (threshold_only, _))] = evaluate(folder, p, seed, size, [price])
        print(f"  seed {seed}, {size} pairs: failures {old.sum()} -> {new.sum()}, "
              f"mean {before.mean():.1f} -> {after.mean():.1f} ms, "
              f"p99 {np.percentile(before,99):.1f} -> {np.percentile(after,99):.1f} ms, "
              f"rescues {np.sum(old & ~new)}, regressions {np.sum(new & ~old)}, "
              f"failures with threshold change alone {threshold_only.sum()}")
