"""Compare the frozen judge router against the likelihood-aware candidate on fresh paired shots."""
import argparse
import os
from pathlib import Path
import sys
from time import perf_counter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kit", type=Path, required=True)
parser.add_argument("--judge", type=Path, required=True)
parser.add_argument("--p", type=float, choices=[0.003, 0.004, 0.005], default=0.005)
parser.add_argument("--shots", type=int, default=128, help="X/Z pairs")
parser.add_argument("--seed", type=int, default=923)
args = parser.parse_args()
if args.shots < 1 or args.seed < 0:
    parser.error("Use positive shots and a nonnegative seed")
sys.path[:0] = [str(args.judge.resolve()), str(args.kit.resolve())]
os.chdir(args.kit)
import judge
import numpy as np
from circuit import sample
from loss_circuit import lossy_circuit, lossy_dem_matrices


class Candidate(judge.SixBitDecoder):
    def __init__(self, H, priors):
        super().__init__(H, priors)
        self.constant = np.log1p(-priors).sum()
        self.log_odds = np.log(priors / (1-priors))
        cells = judge.POLICY["cells"]
        self.table = np.array([c["n"] >= 12 and c["rescue"] - c["regress"] >
                               0.04*c["cost"]/1000 for c in cells]).reshape(16, 4)

    def decode(self, syndrome):
        first = self.fast.decode_detailed(syndrome)
        a, b, c, d, e = judge.POLICY["cut"]
        key = (8*int(not first.success) + 4*int(syndrome.sum() > a)
               + 2*int(first.iterations > b) + int(first.iterations > c))
        row = self.table[key]
        escalate = row[0]
        if np.any(row != escalate):
            confidence = np.nan_to_num(np.abs(np.asarray(first.posterior_ratios)), nan=0).clip(0,100)
            escalate = row[2*int(confidence.mean() < d) + int(np.percentile(confidence,10) < e)]
        if not escalate:
            return first.decoding
        self.calls += 1
        second = self.strong.decode_detailed(syndrome)
        if first.success:
            if not second.success:
                return first.decoding
            def likelihood(result):
                return self.log_odds[np.asarray(result.decoding, bool)].sum() + self.constant
            if likelihood(first) > likelihood(second):
                return first.decoding
        return second.decoding


fail = np.zeros((2, args.shots, 2), bool)
seconds = np.zeros_like(fail, dtype=float)
calls = np.zeros(2, int)
for b, basis in enumerate("XZ"):
    circuit = lossy_circuit(basis, args.p, 0)
    H, L, priors, ids = lossy_dem_matrices(circuit, basis, False)
    det, obs = sample(circuit, args.shots + 1, args.seed + b)
    candidate = judge.SixBitDecoder if args.p == 0.003 else Candidate
    models = [judge.SixBitDecoder(H, priors), candidate(H, priors)]
    for model in models:
        model.decode(det[0,ids])
        model.calls = 0
    for j in range(args.shots):
        syndrome = det[j+1,ids]
        # Alternate first runner. Both models see the same syndromes in the same order.
        for i in ((j+b)%2, (j+b+1)%2):
            start = perf_counter()
            correction = np.asarray(models[i].decode(syndrome), np.uint8)
            seconds[i,j,b] = perf_counter() - start
            fail[i,j,b] = np.any((L @ correction)%2 != obs[j+1])
    calls += [m.calls for m in models]
    print(f"{basis} complete", flush=True)
print(f"p={args.p}, {args.shots} pairs, X seed {args.seed}, Z seed {args.seed+1}")
for name, f, t, n in zip(("frozen router","candidate"), fail, seconds, calls):
    ms = 1000*t.sum(axis=1)
    print(f"{name}: {f.any(axis=1).sum()} failed, mean {ms.mean():.1f} ms, "
          f"p99 {np.percentile(ms,99):.1f} ms, {n} escalations")
old, new = fail.any(axis=2)
print(f"rescues {np.sum(old & ~new)}, regressions {np.sum(new & ~old)}")
