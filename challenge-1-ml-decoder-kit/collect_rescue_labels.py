"""Collect matched rescue labels with fallback evaluated only on failed first decodes."""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import stim

from decoders import relay_decoder
from evaluate import problems
from sweep import FEATURE_NAMES, relay_features

def decode(job):
    H, L, priors, detectors, observables = job
    first = relay_decoder(H, priors, stop_nconv=1)
    fallback = relay_decoder(H, priors, stop_nconv=5)
    fail1 = np.empty(len(detectors), dtype=bool)
    fail5 = np.zeros(len(detectors), dtype=bool)
    features = np.empty((len(detectors), len(FEATURE_NAMES)))
    iterations = np.empty(len(detectors), dtype=np.int32)
    for i, (syndrome, truth) in enumerate(zip(detectors, observables, strict=True)):
        result = first.decode_detailed(syndrome)
        correction = np.asarray(result.decoding, dtype=np.uint8)
        features[i] = relay_features(result, correction, priors, syndrome)
        iterations[i] = result.iterations
        fail1[i] = np.any((L @ correction) % 2 != truth)
        if fail1[i]:
            correction = np.asarray(fallback.decode(syndrome), dtype=np.uint8)
            fail5[i] = np.any((L @ correction) % 2 != truth)
    return fail1, fail5, features, iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sampled-pairs", type=int, default=20000)
    parser.add_argument("--training-pairs", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if not 1 <= args.training_pairs <= args.sampled_pairs or args.workers < 1:
        parser.error("require positive workers and training pairs within sampled pairs")
    args.out.mkdir()
    indices = np.arange(args.training_pairs)
    by_basis = []
    with Pool(args.workers) as pool:
        for b, (H, L, priors, det, obs) in zip("XZ", problems(args.p, 0, args.sampled_pairs, args.seed, False), strict=True):
            print(f"Collecting {b}: first {args.training_pairs} IDs from {args.sampled_pairs} sampled shots", flush=True)
            chunks = np.array_split(indices, min(args.training_pairs, args.workers * 4))
            parts = pool.map(decode, [(H, L, priors, det[ids], obs[ids]) for ids in chunks])
            joined = tuple(np.concatenate(column) for column in zip(*parts, strict=True))
            by_basis.append(joined)
            print(f"{b}: {joined[0].sum()} first-stage failures; "
                  f"{(joined[0] & ~joined[1]).sum()} rescues", flush=True)
    fail1, fail5, features, iterations = (
        np.stack(column, axis=1) for column in zip(*by_basis, strict=True)
    )
    metadata = dict(p=args.p, p_loss=0, seed=args.seed, shots=args.sampled_pairs, cycles=12,
                    stim_version=stim.__version__, shot_idx=indices,
                    seconds=np.zeros((args.training_pairs, 2)), basis=np.asarray(["X", "Z"]))
    np.savez_compressed(args.out / "local_relay1.npz", decoder="relay1", fail=fail1,
                        features=features, feature_names=FEATURE_NAMES, iterations=iterations, **metadata)
    np.savez_compressed(args.out / "local_relay5_labels.npz", decoder="relay5", fail=fail5,
                        evaluated=fail1, **metadata)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
