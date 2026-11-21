"""Evaluate Relay-BP-1 → Relay-BP-5 routing on independent, matched Gross-code shots."""
import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binomtest

from decoders import per_cycle_rate, relay_decoder
from evaluate import problems

FEATURES = ("syndrome_weight", "iterations", "nonconverged", "posterior_abs_mean",
            "posterior_abs_p10")


def decode_chunk(job):
    H, L, priors, detectors, observables = job
    features = np.empty((len(detectors), len(FEATURES)))
    failed = np.empty((len(detectors), 2), dtype=bool)
    seconds = np.empty((len(detectors), 2))
    for stage, stop in enumerate((1, 5)):
        decoder = relay_decoder(H, priors, stop_nconv=stop)
        for i, (syndrome, truth) in enumerate(zip(detectors, observables, strict=True)):
            start = time.perf_counter()
            result = decoder.decode_detailed(syndrome)
            correction = np.asarray(result.decoding, dtype=np.uint8)
            seconds[i, stage] = time.perf_counter() - start
            failed[i, stage] = np.any((L @ correction) % 2 != truth)
            if stage == 0:
                # Clip saturated log-likelihood ratios before computing summaries.
                confidence = np.nan_to_num(np.abs(result.posterior_ratios), nan=0).clip(0, 100)
                features[i] = (syndrome.sum(), result.iterations, not result.success,
                               confidence.mean(), np.percentile(confidence, 10))
    return features, failed, seconds


def collect(pool, args, shots, seed):
    datasets = []
    for H, L, priors, det, obs in problems(args.p, 0, shots, seed, False):
        chunks = np.array_split(np.arange(shots), min(args.workers, shots))
        parts = pool.map(decode_chunk, [(H, L, priors, det[ids], obs[ids]) for ids in chunks])
        datasets.append(tuple(np.concatenate(column) for column in zip(*parts, strict=True)))
    return tuple(np.stack(column, axis=1) for column in zip(*datasets, strict=True))


def train(features, failed):
    x = features.reshape(-1, len(FEATURES))
    y = (failed[..., 0] & ~failed[..., 1]).ravel()
    if not y.any() or y.all():
        raise ValueError("training data needs both rescue and non-rescue examples")
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    standardized = (x - center) / scale

    def loss(weights):
        logits = standardized @ weights[1:] + weights[0]
        residual = expit(logits) - y
        # L2 penalty on feature coefficients, leaving the intercept unpenalized.
        value = np.mean(np.logaddexp(0, logits) - y * logits) + 0.005 * (weights[1:] ** 2).sum()
        gradient = np.r_[residual.mean(), standardized.T @ residual / len(y) + 0.01 * weights[1:]]
        return value, gradient

    fit = minimize(loss, np.zeros(len(FEATURES) + 1), jac=True, method="L-BFGS-B")
    if not fit.success:
        raise ValueError(f"logistic regression failed: {fit.message}")
    return center, scale, fit.x, int(y.sum())


def compare(features, failed, seconds, model, budget):
    center, scale, weights, _ = model
    rng = np.random.default_rng(1)
    ties = rng.random(features.shape[:2])
    count = round(budget * len(features))
    learned = np.zeros_like(ties, dtype=bool)
    simple = learned.copy()
    random = learned.copy()
    score = (features - center) / scale @ weights[1:] + weights[0]
    for basis in range(2):
        # Nonconvergence takes priority, then longer Relay-BP-1 runs.
        order = np.lexsort((ties[:, basis], -features[:, basis, 1], -features[:, basis, 2]))
        simple[order[:count], basis] = True
        order = np.lexsort((ties[:, basis], -score[:, basis]))
        learned[order[:count], basis] = True
        random[np.argsort(ties[:, basis])[:count], basis] = True

    rows = []
    pair_outcomes = {}
    for name, route in (("Relay-BP-1", None), ("Relay-BP-5", None),
                        ("Simple rule", simple), ("Learned", learned), ("Matched random", random)):
        if route is None:
            stage = int(name == "Relay-BP-5")
            pair_failed = failed[..., stage].any(axis=1)
            elapsed = seconds[..., stage].sum(axis=1) * 1000
        else:
            pair_failed = np.where(route, failed[..., 1], failed[..., 0]).any(axis=1)
            elapsed = (seconds[..., 0] + np.where(route, seconds[..., 1], 0)).sum(axis=1) * 1000
        rate, _ = per_cycle_rate(pair_failed, 12)
        pair_outcomes[name] = pair_failed
        rows.append(dict(policy=name, failed_pairs=int(pair_failed.sum()), ler_per_cycle=rate,
                         mean_ms=elapsed.mean(), p99_ms=np.percentile(elapsed, 99),
                         fallback_rate=0 if route is None else route.mean()))

    simple_fail, learned_fail = (pair_outcomes[name] for name in ("Simple rule", "Learned"))
    wins = int((simple_fail & ~learned_fail).sum())
    losses = int((learned_fail & ~simple_fail).sum())
    p_value = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
    return rows, wins, losses, p_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--train-shots", type=int, default=300)
    parser.add_argument("--test-shots", type=int, default=200)
    parser.add_argument("--train-seed", type=int, default=41)
    parser.add_argument("--test-seed", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fallback-rate", type=float, default=0.20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if abs(args.train_seed - args.test_seed) < 2:
        parser.error("train/test seeds must not overlap: each split uses seed and seed + 1")
    if min(args.train_shots, args.test_shots, args.workers) < 1 or not 0 <= args.fallback_rate <= 1:
        parser.error("require positive shot and worker counts and a fallback rate in [0, 1]")

    with Pool(args.workers) as pool:
        print(f"Collecting {args.train_shots} training pairs", flush=True)
        train_features, train_failed, _ = collect(pool, args, args.train_shots, args.train_seed)
        model = train(train_features, train_failed)
        print(f"Training rescue examples: {model[-1]}. Collecting {args.test_shots} held-out pairs.",
              flush=True)
        features, failed, seconds = collect(pool, args, args.test_shots, args.test_seed)

    rows, wins, losses, p_value = compare(features, failed, seconds, model, args.fallback_rate)
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with args.out.with_suffix(".npz").open("xb") as output:
        np.savez_compressed(output, features=features, fail=failed, seconds=seconds,
                            train_features=train_features, train_fail=train_failed,
                            feature_names=FEATURES, center=model[0], scale=model[1], weights=model[2],
                            p=args.p, p_loss=0, train_seed=args.train_seed, test_seed=args.test_seed,
                            workers=args.workers, fallback_rate=args.fallback_rate)
    print(csv_path)
    print(f"Learned-only pair wins: {wins}; simple-only pair wins: {losses}; paired p={p_value:.3g}")
    print("Latency is replayed decoder time and excludes feature extraction and inference.")


if __name__ == "__main__":
    main()
