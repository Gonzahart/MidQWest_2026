"""Compare CPU and GPU relay-BP on the same Gross-code circuit samples."""
import argparse
import csv
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.stats import binomtest

from decoders import _timed, per_cycle_rate, relay_decoder
from gpu_relay import GpuRelay
from sweep import CYCLES, problems


def compare(args):
    shape = (args.shots, 2)
    failures = {name: np.zeros(shape, bool) for name in ("cpu", "gpu")}
    seconds = {name: np.zeros(shape) for name in failures}
    iterations = np.empty(shape, dtype=int)
    agree = np.empty(shape, bool)
    rebuild_ms = []
    for b, (H, L, priors, det, obs, _) in enumerate(problems(args)):
        extra = {"gamma_ensemble_size": args.gamma_ensemble_size} if args.gamma_ensemble_size else {}
        gpu = GpuRelay(H, priors, stop_nconv=args.stop_nconv, **extra)
        start = time.perf_counter()
        gpu.set_priors(priors)
        rebuild_ms.append(1000 * (time.perf_counter() - start))
        cpu = relay_decoder(H, priors, stop_nconv=args.stop_nconv)
        methods = {"cpu": cpu.decode, "gpu": gpu.decode}
        for decode in methods.values():
            decode(det[0])
        for i, syndrome in enumerate(det):
            predictions = {}
            for name in (("cpu", "gpu") if i % 2 == 0 else ("gpu", "cpu")):
                error, seconds[name][i, b] = _timed(methods[name], syndrome)
                predictions[name] = (L @ error) % 2
                failures[name][i, b] = np.any(predictions[name] != obs[i])
            agree[i, b] = np.array_equal(predictions["cpu"], predictions["gpu"])
            iterations[i, b] = gpu.last_iterations
        print(f"{'XZ'[b]} basis complete", flush=True)
        del methods, decode, gpu, cpu

    cpu_only = int((failures["cpu"] & ~failures["gpu"]).sum())
    gpu_only = int((failures["gpu"] & ~failures["cpu"]).sum())
    discordant = cpu_only + gpu_only
    pvalue = binomtest(gpu_only, discordant).pvalue if discordant else 1.0
    rows = []
    arrays = dict(agree=agree, gpu_iterations=iterations, rebuild_ms_xz=rebuild_ms)
    for name in failures:
        fail = failures[name]
        ms = seconds[name].sum(axis=1) * 1000
        rate, _ = per_cycle_rate(fail.any(axis=1), CYCLES)
        row = dict(decoder=name, p=args.p, p_loss=args.p_loss, loss_mode=args.loss_mode,
                   seed=args.seed, shots=args.shots, stop_nconv=args.stop_nconv,
                   gamma_ensemble_size=args.gamma_ensemble_size or 1,
                   fail_x=fail[:, 0].sum(), fail_z=fail[:, 1].sum(),
                   failed_pairs=fail.any(axis=1).sum(), ler_per_cycle=rate,
                   mean_ms=ms.mean(), p50_ms=np.median(ms), p99_ms=np.percentile(ms, 99),
                   max_ms=ms.max(), logical_agreement=agree.mean(),
                   cpu_only_wrong=cpu_only, gpu_only_wrong=gpu_only, sign_p=pvalue)
        rows.append(row)
        arrays[name + "_fail"] = fail
        arrays[name + "_seconds"] = seconds[name]
        print(row, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--p-loss", type=float, default=0)
    parser.add_argument("--loss-mode", choices=("unaware", "rows"), default="unaware")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--stop-nconv", type=int, choices=(1, 5), default=5)
    parser.add_argument("--gamma-ensemble-size", type=int, default=None,
                        help="GPU relay lanes raced in parallel (NVIDIA gamma_ensemble_size); default 1")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.seed < 8 or args.shots < 1 or args.out.suffix != ".csv":
        parser.error("use seed >= 8, shots >= 1, and a .csv output path")
    compare(args)


if __name__ == "__main__":
    main()
