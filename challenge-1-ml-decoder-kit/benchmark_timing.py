"""Measure decoder latency and peak process memory on a single machine."""
import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys

# Set before importing numerical libraries or spawning workers.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from decoders import per_cycle_rate
from evaluate import run_slice

DEFAULT_DECODERS = ("relay1+H", "relay+H", "bp>relay+H")


def cpu_model():
    if sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        result = subprocess.run(["system_profiler", "SPHardwareDataType", "-json"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            items = json.loads(result.stdout).get("SPHardwareDataType", [])
            if items and items[0].get("chip_type"):
                return items[0]["chip_type"]
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.partition(":")[0].strip() == "model name":
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def load_live_model(path, seed):
    from relay_router_pilot import FEATURES

    with np.load(path) as artifact:
        if artifact["feature_names"].tolist() != list(FEATURES):
            raise ValueError("model features do not match the integrated router")
        if abs(int(artifact["train_seed"]) - seed) < 2:
            raise ValueError("training and timing seeds must be disjoint in both bases")
        return artifact["center"], artifact["scale"], artifact["weights"], 0, artifact["thresholds"]


def measure(job):
    import resource

    decoder, args, _, _ = job
    if decoder == "live-relay":
        from relay_router_pilot import live_compare

        result = live_compare(args, args.model)
        row = dict(fail_x=result["failed_x"], fail_z=result["failed_z"],
                   failed_pairs=result["failed_pairs"], ler_per_cycle=result["ler_per_cycle"],
                   mean_ms=result["mean_ms"], p99_ms=result["p99_ms"],
                   fallback_rate=result["fallback_rate"])
    else:
        failed, seconds, escalated, _, _ = run_slice(job)
        failed_pair = failed.any(axis=1)
        rate, _ = per_cycle_rate(failed_pair, 12)
        elapsed = seconds.sum(axis=1) * 1000
        row = dict(fail_x=int(failed[:, 0].sum()), fail_z=int(failed[:, 1].sum()),
                   failed_pairs=int(failed_pair.sum()), ler_per_cycle=rate,
                   mean_ms=elapsed.mean(), p99_ms=np.percentile(elapsed, 99),
                   fallback_rate=escalated / (2 * args.shots))

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return dict(row, peak_rss_mib=peak_bytes / 1024**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.005)
    parser.add_argument("--p-loss", type=float, default=0)
    parser.add_argument("--shots", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--decoders", nargs="+", default=DEFAULT_DECODERS)
    parser.add_argument("--live-model", type=Path, help="trained NPZ from the live integration")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        parser.error("peak RSS measurement requires macOS or Linux")
    if args.shots < 1:
        parser.error("shots must be positive")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")

    if "live-relay" in args.decoders:
        if args.live_model is None or args.p_loss != 0:
            parser.error("live-relay requires --live-model and zero qubit loss")
        try:
            from relay_router_pilot import live_compare  # verify teammate integration is present
            args.model = load_live_model(args.live_model, args.seed)
        except (ImportError, KeyError, ValueError, OSError) as error:
            parser.error(f"cannot load live router: {error}")
        args.test_shots, args.test_seed = args.shots, args.seed

    args.out_npz = None
    machine = {"cpu": cpu_model(), "platform": platform.platform(),
               "logical_cpus": os.cpu_count(), "workers": 1, "math_threads": 1,
               "python": platform.python_version()}
    rows = []
    jobs = [(decoder, args, 0, 1) for decoder in args.decoders]
    # A fresh child per decoder prevents earlier runs contaminating later RSS peaks.
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        for decoder, metrics in zip(args.decoders, pool.imap(measure, jobs), strict=True):
            rows.append(dict(decoder=decoder, p=args.p, p_loss=args.p_loss, seed=args.seed,
                             shot_pairs=args.shots, **metrics, **machine))
            print(f"{decoder}: {metrics['failed_pairs']}/{args.shots} failed, "
                  f"mean {metrics['mean_ms']:.1f} ms, p99 {metrics['p99_ms']:.1f} ms, "
                  f"peak RSS {metrics['peak_rss_mib']:.1f} MiB", flush=True)
    with args.out.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
