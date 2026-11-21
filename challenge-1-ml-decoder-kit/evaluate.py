"""Compare decoders on the Gross code with heralded qubit loss.

  python evaluate.py --p 0.004 --p-loss 0.01 --shots 2000
  python evaluate.py --p 0.004 --p-loss 0.01 --shots 2000 --decoders bp>relay+H relay+H

Decoder names:
  bposd, relay          the kit's baselines; loss looks like extra Pauli noise
  bposd+H, relay+H      the same decoders with the loss heralds as extra rows of H
  bp>relay+H            hierarchical: fast plain BP first; if BP does not converge to a
                        solution of the syndrome, the shot is passed to relay-BP (with heralds)
  bp>bposd+H            the same with BP-OSD as the second stage
  bp+H, relay1+H       standalone fast BP and relay-BP-1 with heralds

Both memory experiments (X and Z) are sampled with the same circuit noise, decoded separately,
and paired shot by shot as in the kit. Output: failures, logical error rate per cycle, mean and
p99 decode time per pair, and for hierarchical decoders the share of shots sent to stage 2.
Results are also appended to results.csv for plotting.
With --out-npz, fail, seconds, iterations, and fallback_iterations are indexed by
(decoder, shot, basis). Iterations counts the standalone decoder or first-stage BP.
Fallback_iterations is -1 when no fallback ran. BP-OSD counts only its BP iterations.
Export timings include the diagnostics used to collect counts. The decoders, shot,
and basis arrays label the axes. The stored seed is for X memory; Z uses seed + 1.
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import stim
from ldpc import BpDecoder

from circuit import sample
from decoders import bposd_decoder, per_cycle_rate, relay_decoder
from loss_circuit import lossy_circuit, lossy_dem_matrices

DEFAULT_DECODERS = ["bposd", "bposd+H", "relay", "relay+H", "bp>relay+H"]
FAST_BP = {"max_iter": 30, "bp_method": "minimum_sum", "ms_scaling_factor": 0.9}


class Hierarchical:
    """Stage 1: plain min-sum BP. Accept its answer when it satisfies the syndrome, else stage 2."""

    def __init__(self, H, priors, second):
        self.first = BpDecoder(H, error_channel=list(priors), **FAST_BP)
        self.second = second
        self.escalated = 0

    def decode(self, syndrome, *, record_iterations=False):
        if record_iterations:
            error, self.iterations = decode_with_iterations(self.first, syndrome)
            self.fallback_iterations = -1
        else:
            error = self.first.decode(syndrome)
        if self.first.converge:
            return error
        self.escalated += 1
        if record_iterations:
            error, self.fallback_iterations = decode_with_iterations(self.second, syndrome)
            return error
        return self.second.decode(syndrome)


def decode_with_iterations(decoder, syndrome):
    """Return a correction and its BP iteration count, excluding any OSD work."""
    if hasattr(decoder, "decode_detailed"):
        result = decoder.decode_detailed(syndrome)
        return result.decoding, result.iterations
    error = decoder.decode(syndrome)
    # ldpc short-circuits a zero syndrome and can retain the previous shot's count.
    return error, decoder.iter if syndrome.any() else 0


def build(name, H, priors):
    if name.startswith("bp>"):
        second = {"relay+H": relay_decoder, "bposd+H": bposd_decoder}[name[3:]]
        return Hierarchical(H, priors, second(H, priors))
    base = name.removesuffix("+H")
    if base == "bp":
        return BpDecoder(H, error_channel=list(priors), **FAST_BP)
    if base == "relay1":
        return relay_decoder(H, priors, stop_nconv=1)
    return {"bposd": bposd_decoder, "relay": relay_decoder}[base](H, priors)


def problems(p, p_loss, shots, seed, use_heralds):
    """[(H, L, priors, syndromes, observables)] for the X and the Z memory experiment."""
    out = []
    for k, basis in enumerate("XZ"):
        circuit = lossy_circuit(basis, p, p_loss)
        det, obs = sample(circuit, shots, seed + k)   # same seed -> same shots for every decoder
        H, L, priors, ids = lossy_dem_matrices(circuit, basis, use_heralds)
        out.append((H, L, priors, det[:, ids], obs))
    return out


def run_slice(job):
    """Decode one worker's share of the shots of both experiments with one decoder."""
    name, args, first, step = job
    failed, seconds, iterations, fallback_iterations, escalated = [], [], [], [], 0
    for H, L, priors, det, obs in problems(args.p, args.p_loss, args.shots, args.seed, name.endswith("+H")):
        decoder = build(name, H, priors)
        f, s = np.zeros(len(det[first::step]), bool), np.zeros(len(det[first::step]))
        counts = np.empty(len(f), dtype=np.int32)
        fallback_counts = np.full(len(f), -1, dtype=np.int32)
        for k, (syndrome, truth) in enumerate(zip(det[first::step], obs[first::step], strict=True)):
            t = time.perf_counter()
            if args.out_npz is None:
                error = decoder.decode(syndrome)
            elif isinstance(decoder, Hierarchical):
                error = decoder.decode(syndrome, record_iterations=True)
                counts[k] = decoder.iterations
                fallback_counts[k] = decoder.fallback_iterations
            else:
                error, counts[k] = decode_with_iterations(decoder, syndrome)
            error = np.asarray(error, np.uint8)
            s[k] = time.perf_counter() - t
            f[k] = ((L @ error) % 2 != truth).any()
        failed.append(f)
        seconds.append(s)
        if args.out_npz is not None:
            iterations.append(counts)
            fallback_iterations.append(fallback_counts)
        escalated += getattr(decoder, "escalated", 0)
    counts = np.column_stack(iterations) if iterations else None
    fallback_counts = np.column_stack(fallback_iterations) if fallback_iterations else None
    return np.column_stack(failed), np.column_stack(seconds), escalated, counts, fallback_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--p-loss", type=float, default=0.01, help="loss probability per data qubit per cycle")
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--decoders", nargs="+", default=DEFAULT_DECODERS)
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--out-npz", type=Path, help="save per-shot outcomes, timings, and iterations to a new NPZ file")
    args = parser.parse_args()
    if args.out_npz is not None:
        if args.cycles != 12:
            parser.error("the exported Gross memory circuits use 12 cycles")
        if args.out_npz.resolve() == Path(args.csv).resolve():
            parser.error("--out-npz and --csv must use different paths")
        if args.out_npz.exists():
            parser.error(f"NPZ output already exists: {args.out_npz}")

    print(f"{args.shots} shots at p={args.p}, p_loss={args.p_loss}, {args.cycles} cycles, failures [X Z]")
    new_file = not os.path.exists(args.csv)
    per_shot = []
    with Pool(args.workers) as pool, open(args.csv, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["decoder", "p", "p_loss", "shots", "fail_X", "fail_Z", "ler_per_cycle", "ler_err",
                             "mean_ms", "p99_ms", "escalated_frac"])
        for name in args.decoders:
            jobs = [(name, args, k, args.workers) for k in range(args.workers)]
            results = pool.map(run_slice, jobs)
            failed = np.empty((args.shots, 2), dtype=bool)
            seconds = np.empty((args.shots, 2))
            if args.out_npz is not None:
                iterations = np.empty((args.shots, 2), dtype=np.int32)
                fallback_iterations = np.empty_like(iterations)
            for worker, (worker_failed, worker_seconds, _, counts, fallback_counts) in enumerate(results):
                positions = slice(worker, None, args.workers)
                failed[positions] = worker_failed
                seconds[positions] = worker_seconds
                if args.out_npz is not None:
                    iterations[positions] = counts
                    fallback_iterations[positions] = fallback_counts
            ms = seconds.sum(axis=1) * 1e3
            if args.out_npz is not None:
                per_shot.append((failed, seconds, iterations, fallback_iterations))
            esc = sum(r[2] for r in results) / (2 * args.shots)
            rate, err = per_cycle_rate(failed.any(axis=1), args.cycles)
            extra = f"  to stage 2 {esc:.1%}" if name.startswith("bp>") else ""
            print(f"{name:12s} failures {failed.sum(axis=0)}  per cycle {rate:.2e} +- {err:.1e}  "
                  f"mean {ms.mean():.1f} ms  p99 {np.percentile(ms, 99):.0f} ms{extra}", flush=True)
            writer.writerow([name, args.p, args.p_loss, args.shots, *failed.sum(axis=0), rate, err,
                             ms.mean(), np.percentile(ms, 99), esc])
            fh.flush()

    if args.out_npz is not None:
        failed_by_decoder, seconds_by_decoder, counts_by_decoder, fallback_by_decoder = zip(*per_shot, strict=True)
        with args.out_npz.open("xb") as output:
            np.savez_compressed(
                output,
                decoders=np.asarray(args.decoders),
                shot=np.arange(args.shots),
                basis=np.asarray(["X", "Z"]),
                fail=np.stack(failed_by_decoder),
                seconds=np.stack(seconds_by_decoder),
                iterations=np.stack(counts_by_decoder),
                fallback_iterations=np.stack(fallback_by_decoder),
                p=args.p,
                p_loss=args.p_loss,
                seed=args.seed,
                cycles=args.cycles,
                workers=args.workers,
                host=os.uname().nodename,
                stim_version=stim.__version__,
                shots=args.shots,
            )


if __name__ == "__main__":
    main()
