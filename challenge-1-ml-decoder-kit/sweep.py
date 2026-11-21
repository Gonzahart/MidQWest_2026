"""Shardable decoder sweep with per-shot results, for running on many machines at once.

Every machine samples the SAME shots (same seed) and decodes only its slice of them, so slices from
different machines can be merged into one result, and every decoder sees identical shots.

  # whole job on one machine
  python sweep.py run --decoder relay5 --p 0.004 --shots 10000
  # the same job split over 3 machines (run one line on each)
  python sweep.py run --decoder relay5 --p 0.004 --shots 10000 --shard 0 --nshards 3
  python sweep.py run --decoder relay5 --p 0.004 --shots 10000 --shard 1 --nshards 3
  python sweep.py run --decoder relay5 --p 0.004 --shots 10000 --shard 2 --nshards 3
  # merge every file in results/ and print one line per (decoder, p, p_loss, loss mode)
  python sweep.py summarize results/*.npz

Decoders: bposd (kit settings), relay5 (kit relay-BP, stop_nconv=5), relay1 (stop_nconv=1), bp (plain
min-sum BP, 80 iterations). With --p-loss > 0, --loss-mode chooses how loss flags are used:
unaware (ignored), rows (flags as extra rows of H)
or reweight (per-shot priors from the flags: fired -> P(column | loss), 1/2 when Y and Z merge; unfired -> ~0; H keeps the kit size). Output: results/<name>_shard<i>of<n>.npz with
shot indices, fail [X, Z], seconds [X, Z] and, for relay/bp, iterations [X, Z].
Each worker process decodes single-threaded; --workers defaults to all cores.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import stim
from ldpc import BpDecoder

from circuit import sample
from decoders import bposd_decoder, per_cycle_rate, relay_decoder
from loss_circuit import HERALD_COORDINATE, lossy_circuit, lossy_dem_matrices

CYCLES = 12


def build(name, H, priors):
    if name == "bposd":
        return bposd_decoder(H, priors)
    if name == "relay5":
        return relay_decoder(H, priors)
    if name == "relay1":
        return relay_decoder(H, priors, stop_nconv=1)
    if name == "bp":
        return BpDecoder(H, error_channel=list(priors), max_iter=80, bp_method="minimum_sum", ms_scaling_factor=1.0)
    raise ValueError(f"unknown decoder {name}")


class Reweighted:
    """Loss-aware priors per shot: columns caused by a FIRED loss flag get prior 1/4 (the lost qubit is
    left maximally mixed), columns of UNFIRED flags get ~0, all others keep their DEM prior. Decodes on
    the check rows only, so H stays the kit's size. BP-OSD updates priors in place; relay-BP is rebuilt."""

    def __init__(self, name, H_full, priors, is_herald, p_loss):
        H_full = H_full.tocsr()
        self.check_rows, self.herald_rows = np.flatnonzero(~is_herald), np.flatnonzero(is_herald)
        H_check, H_herald = H_full[self.check_rows].tocsc(), H_full[self.herald_rows].tocsc()
        keep = np.flatnonzero(np.asarray(H_check.sum(axis=0)).ravel() > 0)
        self.H, self.priors, self.name = H_check[:, keep], priors[keep], name
        self.herald = H_herald[:, keep].tocsr()
        self.has_herald = np.asarray(self.herald.sum(axis=0)).ravel() > 0
        # P(column fired | its herald fired) = prior / p_loss: 1/2 when Y and Z merge into one column.
        self.given_loss = np.clip(self.priors / p_loss, 1e-9, 0.5)
        self.keep = keep
        self.decoder = build(name, self.H, self.priors)

    def priors_for(self, fired):
        p = self.priors.copy()
        p[self.has_herald] = 1e-9
        hit = np.asarray(self.herald[np.flatnonzero(fired)].sum(axis=0)).ravel() > 0
        p[hit] = self.given_loss[hit]
        return p

    def decode(self, syndrome_full):
        p = self.priors_for(syndrome_full[self.herald_rows])
        if self.name == "bposd":
            self.decoder.update_channel_probs(p)
        else:
            self.decoder = build(self.name, self.H, p)
        return self.decoder.decode(syndrome_full[self.check_rows])


def solution_features(error, priors, success):
    """(converged, Hamming weight, log-likelihood of the correction under the priors)."""
    on = error.astype(bool)
    llr = np.log(priors[on] / (1 - priors[on])).sum()
    return float(success), float(on.sum()), float(llr + np.log1p(-priors).sum())


def decode_one(decoder, name, syndrome):
    """(error vector, iterations or -1)."""
    if name.startswith("relay"):
        r = decoder.decode_detailed(syndrome)
        return np.asarray(r.decoding, np.uint8), int(r.iterations)
    error = decoder.decode(syndrome)
    return np.asarray(error, np.uint8), int(getattr(decoder, "iter", -1))


def problems(args):
    out = []
    for k, basis in enumerate("XZ"):
        circuit = lossy_circuit(basis, args.p, args.p_loss)
        det, obs = sample(circuit, args.shots, args.seed + k)
        use_heralds = args.loss_mode in ("rows", "reweight") and args.p_loss > 0
        H, L, priors, ids = lossy_dem_matrices(circuit, basis, use_heralds)
        coords = circuit.get_detector_coordinates()
        is_herald = np.array([coords[d][0] == HERALD_COORDINATE for d in ids])
        out.append((H, L, priors, det[:, ids], obs, is_herald))
    return out


def work(job):
    args, idx = job
    fail = np.zeros((len(idx), 2), bool)
    secs = np.zeros((len(idx), 2))
    iters = np.zeros((len(idx), 2), np.int32)
    feats = np.full((len(idx), 2, 3), np.nan)   # converged, solution weight, log-likelihood
    for b, (H, L, priors, det, obs, is_herald) in enumerate(problems(args)):
        reweight = args.loss_mode == "reweight" and args.p_loss > 0
        if reweight:
            rw = Reweighted(args.decoder, H, priors, is_herald, args.p_loss)
            L = L.tocsc()[:, rw.keep]
        else:
            decoder = build(args.decoder, H, priors)
        for j, k in enumerate(idx):
            t = time.perf_counter()
            if reweight:
                error, iters[j, b] = np.asarray(rw.decode(det[k]), np.uint8), -1
            else:
                error, iters[j, b] = decode_one(decoder, args.decoder, det[k])
                if args.decoder.startswith("relay"):
                    satisfied = not ((H @ error) % 2 != det[k]).any()
                    feats[j, b] = solution_features(error, priors, satisfied)
            secs[j, b] = time.perf_counter() - t
            fail[j, b] = ((L @ error) % 2 != obs[k]).any()
    return idx, fail, secs, iters, feats


def run(args):
    mine = np.arange(args.shots)[args.shard::args.nshards]
    chunks = [c for c in np.array_split(mine, args.workers * 4) if len(c)]
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.decoder}_p{args.p}_l{args.p_loss}_{args.loss_mode}_s{args.seed}_n{args.shots}"
    path = os.path.join(args.out, f"{name}_shard{args.shard}of{args.nshards}.npz")
    print(f"{name}: {len(mine)} shots on {args.workers} workers -> {path}", flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        parts = pool.map(work, [(args, c) for c in chunks])
    idx, fail, secs, iters, feats = (np.concatenate(x) for x in zip(*parts))
    np.savez_compressed(path, decoder=args.decoder, p=args.p, p_loss=args.p_loss, loss_mode=args.loss_mode,
                        seed=args.seed, shots=args.shots, shard=args.shard, nshards=args.nshards,
                        shot_idx=idx, fail=fail, seconds=secs, iterations=iters, features=feats,
                        feature_names=np.asarray(["converged", "weight", "loglik"]), host=os.uname().nodename,
                        workers=args.workers, cycles=CYCLES, basis=np.asarray(["X", "Z"]),
                        stim_version=stim.__version__)
    rate, err = per_cycle_rate(fail.any(axis=1), CYCLES)
    print(f"done in {time.time() - start:.0f}s: failures {fail.sum(axis=0)}, per cycle {rate:.2e} +- {err:.1e}, "
          f"mean {secs.sum(axis=1).mean() * 1e3:.1f} ms", flush=True)


def summarize(args):
    groups = {}
    for path in sorted(set(sum((glob.glob(g) for g in args.files), []))):
        d = np.load(path)
        key = (str(d["decoder"]), float(d["p"]), float(d["p_loss"]), str(d["loss_mode"]), int(d["seed"]))
        groups.setdefault(key, []).append(d)
    print(f"{'decoder':8s} {'p':>6s} {'p_loss':>6s} {'mode':7s} {'shots':>6s} {'fail X':>6s} {'fail Z':>6s} "
          f"{'LER/cycle':>9s} {'+-':>8s} {'mean ms':>8s} {'p99 ms':>8s}")
    for key, parts in sorted(groups.items()):
        idx = np.concatenate([d["shot_idx"] for d in parts])
        if len(np.unique(idx)) != len(idx):
            print(f"warning: duplicate shots in {key}")
        fail = np.concatenate([d["fail"] for d in parts])
        ms = np.concatenate([d["seconds"] for d in parts]).sum(axis=1) * 1e3
        rate, err = per_cycle_rate(fail.any(axis=1), CYCLES)
        print(f"{key[0]:8s} {key[1]:6.3f} {key[2]:6.3f} {key[3]:7s} {len(idx):6d} {fail[:, 0].sum():6d} "
              f"{fail[:, 1].sum():6d} {rate:9.2e} {err:8.1e} {ms.mean():8.1f} {np.percentile(ms, 99):8.0f}")
    print("Timing is only comparable for files from the same machine (see the host field).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--decoder", choices=["bposd", "relay5", "relay1", "bp"], required=True)
    r.add_argument("--p", type=float, default=0.004)
    r.add_argument("--p-loss", type=float, default=0.0)
    r.add_argument("--loss-mode", choices=["unaware", "rows", "reweight"], default="unaware")
    r.add_argument("--shots", type=int, default=10000)
    r.add_argument("--seed", type=int, default=2, help="2 = test seed; use 1 only for router training data")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--out", default="results")
    s = sub.add_parser("summarize")
    s.add_argument("files", nargs="+")
    args = parser.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
