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

Decoders: relay1capK (relay-BP-1 with at most K relay legs, then OSD-0 on its posteriors), bposd (kit settings), relay5 (kit relay-BP, stop_nconv=5), relay1 (stop_nconv=1), bp (plain
min-sum BP, 80 iterations). With --p-loss > 0, --loss-mode chooses how loss flags are used:
unaware (ignored), rows (flags as extra rows of H)
blind / shuffled (controls for reweight: same projected graph and per-shot rebuild, but priors ignore the
flags / use another shot's flags), or reweight (per-shot priors from the flags: fired -> P(column | loss), 1/2 when Y and Z merge; unfired -> ~0; H keeps the kit size). Output: results/<name>_shard<i>of<n>.npz with
shot indices, fail [X, Z], seconds [X, Z] and, for relay/bp, iterations [X, Z].
Each worker process decodes single-threaded; --workers defaults to all cores.
"""
import argparse
import glob
import json
import os
import platform
import time
from multiprocessing import Pool

import numpy as np
import stim
from ldpc import BpDecoder

from circuit import sample
from decoders import bposd_decoder, per_cycle_rate, relay_decoder
from loss_circuit import HERALD_COORDINATE, lossy_circuit, lossy_dem_matrices

CYCLES = 12
RELAY_OVERRIDES = {}  # set per job from --relay-overrides (gamma search); empty = kit settings


def parse_overrides(text):
    o = json.loads(text) if text else {}
    if "gamma_dist_interval" in o:
        o["gamma_dist_interval"] = tuple(o["gamma_dist_interval"])
    return o


class CappedRelay:
    """Relay-BP-1 with at most `cap` relay legs; if it has not converged by then, OSD-0 on its own
    posteriors (ldpc BpOsdDecoder with 1 BP iteration, priors set from the relay posterior ratios).
    decode_detailed returns success=False when the fallback was used."""

    def __init__(self, H, priors, cap):
        from types import SimpleNamespace
        from ldpc import BpOsdDecoder
        self.ns = SimpleNamespace
        self.relay = relay_decoder(H, priors, **(RELAY_OVERRIDES | {"stop_nconv": 1, "num_sets": cap}))
        self.osd = BpOsdDecoder(H, error_channel=list(priors), max_iter=1, bp_method="minimum_sum",
                                osd_method="OSD_0", osd_order=0)
        self.fallbacks = 0

    def decode_detailed(self, syndrome):
        r = self.relay.decode_detailed(syndrome)
        if r.success:
            return r
        self.fallbacks += 1
        llr = np.clip(np.nan_to_num(np.asarray(r.posterior_ratios, float)), -50, 50)
        q = np.clip(1.0 / (1.0 + np.exp(llr)), 1e-12, 1 - 1e-12)   # posterior_ratios = log(P0/P1)
        self.osd.update_channel_probs(q)
        return self.ns(decoding=np.asarray(self.osd.decode(syndrome), np.uint8), iterations=r.iterations,
                       success=False, posterior_ratios=r.posterior_ratios)


def build(name, H, priors):
    if name.startswith("relay1cap"):          # e.g. relay1cap200 = at most 200 relay legs, then OSD-0
        return CappedRelay(H, priors, int(name[len("relay1cap"):]))
    if name == "bposd":
        return bposd_decoder(H, priors)
    if name == "relay5":
        return relay_decoder(H, priors, **RELAY_OVERRIDES)
    if name == "relay1":
        return relay_decoder(H, priors, **(RELAY_OVERRIDES | {"stop_nconv": 1}))
    if name == "bp":
        return BpDecoder(H, error_channel=list(priors), max_iter=80, bp_method="minimum_sum", ms_scaling_factor=1.0)
    raise ValueError(f"unknown decoder {name}")


class Reweighted:
    """Loss-aware priors per shot: columns caused by a FIRED loss flag get prior 1/4 (the lost qubit is
    left maximally mixed), columns of UNFIRED flags get ~0, all others keep their DEM prior. Decodes on
    the check rows only, so H stays the kit's size. BP-OSD updates priors in place; relay-BP is rebuilt."""

    def __init__(self, name, H_full, priors, is_herald, p_loss, control="reweight",
                 fired_prior=None, unfired_prior=1e-9, flag_fn=0.0, flag_fp=0.0, noise_seed=0):
        self.control = control
        # robustness options (defaults = perfect flags, exact 1/2 prior):
        #   fired_prior: prior given to every column of a fired flag (None = prior/p_loss, i.e. ~1/2)
        #   unfired_prior: prior of columns whose flag did not fire
        #   flag_fn: each true flag is missed with this probability
        #   flag_fp: false flags per true flag on average (each unfired flag fires with prob flag_fp * p_loss)
        self.unfired_prior, self.flag_fn, self.flag_fp = unfired_prior, flag_fn, flag_fp
        self.p_loss, self.noise_seed = p_loss, noise_seed
        H_full = H_full.tocsr()
        self.check_rows, self.herald_rows = np.flatnonzero(~is_herald), np.flatnonzero(is_herald)
        H_check, H_herald = H_full[self.check_rows].tocsc(), H_full[self.herald_rows].tocsc()
        keep = np.flatnonzero(np.asarray(H_check.sum(axis=0)).ravel() > 0)
        self.H, self.priors, self.name = H_check[:, keep], priors[keep], name
        self.herald = H_herald[:, keep].tocsr()
        self.has_herald = np.asarray(self.herald.sum(axis=0)).ravel() > 0
        # P(column fired | its herald fired) = prior / p_loss: 1/2 when Y and Z merge into one column.
        self.given_loss = np.clip(self.priors / p_loss, 1e-9, 0.5)
        if fired_prior is not None:
            self.given_loss = np.full_like(self.given_loss, fired_prior)
        self.keep = keep
        self.decoder = build(name, self.H, self.priors)

    def priors_for(self, fired):
        if self.control == "blind":
            return self.priors.copy()  # unconditional DEM priors: herald identity ignored, same graph
        p = self.priors.copy()
        p[self.has_herald] = self.unfired_prior
        hit = np.asarray(self.herald[np.flatnonzero(fired)].sum(axis=0)).ravel() > 0
        p[hit] = self.given_loss[hit]
        return p

    def noisy(self, fired, shot):
        """Flags as an imperfect detector would report them; the same corruption for a shot in every run."""
        if not (self.flag_fn or self.flag_fp):
            return fired
        rng = np.random.default_rng([self.noise_seed, int(shot)])
        out = fired.astype(bool).copy()
        out[out & (rng.random(out.size) < self.flag_fn)] = False
        out[~fired.astype(bool) & (rng.random(out.size) < self.flag_fp * self.p_loss)] = True
        return out

    def decode_detailed(self, syndrome_full, flags_from=None, shot=0):
        """Relay-BP result object (for router features), plus loss features of this shot in self.loss_feats."""
        source = syndrome_full if flags_from is None else flags_from
        fired = self.noisy(source[self.herald_rows], shot)
        p = self.priors_for(fired)
        self.decoder = build(self.name, self.H, p)
        s = syndrome_full[self.check_rows]
        result = self.decoder.decode_detailed(s)
        self.last_priors = p
        hit = np.asarray(self.herald[np.flatnonzero(fired)].sum(axis=0)).ravel() > 0   # columns of fired flags
        lit = np.asarray(self.H[:, hit].T @ s).ravel() > 0                            # ...touching a lit check
        self.loss_feats = (float(np.sum(fired)), float(hit.sum()), float(lit.sum()))
        return result

    def decode(self, syndrome_full, flags_from=None, shot=0):
        source = syndrome_full if flags_from is None else flags_from
        p = self.priors_for(self.noisy(source[self.herald_rows], shot))
        if self.name == "bposd":
            self.decoder.update_channel_probs(p)
        else:
            self.decoder = build(self.name, self.H, p)
        return self.decoder.decode(syndrome_full[self.check_rows])


FEATURE_NAMES = ("converged", "weight", "loglik", "syndrome_weight", "posterior_abs_mean", "posterior_abs_p10")
ROUTER_FEATURE_NAMES = (*FEATURE_NAMES, "iterations")
LOSS_FEATURE_NAMES = ("flags_fired", "flag_columns", "flag_columns_on_lit_checks")


def relay_features(result, error, priors, syndrome):
    """Router features of one relay-BP decode (the last three match relay_router_pilot.py)."""
    on = error.astype(bool)
    loglik = np.log(priors[on] / (1 - priors[on])).sum() + np.log1p(-priors).sum()
    confidence = np.nan_to_num(np.abs(np.asarray(result.posterior_ratios)), nan=0).clip(0, 100)
    return (float(result.success), float(on.sum()), float(loglik), float(syndrome.sum()),
            float(confidence.mean()), float(np.percentile(confidence, 10)))


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
        use_heralds = args.loss_mode in ("rows", "reweight", "blind", "shuffled") and args.p_loss > 0
        H, L, priors, ids = lossy_dem_matrices(circuit, basis, use_heralds)
        coords = circuit.get_detector_coordinates()
        is_herald = np.array([coords[d][0] == HERALD_COORDINATE for d in ids])
        out.append((H, L, priors, det[:, ids], obs, is_herald))
    return out


def work(job):
    args, idx = job
    RELAY_OVERRIDES.clear()
    RELAY_OVERRIDES.update(parse_overrides(args.relay_overrides))
    fail = np.zeros((len(idx), 2), bool)
    secs = np.zeros((len(idx), 2))
    iters = np.zeros((len(idx), 2), np.int32)
    feats = np.full((len(idx), 2, len(FEATURE_NAMES)), np.nan)
    lfeats = np.full((len(idx), 2, len(LOSS_FEATURE_NAMES)), np.nan)
    for b, (H, L, priors, det, obs, is_herald) in enumerate(problems(args)):
        reweight = args.loss_mode in ("reweight", "blind", "shuffled") and args.p_loss > 0
        if reweight:
            rw = Reweighted(args.decoder, H, priors, is_herald, args.p_loss, control=args.loss_mode,
                            fired_prior=args.fired_prior, unfired_prior=args.unfired_prior,
                            flag_fn=args.flag_fn, flag_fp=args.flag_fp, noise_seed=1000 * args.seed + b)
            L = L.tocsc()[:, rw.keep]
        else:
            decoder = build(args.decoder, H, priors)
        for j, k in enumerate(idx):
            t = time.perf_counter()
            if reweight:
                # shuffled control: flags from another shot (fixed derangement k -> k + shots//2)
                other = det[(k + args.shots // 2) % args.shots] if args.loss_mode == "shuffled" else None
                if args.decoder.startswith("relay"):
                    result = rw.decode_detailed(det[k], flags_from=other, shot=k)
                    error, iters[j, b] = np.asarray(result.decoding, np.uint8), int(result.iterations)
                else:
                    error, iters[j, b] = np.asarray(rw.decode(det[k], flags_from=other, shot=k), np.uint8), -1
            else:
                if args.decoder.startswith("relay"):
                    result = decoder.decode_detailed(det[k])
                    error, iters[j, b] = np.asarray(result.decoding, np.uint8), int(result.iterations)
                else:
                    error, iters[j, b] = decode_one(decoder, args.decoder, det[k])
            secs[j, b] = time.perf_counter() - t
            fail[j, b] = ((L @ error) % 2 != obs[k]).any()
            if args.decoder.startswith("relay") and not reweight:
                feats[j, b] = relay_features(result, error, priors, det[k])
            elif args.decoder.startswith("relay"):
                feats[j, b] = relay_features(result, error, rw.last_priors, det[k][rw.check_rows])
                lfeats[j, b] = rw.loss_feats
    return idx, fail, secs, iters, feats, lfeats


def run(args):
    mine = np.arange(args.shots)[args.shard::args.nshards]
    chunks = [c for c in np.array_split(mine, args.workers * 4) if len(c)]
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.decoder}_p{args.p}_l{args.p_loss}_{args.loss_mode}_s{args.seed}_n{args.shots}"
    if args.tag:
        name += f"_{args.tag}"
    path = os.path.join(args.out, f"{name}_shard{args.shard}of{args.nshards}.npz")
    print(f"{name}: {len(mine)} shots on {args.workers} workers -> {path}", flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        parts = pool.map(work, [(args, c) for c in chunks])
    idx, fail, secs, iters, feats, lfeats = (np.concatenate(x) for x in zip(*parts))
    np.savez_compressed(path, decoder=args.decoder, p=args.p, p_loss=args.p_loss, loss_mode=args.loss_mode,
                        seed=args.seed, shots=args.shots, shard=args.shard, nshards=args.nshards,
                        shot_idx=idx, fail=fail, seconds=secs, iterations=iters, features=feats,
                        feature_names=np.asarray(FEATURE_NAMES), host=platform.node(),
                        loss_features=lfeats, loss_feature_names=np.asarray(LOSS_FEATURE_NAMES),
                        workers=args.workers, cycles=CYCLES, basis=np.asarray(["X", "Z"]),
                        stim_version=stim.__version__, tag=args.tag,
                        relay_overrides=json.dumps(parse_overrides(args.relay_overrides), default=list),
                        loss_options=json.dumps({"fired_prior": args.fired_prior, "unfired_prior": args.unfired_prior,
                                                 "flag_fn": args.flag_fn, "flag_fp": args.flag_fp}))
    rate, err = per_cycle_rate(fail.any(axis=1), CYCLES)
    print(f"done in {time.time() - start:.0f}s: failures {fail.sum(axis=0)}, per cycle {rate:.2e} +- {err:.1e}, "
          f"mean {secs.sum(axis=1).mean() * 1e3:.1f} ms", flush=True)


def summarize(args):
    groups = {}
    for path in sorted(set(sum((glob.glob(g) for g in args.files), []))):
        d = np.load(path)
        tag = str(d["tag"]) if "tag" in d.files else ""
        key = (str(d["decoder"]) + (f"[{tag}]" if tag else ""), float(d["p"]), float(d["p_loss"]),
               str(d["loss_mode"]), int(d["seed"]))
        groups.setdefault(key, []).append(d)
    print(f"{'decoder':16s} {'p':>6s} {'p_loss':>6s} {'mode':7s} {'shots':>6s} {'fail X':>6s} {'fail Z':>6s} "
          f"{'LER/cycle':>9s} {'+-':>8s} {'mean ms':>8s} {'p99 ms':>8s}")
    for key, parts in sorted(groups.items()):
        idx = np.concatenate([d["shot_idx"] for d in parts])
        if len(np.unique(idx)) != len(idx):
            print(f"warning: duplicate shots in {key}")
        fail = np.concatenate([d["fail"] for d in parts])
        ms = np.concatenate([d["seconds"] for d in parts]).sum(axis=1) * 1e3
        rate, err = per_cycle_rate(fail.any(axis=1), CYCLES)
        print(f"{key[0]:16s} {key[1]:6.3f} {key[2]:6.3f} {key[3]:7s} {len(idx):6d} {fail[:, 0].sum():6d} "
              f"{fail[:, 1].sum():6d} {rate:9.2e} {err:8.1e} {ms.mean():8.1f} {np.percentile(ms, 99):8.0f}")
    print("Timing is only comparable for files from the same machine (see the host field).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--decoder", required=True,
                   help="bposd, relay5, relay1, bp, or relay1capK (relay-BP-1 with at most K legs, then OSD-0)")
    r.add_argument("--p", type=float, default=0.004)
    r.add_argument("--p-loss", type=float, default=0.0)
    r.add_argument("--loss-mode", choices=["unaware", "rows", "reweight", "blind", "shuffled"], default="unaware")
    r.add_argument("--shots", type=int, default=10000)
    r.add_argument("--seed", type=int, default=2, help="2 = test seed; use 1 only for router training data")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--out", default="results")
    r.add_argument("--relay-overrides", default="",
                   help='JSON dict of relay-BP settings to change, e.g. \'{"gamma0": 0.1, "pre_iter": 120}\'')
    r.add_argument("--tag", default="", help="label appended to the output name (required with --relay-overrides)")
    r.add_argument("--fired-prior", type=float, default=None,
                   help="reweight: prior for columns of fired flags (default: exact, ~1/2)")
    r.add_argument("--unfired-prior", type=float, default=1e-9, help="reweight: prior for columns of unfired flags")
    r.add_argument("--flag-fn", type=float, default=0.0, help="reweight: probability a true flag is missed")
    r.add_argument("--flag-fp", type=float, default=0.0, help="reweight: false flags per true flag")
    s = sub.add_parser("summarize")
    s.add_argument("files", nargs="+")
    args = parser.parse_args()
    if args.cmd == "run" and not args.tag and (args.relay_overrides or args.fired_prior is not None or args.flag_fn
                                               or args.flag_fp or args.unfired_prior != 1e-9):
        parser.error("--relay-overrides / --fired-prior / --unfired-prior / --flag-fn / --flag-fp need --tag "
                     "so results don't overwrite default-setting files")
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
