"""Data for the ML early exit: decode every shot with relay-BP's FIRST LEG only (80 iterations, fixed gamma0),
and for shots that leg leaves unconverged record BOTH possible continuations:
  - keep going: a full relay-BP-1 decode (what the kit's relay-BP-1 would do; repeats the first leg)
  - exit now:   OSD-0 on the first leg's own posteriors (~tens of ms)
Every branch's logical failure and time is saved, so any exit policy can be replayed exactly afterwards
(exit_router.py). No loss here (p_loss = 0); same circuits and sampling as sweep.py.

  python exit_data.py --p 0.005 --shots 10000 --seed 1  --workers 20   # training data
  python exit_data.py --p 0.005 --shots 10000 --seed 13 --workers 20   # fresh test data

Output: results/exit/exitdata_p{p}_s{seed}_n{shots}.npz with arrays of shape (shots, 2) [X, Z]:
  conv_a, fail_a, t_a              first leg: converged?, logical failure, seconds
  fail_relay, t_relay, it_relay    keep going (NaN / -1 where the first leg converged)
  fail_osd, t_osd                  exit to OSD-0 (NaN where the first leg converged)
  feats (shots, 2, F)              first-leg features, names in feature_names
"""
import argparse
import os
import platform
import time
from multiprocessing import Pool

import numpy as np
import stim
from ldpc import BpOsdDecoder

from circuit import sample
from decoders import relay_decoder
from loss_circuit import lossy_circuit, lossy_dem_matrices

FEATURES = ("syndrome_weight", "residual_weight", "weight", "loglik", "post_abs_mean", "post_abs_p10",
            "post_abs_p1", "n_uncertain", "n_flipped_sign")


def first_leg_features(r, H, priors, syndrome):
    e = np.asarray(r.decoding, np.uint8)
    on = e.astype(bool)
    residual = (H @ e + syndrome) % 2                 # checks the first leg's guess leaves unsatisfied
    llr = np.nan_to_num(np.asarray(r.posterior_ratios, float), nan=0.0)
    conf = np.abs(llr).clip(0, 100)
    prior_llr = np.log((1 - priors) / priors)
    loglik = np.log(priors[on] / (1 - priors[on])).sum() + np.log1p(-priors).sum()
    return (float(syndrome.sum()), float(residual.sum()), float(on.sum()), float(loglik), float(conf.mean()),
            float(np.percentile(conf, 10)), float(np.percentile(conf, 1)), float((conf < 1.0).sum()),
            float(((llr < 0) != (prior_llr < 0)).sum()))


def work(job):
    args, idx = job
    n = len(idx)
    out = {k: np.full((n, 2), np.nan) for k in ("fail_a", "t_a", "fail_relay", "t_relay", "fail_osd", "t_osd")}
    out["conv_a"] = np.zeros((n, 2), bool)
    out["it_relay"] = np.full((n, 2), -1, np.int32)
    feats = np.full((n, 2, len(FEATURES)), np.nan)
    for b, basis in enumerate("XZ"):
        circuit = lossy_circuit(basis, args.p, 0.0)
        det, obs = sample(circuit, args.shots, args.seed + b)
        H, L, priors, ids = lossy_dem_matrices(circuit, basis, False)
        H, det = H.tocsr(), det[:, ids]
        first = relay_decoder(H, priors, stop_nconv=1, num_sets=0)
        full = relay_decoder(H, priors, stop_nconv=1)
        osd = BpOsdDecoder(H, error_channel=list(priors), max_iter=1, bp_method="minimum_sum",
                           osd_method="OSD_0", osd_order=0)
        wrong = lambda e, k: bool(((L @ np.asarray(e, np.uint8)) % 2 != obs[k]).any())
        for j, k in enumerate(idx):
            s = det[k]
            t = time.perf_counter()
            r = first.decode_detailed(s)
            out["t_a"][j, b] = time.perf_counter() - t
            out["conv_a"][j, b] = bool(r.success)
            out["fail_a"][j, b] = wrong(r.decoding, k)
            feats[j, b] = first_leg_features(r, H, priors, s)
            if r.success:
                continue
            t = time.perf_counter()
            rr = full.decode_detailed(s)
            out["t_relay"][j, b] = time.perf_counter() - t
            out["fail_relay"][j, b] = wrong(rr.decoding, k)
            out["it_relay"][j, b] = int(rr.iterations)
            t = time.perf_counter()
            llr = np.clip(np.nan_to_num(np.asarray(r.posterior_ratios, float)), -50, 50)
            osd.update_channel_probs(np.clip(1.0 / (1.0 + np.exp(llr)), 1e-12, 1 - 1e-12))
            eo = osd.decode(s)
            out["t_osd"][j, b] = time.perf_counter() - t
            out["fail_osd"][j, b] = wrong(eo, k)
    return idx, out, feats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--seed", type=int, required=True, help="1 = training; never 2 (test cache seed)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/exit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"exitdata_p{args.p}_s{args.seed}_n{args.shots}.npz")
    chunks = [c for c in np.array_split(np.arange(args.shots), args.workers * 4) if len(c)]
    print(f"{path}: {args.shots} shots on {args.workers} workers", flush=True)
    start = time.time()
    with Pool(args.workers) as pool:
        parts = pool.map(work, [(args, c) for c in chunks])
    idx = np.concatenate([p[0] for p in parts])
    data = {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
    feats = np.concatenate([p[2] for p in parts])
    np.savez_compressed(path, shot_idx=idx, feats=feats, feature_names=np.asarray(FEATURES), p=args.p,
                        seed=args.seed, host=platform.node(), workers=args.workers, stim_version=stim.__version__,
                        **data)
    na = ~data["conv_a"]
    pair_fail_relay1 = np.where(na, data["fail_relay"], data["fail_a"]).astype(bool).any(1).sum()
    print(f"done in {time.time() - start:.0f}s: first leg unconverged on {na.mean():.1%} of basis shots; "
          f"of those relay fails {np.nanmean(data['fail_relay'][na]):.1%}, OSD-0 fails {np.nanmean(data['fail_osd'][na]):.1%}; "
          f"relay-BP-1 failed pairs {pair_fail_relay1}", flush=True)


if __name__ == "__main__":
    main()
