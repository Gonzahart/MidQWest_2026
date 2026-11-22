"""Coarse relay-BP memory-strength (gamma) search, accuracy only (Track B).

Candidate 0 is the kit's setting, decoded on the same shots, so every candidate is compared paired.
Seed 8 (Stim seeds 8/9) is the validation seed: disjoint from train (1), test (2/3), calibration (4/5)
and the seed-10 BlueQubit run. Never run this on seed 2.

  python gamma_search.py plan --n 16                    # writes gamma_candidates.json
  python gamma_search.py run --p 0.005 --shots 2000     # runs every candidate not already done
  python gamma_search.py rank --p 0.005 --shots 2000    # paired comparison vs candidate 0

Timings from a shared 20-worker box are not single-threaded benchmarks: use failures and iterations.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

from decoders import RELAY_SETTINGS, per_cycle_rate

CANDIDATES = "gamma_candidates.json"
SEED = 8


def plan(args):
    rng = np.random.default_rng(args.rng)
    n = args.n - 1
    # Latin hypercube over gamma0, interval centre, interval width; pre_iter from a small set
    u = (rng.permuted(np.tile(np.arange(n), (3, 1)), axis=1).T + rng.random((n, 3))) / n
    lo, hi = RELAY_SETTINGS["gamma_dist_interval"]
    cands = [{"tag": "g00", "overrides": {}}]  # kit default
    for i, (a, b, c) in enumerate(u, start=1):
        gamma0 = 0.05 + 0.25 * a
        centre = 0.05 + 0.30 * b
        width = 0.40 + 0.80 * c
        cands.append({"tag": f"g{i:02d}", "overrides": {
            "gamma0": round(gamma0, 4),
            "gamma_dist_interval": [round(centre - width / 2, 4), round(centre + width / 2, 4)],
            "pre_iter": int(rng.choice([40, 80, 120]))}})
    json.dump({"kit": {"gamma0": RELAY_SETTINGS["gamma0"], "interval": [lo, hi],
                       "pre_iter": RELAY_SETTINGS["pre_iter"]}, "candidates": cands}, open(CANDIDATES, "w"), indent=1)
    print(f"wrote {len(cands)} candidates (incl. kit default g00) to {CANDIDATES}")


def path(args, tag):
    return os.path.join(args.out, f"{args.decoder}_p{args.p}_l0.0_unaware_s{SEED}_n{args.shots}_{tag}_shard0of1.npz")


def run(args):
    cands = json.load(open(CANDIDATES))["candidates"]
    if args.only:
        cands = [c for c in cands if c["tag"] in args.only]
    for c in cands:
        if os.path.exists(path(args, c["tag"])):
            print(f"skip {c['tag']} (done)", flush=True)
            continue
        cmd = [sys.executable, "sweep.py", "run", "--decoder", args.decoder, "--p", str(args.p),
               "--shots", str(args.shots), "--seed", str(SEED), "--workers", str(args.workers),
               "--out", args.out, "--tag", c["tag"], "--relay-overrides", json.dumps(c["overrides"])]
        print(c["tag"], c["overrides"], flush=True)
        subprocess.run(cmd, check=True)


def rank(args):
    cands = {c["tag"]: c["overrides"] for c in json.load(open(CANDIDATES))["candidates"]}
    base = np.load(path(args, "g00"))
    bf = base["fail"].any(1)
    rows = []
    for tag, o in cands.items():
        if not os.path.exists(path(args, tag)):
            continue
        d = np.load(path(args, tag))
        assert (d["shot_idx"] == base["shot_idx"]).all()
        f = d["fail"].any(1)
        better, worse = int((bf & ~f).sum()), int((f & ~bf).sum())
        # paired sign test z for (better - worse)
        z = (better - worse) / np.sqrt(max(better + worse, 1))
        rate, err = per_cycle_rate(f, 12)
        rows.append((int(f.sum()), tag, better, worse, z, rate, err, d["iterations"].mean(), d["seconds"].sum(1).mean() * 1e3, o))
    print(f"{'tag':4s} {'fails':>5s} {'+fix':>4s} {'-brk':>4s} {'z':>5s} {'LER/cyc':>9s} {'iters':>7s} {'ms*':>7s}  overrides")
    for r in sorted(rows):
        print(f"{r[1]:4s} {r[0]:5d} {r[2]:4d} {r[3]:4d} {r[4]:5.1f} {r[5]:9.2e} {r[7]:7.0f} {r[8]:7.0f}  {r[9]}")
    print(f"{len(rows)} candidates on {len(bf)} pairs, seed {SEED}. z > 0 = fewer failures than kit (g00), paired.\n"
          "* ms from a shared multi-worker box: relative only. Promote the top ~5 to 20k pairs before trusting z.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--rng", type=int, default=0)
    for name in ("run", "rank"):
        q = sub.add_parser(name)
        q.add_argument("--p", type=float, default=0.005)
        q.add_argument("--shots", type=int, default=2000)
        q.add_argument("--decoder", choices=["relay5", "relay1"], default="relay5")
        q.add_argument("--out", default="results_gamma")
        q.add_argument("--workers", type=int, default=os.cpu_count())
        q.add_argument("--only", nargs="*", help="run only these tags (e.g. promotion: --only g03 g07)")
    a = ap.parse_args()
    {"plan": plan, "run": run, "rank": rank}[a.cmd](a)


if __name__ == "__main__":
    main()
