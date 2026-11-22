"""Headline loss plot: logical error rate per cycle vs loss rate, loss ignored vs loss-reweighted priors.

  python plot_loss.py results/*.npz --out results/loss_sweep.png

Reads sweep.py outputs (any p_loss, loss_mode unaware/reweight; no-loss runs are drawn at p_loss = 0),
merges shards, and draws one panel per decoder with one line pair per physical error rate p.
Also writes a CSV next to the PNG with every point and its error bar.
"""
import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decoders import per_cycle_rate

CYCLES = 12
NAMES = {"relay5": "Relay-BP-5", "relay1": "Relay-BP-1", "bposd": "BP-OSD", "bp": "Plain BP"}
STYLE = {"unaware": ("--", "o", "loss ignored"), "reweight": ("-", "s", "loss-reweighted priors")}


def load(patterns):
    """One row per (decoder, p, p_loss, mode, seed). Shards of one run (same requested shot count and host)
    are merged; when several runs exist for the same point, the largest is used and the others are listed
    as skipped, so a 2k and a 20k run with the same seed are never silently mixed. Shot indices must be
    complete and unique."""
    runs = {}
    for path in sorted(set(sum((glob.glob(g) for g in patterns), []))):
        d = np.load(path)
        mode = str(d["loss_mode"]) if float(d["p_loss"]) > 0 else "none"
        m = re.search(r"_n(\d+)_shard", os.path.basename(path))
        n = int(m.group(1)) if m else len(d["shot_idx"])
        host = str(d["host"]) if "host" in d.files else "unknown"
        point = (str(d["decoder"]), float(d["p"]), float(d["p_loss"]), mode, int(d["seed"]))
        runs.setdefault(point, {}).setdefault((n, host), []).append((path, d))
    rows = []
    for (dec, p, pl, mode, seed), by_run in runs.items():
        (n, host), parts = max(by_run.items(), key=lambda kv: kv[0][0])
        idx = np.concatenate([d["shot_idx"] for _, d in parts])
        if len(idx) != n or len(np.unique(idx)) != n:
            raise SystemExit(f"incomplete or overlapping shards for {dec} p={p} p_loss={pl} {mode} seed {seed} n={n}")
        fail = np.concatenate([d["fail"] for _, d in parts])
        ms = np.concatenate([d["seconds"] for _, d in parts]).sum(axis=1) * 1e3
        rate, err = per_cycle_rate(fail.any(axis=1), CYCLES)
        skipped = [os.path.basename(f) for k, v in by_run.items() if k != (n, host) for f, _ in v]
        rows.append(dict(decoder=dec, p=p, p_loss=pl, mode=mode, seed=seed, shots=len(fail),
                         failed_pairs=int(fail.any(axis=1).sum()), ler=rate, ler_err=err, mean_ms=ms.mean(),
                         host=host, files=" ".join(os.path.basename(f) for f, _ in parts),
                         skipped=" ".join(skipped)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/loss_sweep.png")
    ap.add_argument("--seed", type=int, default=2, help="only plot runs with this seed")
    args = ap.parse_args()
    rows = [r for r in load(args.files) if r["seed"] == args.seed and r["mode"] in ("none", "unaware", "reweight")]
    if not rows:
        raise SystemExit("no matching sweep files")
    decoders = [d for d in NAMES if any(r["decoder"] == d for r in rows)]
    ps = sorted({r["p"] for r in rows})
    colors = dict(zip(ps, plt.cm.viridis(np.linspace(0.1, 0.8, len(ps)))))
    fig, axes = plt.subplots(1, len(decoders), figsize=(5.5 * len(decoders), 4.5), squeeze=False)
    for ax, dec in zip(axes[0], decoders):
        for p in ps:
            base = [r for r in rows if r["decoder"] == dec and r["p"] == p and r["mode"] == "none"]
            for mode, (ls, mk, label) in STYLE.items():
                pts = sorted(base + [r for r in rows if r["decoder"] == dec and r["p"] == p and r["mode"] == mode],
                             key=lambda r: r["p_loss"])
                if len(pts) < 2:
                    continue
                ax.errorbar([r["p_loss"] * 100 for r in pts], [r["ler"] for r in pts],
                            yerr=[r["ler_err"] for r in pts], ls=ls, marker=mk, color=colors[p], capsize=3,
                            label=f"p = {p}, {label}")
        ax.set_yscale("log")
        ax.set_xlabel("Loss rate per data qubit per cycle (%)")
        ax.set_ylabel("Logical error rate per cycle")
        ax.set_title(f"{NAMES[dec]}, Gross code [[144,12,12]]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    with open(args.out.rsplit(".", 1)[0] + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["decoder"], r["p"], r["p_loss"], r["mode"])))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
