"""Evaluate the frozen six-bit truth-table router on a relay-BP-1 / relay-BP-5 pair of sweep files.

  python truth_table_router.py results/06_router/caches/relay1_p0.005_l0.0_unaware_s14_n10000_shard0of1.npz \
                               results/06_router/caches/relay5_p0.005_l0.0_unaware_s14_n10000_shard0of1.npz

The table (results/07_truth_table/table/truth_table_router_frozen.json) was frozen before the seed-14 test data
existed. Nothing here is fitted to the files being evaluated. Both files must hold the same shots: same
machine, seed and shot count (checked).

Policies, all replayed from the recorded per-shot outcomes and decode times:
  fast    relay-BP-1 on every shot
  strong  relay-BP-5 on every shot
  table   relay-BP-1, then relay-BP-5 on basis shots whose 6-bit key is an escalate cell (time = both)
  simple  same number of escalations, chosen by the team's rule: not converged first, then most iterations
  coin    per X+Z pair, relay-BP-1 or relay-BP-5 chosen at random beforehand, with the probability set so the
          expected mean time equals the table's (expected failures, plus a 95% range over 2,000 draws)
Paired tests are exact two-sided binomial tests on pairs where exactly one of the two policies fails.
Timing caveat: these are the decode times recorded in the files (check `workers`); router overhead (a
64-entry lookup on 6 bits) is not included and is microseconds.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

TABLE = Path(__file__).resolve().parent / "results" / "truth_table" / "truth_table_router_frozen.json"
CYCLES = 12


def load(path):
    d = np.load(path, allow_pickle=False)
    order = np.argsort(d["shot_idx"])
    names = list(d["feature_names"])
    feat = d["features"][order]
    return dict(meta={k: d[k] for k in ("host", "seed", "shots", "p", "p_loss", "loss_mode", "workers")},
                shot_idx=d["shot_idx"][order], fail=d["fail"][order].astype(bool),
                seconds=d["seconds"][order], iterations=d["iterations"][order],
                converged=feat[..., names.index("converged")],
                syndrome_weight=feat[..., names.index("syndrome_weight")],
                posterior_abs_mean=feat[..., names.index("posterior_abs_mean")],
                posterior_abs_p10=feat[..., names.index("posterior_abs_p10")])


def keys(fast, cut):
    bits = [fast["converged"] < 0.5, fast["syndrome_weight"] > cut[0], fast["iterations"] > cut[1],
            fast["iterations"] > cut[2], fast["posterior_abs_mean"] < cut[3],
            fast["posterior_abs_p10"] < cut[4]]
    return sum(b.astype(int) << (5 - i) for i, b in enumerate(bits))


def per_cycle(failed_pairs, n):
    P = failed_pairs / n
    s = math.sqrt(P * (1 - P) / n)
    rate = 1 - (1 - P) ** (1 / CYCLES)
    return rate, (1 - (1 - min(P + s, 1)) ** (1 / CYCLES)) - rate


def summarize(name, routed, fast, strong, strong_only=False):
    fail = np.where(routed, strong["fail"], fast["fail"]).any(axis=1)
    sec = strong["seconds"] if strong_only else fast["seconds"] + routed * strong["seconds"]
    ms = sec.sum(axis=1) * 1000
    rate, err = per_cycle(fail.sum(), len(fail))
    return dict(policy=name, escalated=float(routed.mean()), failed_pairs=int(fail.sum()), ler=rate, ler_err=err,
                mean_ms=float(ms.mean()), p50_ms=float(np.median(ms)), p99_ms=float(np.percentile(ms, 99)),
                max_ms=float(ms.max())), fail


def paired(a, b):
    x, y = int((a & ~b).sum()), int((b & ~a).sum())
    return x, y, (binomtest(x, x + y).pvalue if x + y else 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("relay1", type=Path)
    parser.add_argument("relay5", type=Path)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--out", type=Path, help="CSV path for the summary rows")
    args = parser.parse_args()
    spec = json.loads(args.table.read_text())
    fast, strong = load(args.relay1), load(args.relay5)
    for k in ("host", "seed", "shots", "p", "p_loss", "loss_mode"):
        assert np.array_equal(fast["meta"][k], strong["meta"][k]), f"files differ in {k}: not the same shots"
    assert np.array_equal(fast["shot_idx"], strong["shot_idx"])
    m = fast["meta"]
    print(f"host {m['host']}, p {float(m['p'])}, loss {float(m['p_loss'])} ({m['loss_mode']}), seed {int(m['seed'])}, "
          f"{int(m['shots'])} pairs, workers {int(m['workers'])}")

    escalate = np.array(spec["escalate"], bool)
    k = keys(fast, spec["cut"])
    table = escalate[k]
    n_esc = int(table.sum())
    order = np.lexsort((np.arange(table.size), -fast["iterations"].astype(float).ravel(), fast["converged"].ravel()))
    simple = np.zeros(table.size, bool)
    simple[order[:n_esc]] = True
    simple = simple.reshape(table.shape)

    rows, fails = [], {}
    for name, routed, strong_only in [("fast (relay-BP-1)", np.zeros_like(table), False),
                                      ("strong (relay-BP-5)", np.ones_like(table), True),
                                      ("truth table (frozen)", table, False),
                                      ("simple rule, same escalations", simple, False)]:
        row, f = summarize(name, routed, fast, strong, strong_only)
        rows.append(row)
        fails[name] = f

    t, fa, st = rows[2], rows[0], rows[1]
    q = min(max((t["mean_ms"] - fa["mean_ms"]) / (st["mean_ms"] - fa["mean_ms"]), 0.0), 1.0)
    rng = np.random.default_rng(0)
    fast_f, strong_f = fails["fast (relay-BP-1)"], fails["strong (relay-BP-5)"]
    draws = [int(np.where(rng.random(len(fast_f)) < q, strong_f, fast_f).sum()) for _ in range(2000)]
    expect = (1 - q) * fast_f.sum() + q * strong_f.sum()
    rows.append(dict(policy=f"coin flip at equal mean (q = {q:.3f})", escalated=q, failed_pairs=round(expect, 1),
                     ler=per_cycle(expect, len(fast_f))[0], ler_err=float("nan"), mean_ms=t["mean_ms"],
                     p50_ms=float("nan"), p99_ms=float("nan"), max_ms=float("nan"),
                     range95=(int(np.percentile(draws, 2.5)), int(np.percentile(draws, 97.5)))))

    print(f"\n{'policy':38s} {'escalated':>9s} {'failed':>8s} {'LER/cycle':>18s} {'mean ms':>9s} {'p99 ms':>9s}")
    for r in rows:
        extra = f"  95% range {r['range95'][0]}-{r['range95'][1]}" if "range95" in r else ""
        ler = f"{r['ler']:.2e} +- {r['ler_err']:.1e}" if not math.isnan(r["ler_err"]) else f"{r['ler']:.2e}"
        p99 = f"{r['p99_ms']:9.0f}" if not math.isnan(r["p99_ms"]) else f"{'-':>9s}"
        print(f"{r['policy']:38s} {r['escalated']:9.2%} {r['failed_pairs']:8} {ler:>18s} {r['mean_ms']:9.1f} {p99}{extra}")

    tab = fails["truth table (frozen)"]
    print("\nPaired (pairs where exactly one fails: table only / other only, exact p):")
    for other in ("fast (relay-BP-1)", "strong (relay-BP-5)", "simple rule, same escalations"):
        x, y, p = paired(tab, fails[other])
        print(f"  table vs {other:32s} {x:4d} / {y:4d}   p = {p:.3g}")
    used = sorted({format(int(c), '06b') for c in np.unique(k[table])})
    print(f"\nEscalate cells that fired: {used}; basis shots escalated: {n_esc} of {table.size}")

    if args.out:
        import csv
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["policy", "escalated", "failed_pairs", "ler", "ler_err", "mean_ms",
                                               "p50_ms", "p99_ms", "max_ms", "host", "p", "seed", "shots",
                                               "workers"], extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow(r | {"host": str(m["host"]), "p": float(m["p"]), "seed": int(m["seed"]),
                                "shots": int(m["shots"]), "workers": int(m["workers"])})
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
