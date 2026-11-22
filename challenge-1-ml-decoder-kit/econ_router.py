"""Cost-aware ("economic") router, replayed on paired relay-BP-1 / relay-BP-5 caches (Sean's question).

Escalate a basis shot to relay-BP-5 when  predicted net benefit  >  lambda x predicted extra milliseconds,
i.e. rank by benefit / cost; lambda (the threshold) is fixed on a calibration set at several escalation budgets.
  net benefit = int(relay-BP-1 wrong) - int(relay-BP-5 wrong)  (+1 rescue, -1 regression, 0 same)
  extra ms    = the full relay-BP-5 call (no warm start), predicted from relay-BP-1's features
Both models use only relay-BP-1's features (what exists at decision time). lambda is chosen per target mean time on a
CALIBRATION set (relay-BP-1 only, predicted costs), so the rule is a fixed per-shot threshold, then applied once to TEST.

Controls on the same test pairs: relay-BP-1 / relay-BP-5 alone, benefit-only ranking (constant cost), the simple
rule (not converged first, then most iterations), random escalation, and random decoder choice per pair at the same
mean time. Pair fails if either basis fails; pair time = X + Z.

  python econ_router.py --train-relay1 A1 --train-relay5 A5 --calibration C1 --test-relay1 T1 --test-relay5 T5 --out OUT
Training files must contain the SAME shots (same seed, same shot count, same machine): checked via syndrome weights.
"""
import argparse
import csv

import numpy as np

FRACS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)   # calibration escalation fractions (basis shots)


def load(path):
    d = np.load(path)
    o = np.argsort(d["shot_idx"])
    X = d["features"][o]
    it = d["iterations"][o].astype(float)
    parts = [X, it[..., None]]
    if "loss_features" in d.files and not np.isnan(d["loss_features"]).all():
        parts.append(d["loss_features"][o])                    # flags fired, their columns, columns on lit checks
    return dict(fail=d["fail"][o].astype(bool), t=d["seconds"][o] * 1e3, X=np.concatenate(parts, -1),
                host=str(d["host"]), it=it)


def design(D):
    X = D["X"].reshape(-1, D["X"].shape[-1])
    return np.column_stack([X, np.log1p(np.abs(X))])


def evaluate(r1, r5, esc):
    fail = np.where(esc, r5["fail"], r1["fail"])
    t = r1["t"] + np.where(esc, r5["t"], 0.0)
    pt = t.sum(1)
    return int(fail.any(1).sum()), pt.mean(), np.percentile(pt, 99), esc.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("train-relay1", "train-relay5", "calibration", "test-relay1", "test-relay5", "out"):
        ap.add_argument("--" + k, required=True)
    args = ap.parse_args()
    a1, a5, c1 = load(args.train_relay1), load(args.train_relay5), load(args.calibration)
    t1, t5 = load(args.test_relay1), load(args.test_relay5)
    for x, y, nm in ((a1, a5, "train"), (t1, t5, "test")):
        same = (x["X"][..., 3] == y["X"][..., 3]).mean()
        assert same > 0.999, f"{nm} relay1/relay5 files are not the same shots ({same:.1%} syndrome weights match)"
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xa = design(a1)
    benefit = a1["fail"].ravel().astype(int) - a5["fail"].ravel().astype(int)
    ben = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15).fit(Xa, benefit)
    cost = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15).fit(
        Xa, np.log(a5["t"].ravel() + 0.1))
    print(f"train: {benefit.size} basis shots, rescues {(benefit > 0).sum()}, regressions {(benefit < 0).sum()} "
          f"(host {a1['host']}); test host {t1['host']}")

    def scores(D):
        X = design(D)
        b = ben.predict(X).reshape(-1, 2)
        c = np.exp(cost.predict(X)).reshape(-1, 2) - 0.1
        return b, np.maximum(c, 1.0)

    bc, cc = scores(c1)
    bt, ct = scores(t1)
    # time scale: predicted costs are in training-machine ms; rescale calibration/test targets by relay-1 mean ratio
    m1, m5 = t1["t"].sum(1).mean(), t5["t"].sum(1).mean()
    rows = [dict(policy="relay-BP-1 alone", target="", failed_pairs=int(t1["fail"].any(1).sum()), mean_ms=m1,
                 p99_ms=np.percentile(t1["t"].sum(1), 99), esc=0.0),
            dict(policy="relay-BP-5 alone", target="", failed_pairs=int(t5["fail"].any(1).sum()), mean_ms=m5,
                 p99_ms=np.percentile(t5["t"].sum(1), 99), esc=1.0)]
    rng = np.random.default_rng(0)
    jit = lambda a: a + 1e-9 * rng.random(a.shape)                          # break ties
    conv_t, conv_c = t1["X"][..., 0] > 0.5, c1["X"][..., 0] > 0.5
    policies = {                                                             # higher score = escalate first
        "economic (benefit / cost)": (bt / ct, bc / cc),
        "benefit only (constant cost)": (bt, bc),
        "simple rule": (np.where(conv_t, t1["it"], 1e6 + t1["it"]) + rng.random(t1["it"].shape),   # random among ties
                        np.where(conv_c, c1["it"], 1e6 + c1["it"]) + rng.random(c1["it"].shape)),
    }
    for name, (st, sc) in policies.items():
        st, sc = jit(st), jit(sc)
        for frac in FRACS:
            thr = np.quantile(sc, 1 - frac)                                  # threshold fixed on calibration
            r = evaluate(t1, t5, st > thr)
            rows.append(dict(policy=name, target=frac, failed_pairs=r[0], mean_ms=r[1], p99_ms=r[2], esc=r[3]))
    # random decoder choice per pair, as a curve
    for f in np.linspace(0, 1, 11):
        draws = []
        for _ in range(300):
            m = rng.random(len(t1["fail"])) < f
            fl = np.where(m, t5["fail"].any(1), t1["fail"].any(1))
            tt = np.where(m, t5["t"].sum(1), t1["t"].sum(1))
            draws.append((fl.sum(), tt.mean(), np.percentile(tt, 99)))
        d = np.mean(draws, 0)
        rows.append(dict(policy="random decoder choice", target=round(f, 2), failed_pairs=d[0], mean_ms=d[1], p99_ms=d[2], esc=f))
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rc = [r for r in rows if r["policy"] == "random decoder choice"]
    xs, ys = [r["mean_ms"] for r in rc], [r["failed_pairs"] for r in rc]
    print(f"\n{'policy':30s} {'target':>6s} {'esc':>6s} {'failed':>7s} {'mean ms':>8s} {'p99 ms':>8s} {'random@same mean':>17s}")
    for r in rows:
        coin = np.interp(r["mean_ms"], xs, ys)
        print(f"{r['policy']:30s} {str(r['target']):>6s} {r['esc']:6.3f} {r['failed_pairs']:7.1f} {r['mean_ms']:8.1f} "
              f"{r['p99_ms']:8.0f} {coin:17.1f}")
    print("wrote", args.out + ".csv")


if __name__ == "__main__":
    main()
