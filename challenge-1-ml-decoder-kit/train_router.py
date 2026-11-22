"""Train and evaluate a relay-BP-1 -> relay-BP-5 router from sweep.py caches (replay, no re-decoding).

Every shot is decoded by relay-BP-1. The router looks at features of relay-BP-1's answer and decides
whether to also run relay-BP-5 and use its answer instead. Each escalated basis shot costs
relay-BP-1 time + relay-BP-5 time.

  python train_router.py \
      --train results/relay1_p0.005_l0.0_unaware_s1_n20000_shard*.npz \
      --test-relay1 results/relay1_p0.004_l0.0_unaware_s2_n10000_shard*.npz \
      --test-relay5 results/relay5_p0.004_l0.0_unaware_s2_n10000_shard*.npz \
      --out results/router_p0.004

Training: relay-BP-1 on seed 1 (never the test seed). Label = relay-BP-1 got that basis shot wrong.
Models: logistic regression (numpy, always) and gradient-boosted trees (if scikit-learn is installed).
Escalation budgets (default 0.5-20% of basis shots) are turned into score thresholds on the TRAINING
scores, so nothing is tuned on the test set. Baselines at each budget: random routing, and a simple
rule (not converged first, then most iterations). Also reported: relay-BP-1 alone, relay-BP-5 alone
and the perfect router (escalate exactly relay-BP-1's wrong shots; an upper bound, not a result).

Optional calibration sets basis-specific thresholds. Matched Relay-BP-5 training shots
switch the positive label to a first-stage failure rescued by Relay-BP-5.

Outputs: <out>.csv (one row per policy and budget), <out>.png (error rate vs mean time) and
<out>_decisions_<model>_<budget>.npz in the format router_report.py --decisions expects
(shot, basis, route + run metadata). Replay times exclude feature extraction and router inference.
"""
import argparse
import csv
import glob

import numpy as np
from scipy.stats import rankdata

from decoders import per_cycle_rate

from sweep import ROUTER_FEATURE_NAMES

FEATURES = list(ROUTER_FEATURE_NAMES)
META = ("p", "p_loss", "seed", "shots", "cycles", "stim_version")


def load(patterns, decoder, *, allow_partial_labels=False):
    """Merge shards of one sweep run: (meta, shot ids, fail [n,2], seconds [n,2], X [n,2,f])."""
    paths = sorted(set(sum((glob.glob(p) for p in patterns), [])))
    if not paths:
        raise SystemExit(f"no files match {patterns}")
    parts = [np.load(p) for p in paths]
    for d, path in zip(parts, paths):
        if str(d["decoder"]) != decoder:
            raise SystemExit(f"{path}: expected decoder {decoder}, got {d['decoder']}")
    meta = {k: parts[0][k].item() for k in META if k in parts[0]}
    for d, path in zip(parts, paths):
        if any(d[k].item() != meta[k] for k in meta):
            raise SystemExit(f"{path}: shards come from different runs")
    idx = np.concatenate([d["shot_idx"] for d in parts])
    order = np.argsort(idx)
    if len(np.unique(idx)) != len(idx):
        raise SystemExit(f"{decoder}: duplicate shot ids across shards")
    fail = np.concatenate([d["fail"] for d in parts])[order]
    if any("evaluated" in d for d in parts):
        if not allow_partial_labels:
            raise SystemExit(f"{decoder}: partially evaluated archives are for rescue training only")
        evaluated = np.concatenate([d["evaluated"] if "evaluated" in d
                                    else np.ones_like(d["fail"]) for d in parts])[order]
        fail = np.ma.array(fail, mask=~evaluated)
    secs = np.concatenate([d["seconds"] for d in parts])[order]
    feats = None
    if "features" in parts[0] and parts[0]["features"].shape[-1] >= 6:
        names = list(parts[0]["feature_names"])
        f = np.concatenate([d["features"] for d in parts])[order]
        it = np.concatenate([d["iterations"] for d in parts])[order][..., None].astype(float)
        cols = [f[..., names.index(n)] for n in FEATURES[:-1]]
        feats = np.concatenate([np.stack(cols, axis=-1), it], axis=-1)
    return meta, idx[order], fail, secs, feats


def flat(feats, fail=None):
    x = feats.reshape(-1, feats.shape[-1])
    return x if fail is None else (x, fail.reshape(-1))


class Logistic:
    """L2-regularised logistic regression with class weighting (positives are rare)."""

    def fit(self, x, y, l2=1e-2, iters=500, lr=0.5):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        z = (x - self.mu) / self.sd
        w_pos = 0.5 / max(y.mean(), 1e-6)
        w_neg = 0.5 / max(1 - y.mean(), 1e-6)
        sw = np.where(y, w_pos, w_neg)
        self.w, self.b = np.zeros(z.shape[1]), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(z @ self.w + self.b)))
            g = sw * (p - y)
            self.w -= lr * (z.T @ g / len(y) + l2 * self.w)
            self.b -= lr * g.mean()
        return self

    def score(self, x):
        return (x - self.mu) / self.sd @ self.w + self.b


class Boosted:
    def fit(self, x, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                                random_state=0).fit(x, y)
        return self

    def score(self, x):
        return self.m.predict_proba(x)[:, 1]


def auroc(score, y):
    ranks = rankdata(score)
    pos = y.astype(bool)
    if pos.sum() == 0 or pos.sum() == len(y):
        return float("nan")
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def replay(route, r1, r5, cycles):
    fail = np.where(route, r5["fail"], r1["fail"])
    ms = (r1["secs"] + np.where(route, r5["secs"], 0)).sum(axis=1) * 1e3
    rate, err = per_cycle_rate(fail.any(axis=1), cycles)
    return dict(failed_pairs=int(fail.any(axis=1).sum()), ler=rate, ler_err=err, mean_ms=ms.mean(),
                p99_ms=np.percentile(ms, 99), escalated=float(route.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="relay1 sweep files, training seed")
    ap.add_argument("--train-relay5", nargs="+", help="matched Relay-BP-5 subset for rescue labels")
    ap.add_argument("--calibration", nargs="+", help="separate Relay-BP-1 cache at the test noise level")
    ap.add_argument("--save-models", action="store_true", help="export trained models and locked thresholds")
    ap.add_argument("--test-relay1", nargs="+", required=True)
    ap.add_argument("--test-relay5", nargs="+", required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--random-seed", type=int, default=1)
    args = ap.parse_args()

    tr_meta, tr_idx, tr_fail, _, tr_feats = load(args.train, "relay1")
    te_meta, te_idx, f1, s1, te_feats = load(args.test_relay1, "relay1")
    m5, idx5, f5, s5, _ = load(args.test_relay5, "relay5")
    if tr_feats is None or te_feats is None:
        raise SystemExit("relay1 files lack router features: rerun with the current sweep.py")
    if tr_meta["seed"] == te_meta["seed"]:
        raise SystemExit("train and test use the same seed")
    if {k: te_meta[k] for k in ("p", "p_loss", "seed", "shots")} != {k: m5[k] for k in ("p", "p_loss", "seed", "shots")} \
            or not np.array_equal(te_idx, idx5):
        raise SystemExit("relay1 and relay5 test files must cover the same shots (same p, seed, shots)")
    for key in ("cycles", "stim_version"):
        if te_meta.get(key) != m5.get(key):
            raise SystemExit(f"Relay-BP test caches differ in {key}")
    train_labels = tr_fail
    if args.train_relay5:
        train5_meta, train5_idx, train5_fail, _, _ = load(args.train_relay5, "relay5", allow_partial_labels=True)
        if train5_meta != tr_meta:
            raise SystemExit("rescue labels require the original training sample batch metadata, including total shots")
        positions = np.searchsorted(tr_idx, train5_idx)
        if not len(positions) or (positions >= len(tr_idx)).any() or not np.array_equal(tr_idx[positions], train5_idx):
            raise SystemExit("Relay-BP-5 training shot IDs are not a subset of Relay-BP-1 training IDs")
        tr_feats = tr_feats[positions]
        if np.any(tr_fail[positions] & np.ma.getmaskarray(train5_fail)):
            raise SystemExit("a failed Relay-BP-1 training shot is missing its fallback label")
        train_labels = tr_fail[positions] & ~np.ma.filled(train5_fail, False)

    calibration_feats = tr_feats
    calibration_seed = tr_meta["seed"]
    if args.calibration:
        cal_meta, _, _, _, calibration_feats = load(args.calibration, "relay1")
        if calibration_feats is None:
            raise SystemExit("calibration cache lacks router features")
        if cal_meta["seed"] in (tr_meta["seed"], te_meta["seed"]):
            raise SystemExit("calibration must use a different base seed from train and test")
        for key in ("p", "p_loss", "cycles", "stim_version"):
            if cal_meta.get(key) != te_meta.get(key):
                raise SystemExit(f"calibration and test caches differ in {key}")
        calibration_seed = cal_meta["seed"]
        if abs(calibration_seed - te_meta["seed"]) == 1:
            print("Calibration and test share one numeric seed across opposite bases; "
                  "thresholds are calibrated within each basis.")
    cycles = int(te_meta.get("cycles", 12))
    r1, r5 = dict(fail=f1, secs=s1), dict(fail=f5, secs=s5)

    x_tr, y_tr = flat(tr_feats, train_labels)
    if not y_tr.any() or y_tr.all():
        raise SystemExit("training labels must contain both classes")
    x_te, y_te = flat(te_feats, f1)
    print(f"train: {len(y_tr)} basis shots, {int(y_tr.sum())} positive labels (p={tr_meta['p']}, seed {tr_meta['seed']})")
    print(f"test:  {len(y_te)} basis shots, {int(y_te.sum())} relay-BP-1 errors (p={te_meta['p']}, seed {te_meta['seed']})")

    models = {"logistic": Logistic().fit(x_tr, y_tr)}
    try:
        models["boosted"] = Boosted().fit(x_tr, y_tr)
    except ImportError:
        print("scikit-learn not installed: skipping boosted trees")
    # Simple rule: not converged first, then more iterations (no learning).
    conv, its = FEATURES.index("converged"), FEATURES.index("iterations")
    rule = lambda x: (1 - x[:, conv]) * 1e6 + x[:, its]

    rows = []
    shape = f1.shape
    for name, r in [("relay-BP-1 alone", np.zeros(shape, bool)), ("relay-BP-5 alone", None),
                    ("perfect router (upper bound)", f1.astype(bool))]:
        if r is None:
            rate, err = per_cycle_rate(f5.any(axis=1), cycles)
            ms = s5.sum(axis=1) * 1e3
            row = dict(failed_pairs=int(f5.any(axis=1).sum()), ler=rate, ler_err=err, mean_ms=ms.mean(),
                       p99_ms=np.percentile(ms, 99), escalated=float("nan"))
        else:
            row = replay(r, r1, r5, cycles)
        rows.append(dict(policy=name, budget="", auroc="", **row))

    rng = np.random.default_rng(args.random_seed)
    matched_rng = np.random.default_rng(args.random_seed)
    scorers = {n: m.score for n, m in models.items()} | {"simple rule": rule}
    if args.calibration:
        simple_scores = rule(x_te).reshape(shape)
        tie_breaks = np.random.default_rng(args.random_seed).random(shape)
        simple_ranks = np.argsort(np.lexsort((tie_breaks, -simple_scores), axis=0), axis=0)
    for name, scorer in scorers.items():
        s_cal = scorer(flat(calibration_feats)).reshape(calibration_feats.shape[:2])
        s_te = scorer(x_te).reshape(shape)
        auc = auroc(s_te.ravel(), y_te)
        for budget in args.budgets:
            thresholds = (np.quantile(s_cal, 1 - budget, axis=0) if args.calibration
                          else np.repeat(np.quantile(s_cal, 1 - budget), 2))
            route = s_te > thresholds
            rows.append(dict(policy=name, budget=budget, auroc=round(auc, 4), **replay(route, r1, r5, cycles)))
            # Shuffle each basis separately to preserve its realized escalation count.
            matched = np.column_stack([matched_rng.permutation(column) for column in route.T])
            rows.append(dict(policy=f"random matched to {name}", budget=budget, auroc="",
                             **replay(matched, r1, r5, cycles)))
            if name in models:
                controls = {"random_route": matched}
                if args.calibration:
                    # Offline baseline at the learned policy's exact realized count.
                    simple_matched = simple_ranks < route.sum(axis=0)
                    controls["simple_route"] = simple_matched
                    rows.append(dict(policy=f"simple matched to {name}", budget=budget, auroc="",
                                     **replay(simple_matched, r1, r5, cycles)))
                np.savez_compressed(f"{args.out}_decisions_{name}_{budget}.npz", shot=te_idx, basis=np.asarray(["X", "Z"]),
                                    route=route, **controls, **{k: te_meta[k] for k in META if k in te_meta})
                if args.save_models:
                    from joblib import dump

                    candidate = models[name]
                    scorer_state = ({"center": candidate.mu, "scale": candidate.sd,
                                     "weights": candidate.w, "intercept": candidate.b}
                                    if name == "logistic" else {"estimator": candidate.m})
                    dump(dict(kind=name, features=FEATURES, thresholds=thresholds,
                              train_seed=tr_meta["seed"], calibration_seed=calibration_seed,
                              p=te_meta["p"], p_loss=te_meta["p_loss"], budget=budget,
                              **scorer_state), f"{args.out}_model_{name}_{budget}.joblib")
    for budget in args.budgets:
        k = int(round(budget * f1.size))
        flat_route = np.zeros(f1.size, bool)
        flat_route[rng.choice(f1.size, k, replace=False)] = True
        rows.append(dict(policy="random", budget=budget, auroc="", **replay(flat_route.reshape(shape), r1, r5, cycles)))

    with open(f"{args.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'policy':30s} {'budget':>6s} {'AUROC':>6s} {'esc':>6s} {'fail':>5s} {'LER/cycle':>9s} {'+-':>8s} {'mean ms':>8s}")
    for r in rows:
        print(f"{r['policy']:30s} {str(r['budget']):>6s} {str(r['auroc']):>6s} {r['escalated']:6.3f} {r['failed_pairs']:5d} "
              f"{r['ler']:9.2e} {r['ler_err']:8.1e} {r['mean_ms']:8.1f}")
    plot(rows, te_meta, f"{args.out}.png")
    print(f"\nwrote {args.out}.csv, {args.out}.png, decision files. Replay times exclude router inference.")


def plot(rows, meta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in dict.fromkeys(r["policy"] for r in rows):
        pts = [r for r in rows if r["policy"] == name]
        x, y, e = [r["mean_ms"] for r in pts], [r["ler"] for r in pts], [r["ler_err"] for r in pts]
        style = dict(marker="o", ls="-") if len(pts) > 1 else dict(marker="*", ms=14, ls="none")
        ax.errorbar(x, y, yerr=e, capsize=3, label=name, **style)
    ax.set_xlabel("Mean decode time per X/Z pair (ms, replay)")
    ax.set_ylabel("Logical error rate per cycle")
    ax.set_yscale("log")
    ax.set_title(f"Relay-BP-1 → relay-BP-5 routing, p = {meta['p']}, seed {meta['seed']} ({meta['shots']} pairs)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()
