"""ML early exit for relay-BP-1, evaluated by exact replay of exit_data.py files.

Every shot first runs relay-BP's first leg (80 iterations). If that leg converges we are done. If not, the policy
chooses: KEEP GOING (full relay-BP-1) or EXIT NOW (OSD-0 on the first leg's posteriors). exit_data.py recorded both
outcomes and times for every unconverged shot, so each policy is replayed exactly.

Policies, each at a range of "keep going" fractions q (fraction of unconverged basis shots that continue):
  ml-logistic / ml-boosted   continue the q shots the model rates most likely to need relay (label below)
  rule                       continue the q shots with the fewest unsatisfied checks after the first leg
  random                     continue a random q of them
plus relay-BP-1 alone (always keep going) and always-exit. Label = "keep going helps": OSD-0 wrong, relay right.
Thresholds come from the TRAINING scores (seed 1); test once on a fresh seed.

  python exit_router.py --train results/exit/exitdata_p0.005_s1_n10000.npz --test results/exit/exitdata_p0.005_s13_n10000.npz \
      --out results/exit/exit_p0.005

Writes <out>.csv and <out>.png (failed pairs vs mean ms, and vs p99 ms). Times are per X+Z pair and include the
first leg; "keep going" pays the first leg twice (relay-BP has no warm start), relay-BP-1 alone does not.
"""
import argparse
import csv

import numpy as np

QS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load(path):
    d = np.load(path)
    D = {k: d[k] for k in d.files}
    D["un"] = ~D["conv_a"]
    D["help"] = D["un"] & (np.nan_to_num(D["fail_osd"]) > 0) & (np.nan_to_num(D["fail_relay"], nan=1) == 0)
    return D


def outcome(D, cont):
    """cont: bool (shots, 2), True = keep going (only matters where the first leg did not converge)."""
    un = D["un"]
    fail = np.where(un, np.where(cont, D["fail_relay"], D["fail_osd"]), D["fail_a"]).astype(bool)
    t = D["t_a"] + np.where(un, np.where(cont, D["t_relay"], D["t_osd"]), 0.0)
    pair_t = t.sum(1) * 1e3
    return int(fail.any(1).sum()), pair_t.mean(), np.percentile(pair_t, 99)


def relay1_alone(D):
    fail = np.where(D["un"], D["fail_relay"], D["fail_a"]).astype(bool)
    t = np.where(D["un"], D["t_relay"], D["t_a"]).sum(1) * 1e3
    return int(fail.any(1).sum()), t.mean(), np.percentile(t, 99)


def features(D):
    X = D["feats"].reshape(-1, D["feats"].shape[-1])
    return np.column_stack([X, np.log1p(np.abs(X))])


def fit(train):
    m = train["un"].ravel()
    X, y = features(train)[m], train["help"].ravel()[m]
    models = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models["ml-logistic"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X, y)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["ml-boosted"] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                              max_leaf_nodes=15).fit(X, y)
    except ImportError:
        pass
    return models, X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args()
    train, test = load(args.train), load(args.test)
    models, Xtr, ytr = fit(train)
    m_te = test["un"].ravel()
    Xte = features(test)
    rows = []
    add = lambda pol, q, actual, r: rows.append(dict(policy=pol, q=q, cont_frac=round(actual, 4), failed_pairs=r[0],
                                                     mean_ms=round(r[1], 1), p99_ms=round(r[2], 1)))
    n_pairs = len(test["fail_a"])
    print(f"train {args.train}: {train['un'].sum()} unconverged basis shots, {train['help'].sum()} where keep-going helps")
    print(f"test  {args.test}: {n_pairs} pairs, {test['un'].sum()} unconverged basis shots "
          f"({test['un'].mean():.1%}), keep-going helps on {test['help'].sum()}")
    add("relay-BP-1 alone", 1.0, 1.0, relay1_alone(test))
    rng = np.random.default_rng(args.random_seed)
    resid = test["feats"][..., list(test["feature_names"]).index("residual_weight")]
    resid_tr = train["feats"][..., list(train["feature_names"]).index("residual_weight")][train["un"]]
    for name, model in models.items():
        s_tr = model.predict_proba(Xtr)[:, 1]
        s_te = np.zeros(m_te.size)
        s_te[m_te] = model.predict_proba(Xte[m_te])[:, 1]
        s_te = s_te.reshape(-1, 2)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(test["help"].ravel()[m_te], s_te.ravel()[m_te])
            print(f"{name}: test AUROC for 'keep going helps' = {auc:.3f}")
        except Exception:
            pass
        for q in QS:
            thr = np.quantile(s_tr, 1 - q) if 0 < q < 1 else (np.inf if q == 0 else -np.inf)
            cont = (s_te >= thr) & test["un"]
            add(name, q, cont.sum() / max(test["un"].sum(), 1), outcome(test, cont))
    for q in QS:
        thr = np.quantile(resid_tr, q) if 0 < q < 1 else (-np.inf if q == 0 else np.inf)
        cont = (resid <= thr) & test["un"] if q > 0 else np.zeros_like(test["un"])
        add("rule (fewest unsatisfied checks)", q, cont.sum() / max(test["un"].sum(), 1), outcome(test, cont))
        draws = [outcome(test, (rng.random(test["un"].shape) < q) & test["un"]) for _ in range(200)]
        add("random", q, q, tuple(np.mean([d[i] for d in draws]) for i in range(3)))
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'policy':34s} {'q':>5s} {'cont':>6s} {'failed':>7s} {'mean ms':>8s} {'p99 ms':>8s}")
    for r in rows:
        print(f"{r['policy']:34s} {r['q']:5.2f} {r['cont_frac']:6.3f} {r['failed_pairs']:7.1f} {r['mean_ms']:8.1f} {r['p99_ms']:8.1f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, key, lab in ((axes[0], "mean_ms", "mean ms per pair"), (axes[1], "p99_ms", "p99 ms per pair")):
            for pol in dict.fromkeys(r["policy"] for r in rows):
                pts = [r for r in rows if r["policy"] == pol]
                ax.plot([r[key] for r in pts], [r["failed_pairs"] for r in pts], "o-" if len(pts) > 1 else "*",
                        label=pol, ms=4 if len(pts) > 1 else 12)
            ax.set_xlabel(lab)
            ax.set_ylabel(f"failed pairs (of {n_pairs})")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle("Early exit after relay-BP's first leg: keep going (relay-BP-1) vs exit to OSD-0")
        fig.tight_layout()
        fig.savefig(args.out + ".png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    print("wrote", args.out + ".csv")


if __name__ == "__main__":
    main()
