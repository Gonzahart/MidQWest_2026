"""Compare Gross-code decoders and routing policies from paired NPZ caches."""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decoders import per_cycle_rate

RUN_KEYS = ("p", "p_loss", "seed", "shots", "cycles", "stim_version")
TIMING_KEYS = ("host", "workers")
BASIS = ["X", "Z"]
HIERARCHICAL = "bp>relay+H"


@dataclass
class Batch:
    shot: np.ndarray
    fail: np.ndarray
    seconds: np.ndarray
    route: np.ndarray | None = None


def metadata(data, keys):
    return {key: data[key].item() for key in keys}


def read_cache(path):
    with np.load(path) as data:
        if data["basis"].tolist() != BASIS:
            raise ValueError(f"{path}: expected X/Z basis ordering")
        settings = metadata(data, RUN_KEYS + TIMING_KEYS)
        if "shot_idx" in data:
            name = str(data["decoder"])
            mode = str(data["loss_mode"])
            if name in ("relay1", "relay5") and (mode == "rows" or settings["p_loss"] == 0):
                label = {"relay1": "relay1+H", "relay5": "relay+H"}[name]
                return settings, [(label, Batch(data["shot_idx"], data["fail"], data["seconds"]))]
            raise ValueError(f"{path}: expected a relay1 or relay5 sweep with matching herald handling")

        entries = []
        for i, name in enumerate(data["decoders"].tolist()):
            if name in ("relay1+H", "relay+H", "bp+H", HIERARCHICAL):
                route = data["fallback_iterations"][i] >= 0 if name == HIERARCHICAL else None
                entries.append((name, Batch(data["shot"], data["fail"][i], data["seconds"][i], route)))
        return settings, entries


def align(parts, shots, name):
    """Sort shards by shot ID and reject overlaps, gaps, or invalid measurements."""
    ids = np.concatenate([part.shot for part in parts])
    order = np.argsort(ids)
    if ids.dtype.kind not in "iu" or not np.array_equal(ids[order], np.arange(shots)):
        raise ValueError(f"{name}: expected every shot ID exactly once")
    fail = np.concatenate([part.fail for part in parts])[order]
    seconds = np.concatenate([part.seconds for part in parts])[order]
    if fail.shape != (shots, 2) or fail.dtype != np.bool_:
        raise ValueError(f"{name}: expected boolean failures with shape (shots, 2)")
    if seconds.shape != fail.shape or not np.isfinite(seconds).all() or (seconds < 0).any():
        raise ValueError(f"{name}: invalid decode times")
    route = None
    if name == HIERARCHICAL:
        route = np.concatenate([part.route for part in parts])[order]
        if route.shape != fail.shape:
            raise ValueError(f"{name}: invalid fallback mask")
    return Batch(ids[order], fail, seconds, route)


def load_comparison(paths):
    settings = None
    groups = {}
    for path in paths:
        try:
            current, entries = read_cache(path)
        except KeyError as error:
            raise ValueError(f"{path}: missing {error}; regenerate using current cache writers") from error
        if settings is None:
            settings = current
        different = [key for key in settings if settings[key] != current[key]]
        if different:
            raise ValueError(f"{path}: run or timing settings differ for {', '.join(different)}")
        for name, batch in entries:
            groups.setdefault(name, []).append(batch)
    required = {"relay1+H", "relay+H", HIERARCHICAL}
    if missing := required - groups.keys():
        raise ValueError(f"missing caches for {', '.join(sorted(missing))}")
    batches = {name: align(parts, settings["shots"], name) for name, parts in groups.items()}
    return settings, batches


def load_decisions(path, settings):
    with np.load(path) as data:
        if metadata(data, RUN_KEYS) != {key: settings[key] for key in RUN_KEYS}:
            raise ValueError("learned decisions use different run settings")
        if data["basis"].tolist() != BASIS:
            raise ValueError("learned decisions must use X/Z basis ordering")
        ids, route = data["shot"], data["route"]
        order = np.argsort(ids)
        if ids.dtype.kind not in "iu" or not np.array_equal(ids[order], np.arange(settings["shots"])):
            raise ValueError("learned decisions must cover every shot ID exactly once")
        if route.shape != (settings["shots"], 2) or route.dtype != np.bool_:
            raise ValueError("learned route must be a boolean (shots, 2) array")
        return route[order]


def replay(bp, relay, route):
    """Replay a decision per basis; times exclude feature extraction and router inference."""
    return Batch(bp.shot, np.where(route, relay.fail, bp.fail),
                 bp.seconds + np.where(route, relay.seconds, 0), route)


def cycle_interval(failed_pairs, shots, cycles):
    """Transform a 95% Wilson interval on pair failures to per-cycle rates."""
    z = 1.959963984540054
    fraction = failed_pairs / shots
    denominator = 1 + z**2 / shots
    center = (fraction + z**2 / (2 * shots)) / denominator
    radius = z * np.sqrt(fraction * (1 - fraction) / shots + z**2 / (4 * shots**2)) / denominator
    bounds = np.clip([center - radius, center + radius], 0, 1)
    return 1 - (1 - bounds) ** (1 / cycles)


def comparison_rows(batches, cycles, decisions=None, random_seed=1):
    policies = [
        ("Relay-BP-1", batches["relay1+H"], "measured"),
        ("Relay-BP-5", batches["relay+H"], "measured"),
        ("Convergence-only", batches[HIERARCHICAL], "measured"),
    ]
    if decisions is not None:
        bp, relay = batches["bp+H"], batches["relay+H"]
        rng = np.random.default_rng(random_seed)
        random_route = np.column_stack([rng.permutation(column) for column in decisions.T])
        policies += [
            ("Learned router", replay(bp, relay, decisions), "replay"),
            ("Random matched", replay(bp, relay, random_route), "replay"),
        ]

    rows = []
    for label, batch, timing in policies:
        pair_fail = batch.fail.any(axis=1)
        rate, error = per_cycle_rate(pair_fail, cycles)
        failed_pairs = int(pair_fail.sum())
        low, high = cycle_interval(failed_pairs, len(batch.shot), cycles)
        elapsed = batch.seconds.sum(axis=1) * 1e3
        rows.append({
            "policy": label, "timing": timing, "shots": len(batch.shot),
            "fail_x": int(batch.fail[:, 0].sum()), "fail_z": int(batch.fail[:, 1].sum()),
            "failed_pairs": failed_pairs, "ler_per_cycle": rate, "ler_se": error,
            "ler_ci95_low": low, "ler_ci95_high": high,
            "mean_ms": elapsed.mean(), "p99_ms": np.percentile(elapsed, 99),
            "fallback_rate": None if batch.route is None else batch.route.mean(),
        })
    return rows


def plot_report(rows, settings, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for row in rows:
        label = row["policy"]
        marker = "o"
        if row["timing"] == "replay":
            label += " (replay)"
            marker = "s"
        rate = row["ler_per_cycle"]
        below = max(0, rate - row["ler_ci95_low"])
        above = max(0, row["ler_ci95_high"] - rate)
        ax.errorbar(row["mean_ms"], rate, yerr=[[below], [above]],
                    fmt=marker, capsize=4, label=label)
    ax.set(xlabel="Mean decode latency per X/Z pair (ms)", ylabel="Logical error rate per cycle",
           title=f"Gross code at p={settings['p']}, loss={settings['p_loss']} ({settings['shots']} pairs)")
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.grid(alpha=0.2)
    ax.legend(title="95% Wilson intervals")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--decisions", type=Path, help="optional NPZ of held-out learned routing decisions")
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True, help="output prefix for .csv and .png")
    args = parser.parse_args()
    try:
        settings, batches = load_comparison(args.caches)
        if args.decisions and "bp+H" not in batches:
            raise ValueError("learned replay requires a standalone bp+H cache")
        decisions = load_decisions(args.decisions, settings) if args.decisions else None
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))

    rows = comparison_rows(batches, settings["cycles"], decisions, args.random_seed)
    for row in rows:
        row.update(settings)
    csv_path = args.out.with_suffix(".csv")
    plot_path = args.out.with_suffix(".png")
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_report(rows, settings, plot_path)
    print(f"{csv_path}\n{plot_path}")
    if decisions is not None:
        print("Replay latency excludes feature extraction and router inference.")


if __name__ == "__main__":
    main()
