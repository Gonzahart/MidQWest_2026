"""Projection: the frozen six-bit table with GPU costs for the strong pass (estimate, not a measurement)."""
"""Run from challenge-1-ml-decoder-kit: python "results/07_truth_table/table/gpu_cost_projection.py"
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from truth_table_router import load, keys, per_cycle

spec = json.load(open("results/07_truth_table/table/truth_table_router_frozen.json"))
cells, cut, price = spec["cells"], spec["cut"], spec["price_failures_avoided_per_second"]

def table_for(ratio):
    return np.array([c["n"] >= 12 and c["rescue"] - c["regress"] > price * (c["cost"] / ratio) / 1000 for c in cells])

frozen = np.array(spec["escalate"])
for r in (1, 5, 10):
    t = table_for(r)
    print(f"strong pass {r:>2}x cheaper: {t.sum()} escalate cells "
          f"{'(= frozen)' if r == 1 else ''}; cells added vs frozen: {[c['bits'] for c, a, b in zip(cells, t, frozen) if a and not b]}")
pos = [c["bits"] for c in cells if c["n"] >= 12 and c["rescue"] - c["regress"] <= 0]
print(f"cells that never escalate however cheap (net benefit <= 0, n >= 12): {len(pos)}")

def run(f1, f5, ratio, label):
    fast, strong = load(f1), load(f5)
    k = keys(fast, cut)
    out = []
    def row(name, routed, strong_cost, strong_only=False):
        fail = np.where(routed, strong["fail"], fast["fail"]).any(1)
        sec = (strong["seconds"] * strong_cost) if strong_only else fast["seconds"] + routed * strong["seconds"] * strong_cost
        ms = sec.sum(1) * 1e3
        out.append((name, routed.mean(), int(fail.sum()), ms.mean(), np.percentile(ms, 99)))
    z = np.zeros(k.shape, bool)
    row("CPU relay-BP-1", z, 1)
    row("CPU relay-BP-5", ~z, 1, True)
    row("frozen table, CPU fallback (measured)", frozen[k], 1)
    row(f"frozen table, GPU fallback (est. /{ratio})", frozen[k], 1 / ratio)
    row(f"GPU-cost table, GPU fallback (est. /{ratio})", table_for(ratio)[k], 1 / ratio)
    row(f"always GPU relay-BP-5 (est. /{ratio})", ~z, 1 / ratio, True)
    print(f"\n{label}, strong-pass speed-up {ratio}x")
    print(f"{'policy':44s} {'escalated':>9s} {'failed':>7s} {'mean ms':>9s} {'p99 ms':>8s}")
    for n, e, f, m, q in out:
        print(f"{n:44s} {e:9.1%} {f:7d} {m:9.1f} {q:8.0f}")

T = "results/05_timing/final_timing/"
for ratio in (10, 5):
    run("results/06_router/caches/relay1_p0.005_l0.0_unaware_s10_n10000_shard0of1.npz", "results/06_router/caches/relay5_p0.005_l0.0_unaware_s10_n10000_shard0of1.npz", ratio, "p = 0.005, seed 10, 10k pairs (20-worker times)")
run(T + "relay1_p0.005_l0.0_unaware_s16_n1000_shard0of1.npz", T + "relay5_p0.005_l0.0_unaware_s16_n1000_shard0of1.npz", 10, "p = 0.005, seed 16, 1k pairs (single-worker times)")
run(T + "relay1_p0.004_l0.0_unaware_s16_n1000_shard0of1.npz", T + "relay5_p0.004_l0.0_unaware_s16_n1000_shard0of1.npz", 8, "p = 0.004, seed 16, 1k pairs (single-worker times)")
