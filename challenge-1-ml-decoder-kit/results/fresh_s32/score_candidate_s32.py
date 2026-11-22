import sys, numpy as np
from pathlib import Path
from scipy.stats import binomtest
J=Path("results/relay_lab_judge"); sys.path.insert(0,str(J.resolve())); import judge
src=open("results/relay_lab_candidate/replay_compare.py").read()
s=src.index("def evaluate"); e=src.index("prices = np.array")
exec(src[s:e])
for p in (0.005,0.004):
    (old,bt),[((new,at),(thr,tt))]=evaluate(Path("results/fresh_s32"),p,32,10000,[0.04])
    print(f"p={p} seed32: frozen {old.sum()} -> price-only {thr.sum()} -> candidate {new.sum()} | mean {bt.mean():.1f} -> {at.mean():.1f} ms | p99 {np.percentile(bt,99):.0f} -> {np.percentile(at,99):.0f} | fixes {np.sum(old&~new)} breaks {np.sum(new&~old)} p={binomtest(int(np.sum(old&~new)),int(np.sum(old&~new)+np.sum(new&~old))).pvalue:.3g}")
