"""Random decoder choice at matched mean time: for each X/Z pair pick relay-BP-1 or relay-BP-5 at random before
decoding, with P(relay-BP-5) set so the mean time equals the target. Prints failed pairs (mean, 95% range) and p99.

  python router_coinflip.py results/relay1_<...>.npz results/relay5_<...>.npz 322 469
"""
import numpy as np,sys
def load(f):
    d=np.load(f); o=np.argsort(d['shot_idx']); return d['fail'][o].any(1), d['seconds'][o].sum(1)*1e3
f1,t1=load(sys.argv[1]); f5,t5=load(sys.argv[2])
print("pairs",len(f1),"r1",f1.sum(),t1.mean(),"r5",f5.sum(),t5.mean())
rng=np.random.default_rng(0)
for target in map(float,sys.argv[3:]):
    f=(target-t1.mean())/(t5.mean()-t1.mean())
    fails=[];p99=[];mean=[]
    for _ in range(2000):
        m=rng.random(len(f1))<f
        fails.append(np.where(m,f5,f1).sum()); tt=np.where(m,t5,t1); p99.append(np.percentile(tt,99)); mean.append(tt.mean())
    fails=np.array(fails)
    print(f"target {target:.0f} ms: frac {f:.3f} fails mean {fails.mean():.1f} [2.5-97.5%: {np.percentile(fails,2.5):.0f}-{np.percentile(fails,97.5):.0f}] mean ms {np.mean(mean):.0f} p99 {np.mean(p99):.0f}")
