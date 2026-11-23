import os, sys, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "challenge-1-ml-decoder-kit"))
from circuit import load_circuit, dem_matrices, sample, GROSS_CIRCUITS
from decoders import relay_decoder, bposd_decoder
from ldpc import BpOsdDecoder, BpLsdDecoder
p = float(sys.argv[1]); N = int(sys.argv[2])
c = load_circuit(GROSS_CIRCUITS["Z"], p)
H, L, pri = dem_matrices(c, "Z"); H = H.tocsr()
print("H", H.shape, flush=True)
det, obs = sample(c, N, seed=7)
det = det[:, :]  # basis rows
from circuit import detector_ids
det = det[:, detector_ids(c, "Z")]
r1 = relay_decoder(H, pri, stop_nconv=1)
r5 = relay_decoder(H, pri)
# second stage decoders: priors replaced per shot
osd0 = BpOsdDecoder(H, error_channel=list(pri), max_iter=1, bp_method="minimum_sum", osd_method="OSD_0", osd_order=0)
osdcs = BpOsdDecoder(H, error_channel=list(pri), max_iter=1, bp_method="minimum_sum", osd_method="OSD_CS", osd_order=10)
lsd0 = BpLsdDecoder(H, error_channel=list(pri), max_iter=1, bp_method="minimum_sum", lsd_method="LSD_0", lsd_order=0)
lsdcs = BpLsdDecoder(H, error_channel=list(pri), max_iter=1, bp_method="minimum_sum", lsd_method="LSD_CS", lsd_order=4)
bplsd = BpLsdDecoder(H, error_channel=list(pri), max_iter=80, bp_method="minimum_sum", ms_scaling_factor=0.0, lsd_method="LSD_CS", lsd_order=4)
bposd = bposd_decoder(H, pri)
rc = relay_decoder(H, pri, stop_nconv=1, num_sets=int(sys.argv[4]))
names = ["r1","r5","bposd","bplsd","rc","rc+osd0","rc+osdcs10","rc+lsd0","rc+lsdcs4"]
fail = {k: np.zeros(N,bool) for k in names}; t = {k: np.zeros(N) for k in names}
signchk=[]
def T(f,*a):
    s=time.perf_counter(); o=f(*a); return o, time.perf_counter()-s
def lerr(e, k): return ((L @ np.asarray(e,np.uint8)) % 2 != obs[k]).any()
nfail1=0
r1conv=np.zeros(N,bool)
for k in range(N):
    s = det[k]
    res, dt = T(r1.decode_detailed, s); fail["r1"][k]=lerr(res.decoding,k); t["r1"][k]=dt; r1conv[k]=res.success
    if k < int(sys.argv[3]):
        e, dt = T(bposd.decode, s); fail["bposd"][k]=lerr(e,k); t["bposd"][k]=dt
    res, dt = T(rc.decode_detailed, s); fail["rc"][k]=lerr(res.decoding,k); t["rc"][k]=dt
    llr = np.nan_to_num(np.asarray(res.posterior_ratios, float), nan=0.0, posinf=50, neginf=-50).clip(-50, 50)
    if len(signchk) < 20 and res.success:
        d = np.asarray(res.decoding); 
        if d.sum(): signchk.append(float(np.mean(llr[d==1] < 0)))
    for name, dec in [("rc+osd0",osd0),("rc+osdcs10",osdcs),("rc+lsd0",lsd0),("rc+lsdcs4",lsdcs)]:
        if res.success:
            fail[name][k]=fail["rc"][k]; t[name][k]=t["rc"][k]; continue
        def stage2():
            q = 1.0/(1.0+np.exp(llr))   # P(error) if llr = log(P0/P1)
            q = np.clip(q, 1e-12, 0.5-1e-9) if False else np.clip(q,1e-12,1-1e-12)
            dec.update_channel_probs(q)
            return dec.decode(s)
        e, dt = T(stage2); fail[name][k]=lerr(e,k); t[name][k]=t["rc"][k]+dt
    nfail1 += (not res.success)
print("r1 nonconverged", (~r1conv).sum(), "r1 fails among nonconverged", (fail["r1"]&~r1conv).sum(), "r1 time on nonconv mean ms", 1e3*t["r1"][~r1conv].mean() if (~r1conv).any() else 0)
print("capped relay nonconverged:", nfail1, "/", N, " sign check frac(llr<0 where decoding=1):", np.mean(signchk) if signchk else None)
for k in [n for n in names if n not in ("r5","bplsd","bposd")]:
    n = int(sys.argv[3]) if k=="bposd" else N
    f=fail[k][:n]; tt=t[k][:n]
    print(f"{k:12s} fails {f.sum():4d}/{n}  mean {1e3*tt.mean():8.2f} ms  p99 {1e3*np.percentile(tt,99):8.2f}  max {1e3*tt.max():8.2f}")
# stage-2 only time on nonconverged shots
mask = None
