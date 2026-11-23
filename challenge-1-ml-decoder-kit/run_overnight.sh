#!/bin/bash
# Overnight queue (BlueQubit, after the seed-14 and seed-26 runs). Everything writes to results/06_router/preregistered_s14_s26/.
#   1. Fresh-seed confirmation of the net-benefit router (no loss), seed 14, all three p (pre-registered:
#      primary = 10% budget vs simple rule at equal measured mean time, paired bootstrap; see econ_router.py).
#   2. Loss-aware router: training (seed 22) and threshold (seed 24) data at 1% loss, p = 0.005 / 0.004.
#   3. Loss-aware router test on seed 26 (data made by the previous queue step).
#   bash run_overnight.sh > overnight.log 2>&1
S=sweep.py
O=results/06_router/preregistered_s14_s26
R=results/06_router/caches   # relay-BP-1 / relay-BP-5 caches (seeds 2, 4, 14)
L=results/06_router/loss_router
mkdir -p $O $L
W=${WORKERS:-20}

echo "=== 1. net-benefit router, fresh seed 14 ($(date))"
for spec in "0.005 10000 20000 10000" "0.004 10000 10000 10000" "0.003 50000 50000 50000"; do
  set -- $spec; p=$1; ntr=$2; ncal=$3; nte=$4
  python econ_router.py --monotone \
    --train-relay1 $R/relay1_p${p}_l0.0_unaware_s2_n${ntr}_shard0of1.npz \
    --train-relay5 $R/relay5_p${p}_l0.0_unaware_s2_n${ntr}_shard0of1.npz \
    --calibration $R/relay1_p${p}_l0.0_unaware_s4_n${ncal}_shard0of1.npz \
    --test-relay1 $R/relay1_p${p}_l0.0_unaware_s14_n${nte}_shard0of1.npz \
    --test-relay5 $R/relay5_p${p}_l0.0_unaware_s14_n${nte}_shard0of1.npz \
    --out $O/router_s14_p$p > $O/router_s14_p$p.txt 2>&1
  echo "p=$p done"; grep -A8 MATCHED $O/router_s14_p$p.txt
done

echo "=== 2. loss-aware router training/threshold data ($(date))"
for p in 0.005 0.004; do
  for d in relay1 relay5; do
    [ -f $L/${d}_p${p}_l0.01_reweight_s22_n6000_shard0of1.npz ] || \
      python $S run --decoder $d --p $p --p-loss 0.01 --loss-mode reweight --shots 6000 --seed 22 --workers $W --out $L
  done
  [ -f $L/relay1_p${p}_l0.01_reweight_s24_n10000_shard0of1.npz ] || \
    python $S run --decoder relay1 --p $p --p-loss 0.01 --loss-mode reweight --shots 10000 --seed 24 --workers $W --out $L
done

echo "=== 3. loss-aware router test, seed 26 ($(date))"
for p in 0.005 0.004; do
  python econ_router.py --monotone \
    --train-relay1 $L/relay1_p${p}_l0.01_reweight_s22_n6000_shard0of1.npz \
    --train-relay5 $L/relay5_p${p}_l0.01_reweight_s22_n6000_shard0of1.npz \
    --calibration $L/relay1_p${p}_l0.01_reweight_s24_n10000_shard0of1.npz \
    --test-relay1 $L/relay1_p${p}_l0.01_reweight_s26_n10000_shard0of1.npz \
    --test-relay5 $L/relay5_p${p}_l0.01_reweight_s26_n10000_shard0of1.npz \
    --out $O/loss_router_s26_p$p > $O/loss_router_s26_p$p.txt 2>&1
  echo "p=$p done"; grep -E "alone|^train" $O/loss_router_s26_p$p.txt; grep -A8 MATCHED $O/loss_router_s26_p$p.txt
done
python $S summarize $L/*.npz > $O/loss_router_summary.txt 2>&1
echo "=== ALL DONE ($(date))"
