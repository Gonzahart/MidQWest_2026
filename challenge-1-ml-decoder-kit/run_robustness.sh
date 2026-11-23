#!/bin/bash
# Track D: how robust is loss-aware decoding? relay-BP-5, p = 0.004, 1% loss, seed 12 (fresh), 2,000 pairs per run.
# Accuracy only (20 workers). Every run decodes the same shots; noisy flags are seeded per shot, so runs pair up.
#   bash run_robustness.sh            then: python sweep.py summarize results/04_robustness/*.npz
set -e
W=${WORKERS:-20}
A="--decoder relay5 --p 0.004 --p-loss 0.01 --shots 2000 --seed 12 --workers $W --out results/04_robustness"
python sweep.py run $A --loss-mode unaware
python sweep.py run $A --loss-mode reweight
for fp in 0.2 0.3 0.4; do python sweep.py run $A --loss-mode reweight --fired-prior $fp --tag prior$fp; done
for fn in 0.05 0.2; do python sweep.py run $A --loss-mode reweight --flag-fn $fn --tag miss$fn; done
python sweep.py run $A --loss-mode reweight --flag-fn 0.2 --unfired-prior 0.001 --tag miss0.2_aware
for fa in 0.05 0.2; do python sweep.py run $A --loss-mode reweight --flag-fp $fa --tag false$fa; done
python sweep.py summarize results/04_robustness/*.npz
