#!/usr/bin/env bash
# One-time setup on a BlueQubit Hybrid notebook (or any fresh Linux box).
#   bash bq_setup.sh
# Prints machine specs first, so you can decide whether the box is worth paying for.
set -e
echo "== machine: $(nproc) cores, $(free -g 2>/dev/null | awk '/Mem:/{print $2}') GB RAM, $(python3 --version)"
cd "$(dirname "$0")"
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
python3 -m pip install -q -r requirements.txt
python3 -c "import stim, ldpc, relay_bp; print('== ready: stim', stim.__version__, 'ldpc', ldpc.__version__)"
python3 sweep.py run --decoder relay5 --p 0.004 --shots 40 --out /tmp/smoke
echo "== smoke test passed. Start your shard with: nohup python3 sweep.py run ... > sweep.log 2>&1 &"
