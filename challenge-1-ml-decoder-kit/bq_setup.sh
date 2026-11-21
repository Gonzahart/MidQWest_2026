#!/usr/bin/env bash
# One-time setup on a BlueQubit Hybrid notebook (or any fresh Linux box).
#   bq env create midq        # BlueQubit only: the base environment is read-only
#   bash bq_setup.sh
# Prints machine specs first, so you can decide whether the box is worth paying for.
set -e
echo "== machine: $(nproc) cores, $(free -g 2>/dev/null | awk '/Mem:/{print $2}') GB RAM, $(python3 --version)"
cd "$(dirname "$0")"

# relay-bp normally builds from source with Rust + a C compiler. BlueQubit notebooks have no C
# compiler, so wheels/ holds a prebuilt Linux x86_64 build for Python 3.11 (manylinux2014).
if ! python3 -m pip install -q wheels/relay_bp-0.2.2-cp311-*.whl 2>/dev/null; then
  echo "== no matching prebuilt relay-bp wheel; building from source (needs Rust and cc)"
  if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  fi
  source "$HOME/.cargo/env"
fi
if ! python3 -m pip install -q -r requirements.txt; then
  echo "== pip install failed. On BlueQubit run 'bq env create midq' first (the base environment is read-only)."
  exit 1
fi
python3 -c "import stim, ldpc, relay_bp; print('== ready: stim', stim.__version__, 'ldpc', ldpc.__version__)"
python3 sweep.py run --decoder relay5 --p 0.004 --shots 40 --out /tmp/smoke
echo "== smoke test passed. Start your job with: nohup sh -c '...python3 sweep.py run ...' > sweep.log 2>&1 &"
