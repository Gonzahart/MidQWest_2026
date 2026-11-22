# Exploratory GPU relay-BP timing

## Results

Each row summarizes 1,000 X+Z pairs. Latencies are in milliseconds per pair. All runs used relay-BP-5 with zero loss.

| p | Decoder | Failed X / Z / pairs | LER per cycle | Mean | p50 | p99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.003 | CPU | 0 / 0 / 0 | 0 | 46.2 | 31.9 | 303.4 | 699.1 |
| 0.003 | GPU | 0 / 0 / 0 | 0 | 14.0 | 9.5 | 100.9 | 464.7 |
| 0.004 | CPU | 3 / 5 / 8 | 0.000669124 | 208.9 | 69.9 | 3,332.2 | 3,481.8 |
| 0.004 | GPU | 3 / 8 / 11 | 0.000921321 | 45.8 | 14.7 | 588.1 | 804.6 |
| 0.005 | CPU | 50 / 44 / 93 | 0.00810141 | 922.8 | 271.7 | 4,820.6 | 6,676.0 |
| 0.005 | GPU | 43 / 57 / 96 | 0.00837522 | 190.8 | 60.1 | 995.1 | 1,539.8 |

| p | Logical agreement | CPU-only wrong | GPU-only wrong | Exact p | CPU p99 / GPU p99 | Prior rebuild X / Z (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 0.003 | 100.00% | 0 | 0 | 1 | 3.01x | 4.53 / 5.43 |
| 0.004 | 99.45% | 2 | 5 | 0.4531 | 5.67x | 4.66 / 5.03 |
| 0.005 | 94.15% | 27 | 33 | 0.519 | 4.84x | 5.49 / 5.16 |

A prior-update smoke test used H=[[1,1]] and syndrome [1]. Swapping the two priors from [0.2,0.001] to [0.001,0.2] switched the returned correction from [1,0] to [0,1].

The primary p=0.004 run reached the full GPU iteration budget of 18,080 on at least one basis shot. Device-wide VRAM readings during that run were about 1.7 GiB including the Windows display and other applications. WSL did not expose per-process GPU memory in the query. For both 936 x 8,784 Gross-code matrices, a separate CUDA free-memory probe saw 11,241 MiB before construction and 11,219 MiB after the first decode, a 22 MiB device-wide change. These are snapshots, not peak allocation measurements.

The Windows CPU timing sweep had exited. Desktop applications remained active, so the measurements describe this shared desktop. The benchmark used a single base seed. With only seven discordant basis outcomes at p=0.004, the data give limited precision for comparing small differences in failure rate.

Use the p=0.004 result as an exploratory slide about tail latency. Present the failure counts alongside the speedup and retain the official CPU results as the competition timing.

This experiment compares single-shot CUDA-Q QEC and Rust relay-BP for the [[144,12,12]] Gross code. The project's official timing comes from its CPU timing harness.

## Environment and procedure

- Intel Core i5-14600K, GeForce RTX 3060 (12,288 MiB), Ubuntu 26.04.1 on WSL2, Windows driver 610.88. The WSL driver reports CUDA compatibility 13.3. The Python environment has CUDA runtime 13.5.37, CUDA-Q 0.17.0, CUDA-Q QEC 0.9.0 (cu13), Python 3.11.18 and relay-bp 0.2.2 from the kit wheel.
- The Python 3.11 environment was created with uv. Installation used the binary relay-bp wheel, requirements.txt, pytest and cudaq-qec. The kit suite passed all 24 tests. The Windows seed-6 timing sweep finished before validation.
- Each basis uses 12 cycles and 1,000 shots from a single Stim sample, X seed 8 and Z seed 9. Each decoder gets the same syndrome and truth. One warm-up call per basis and decoder precedes measurement. Decoder order alternates by shot. Both decoders run in one sequential process with BLAS, OpenMP and Rayon set to one thread.
- Latency includes the call and conversion to a host uint8 correction, using the kit timer. Each pair sums X and Z milliseconds before computing percentiles. Sampling, construction, warm-up and logical scoring are outside the timer. Batch size is one.
- Acceptance uses the two-sided exact binomial test on basis shots where just one decoder is wrong. Logical agreement compares the full observable prediction vectors, including when both are wrong. A high sign-test p-value alone cannot establish statistical equivalence.

## Decoder settings

The CPU uses the kit's relay-BP-5 settings. The GPU uses sparse min-sum with disordered memory (`bp_method=3`), sequential relay (`composition=1`), scale 1, fp64, gamma0 0.15, the kit gamma interval [-0.22628432386414646, 0.6216020925981884], 80 initial iterations, up to 300 relay sets, 60 iterations per set and NConv=5. The GPU uses BP seed 8, one ensemble lane (default), per-column DEM priors, hard error output, no OSD and iteration reporting. A fixed gamma seed does not guarantee bitwise repeatability with the default GPU reduction mode.

A separate GPU probe used two identical parity rows, syndrome [1,0], three initial iterations and four relay sets. The inconsistent syndrome exhausted the budget. With `max_iterations=2` and 4 the decoder reported 11 and 19 iterations, respectively, verifying that the limit applies to each leg.

The installed Decoder exposes `decode_batch` and synchronous `decode`. Results expose a correction, convergence flag and optional iteration count. The Python API exposes prior probabilities at construction, so `set_priors` rebuilds the native decoder. Rebuild times below cover replacement of a live decoder at the same priors, one call per basis. They do not measure loss-reweighted accuracy or end-to-end loss-aware timing. [NVIDIA decoder API](https://nvidia.github.io/cudaq-qec/api/qec/python_api.html)

## Reproduction

From `challenge-1-ml-decoder-kit` in WSL:

```bash
.venv/bin/python -m pytest test_kit.py -q
PYTHONPATH=. .venv/bin/python -u 'results/Sean GPU-relay/gpu_check.py' --p 0.004 --shots 1000 --seed 8 --out 'results/Sean GPU-relay/gpu_check_p0.004_s8.csv'
PYTHONPATH=. .venv/bin/python -u 'results/Sean GPU-relay/gpu_check.py' --p 0.003 --shots 1000 --seed 8 --out 'results/Sean GPU-relay/gpu_check_p0.003_s8.csv'
PYTHONPATH=. .venv/bin/python -u 'results/Sean GPU-relay/gpu_check.py' --p 0.005 --shots 1000 --seed 8 --out 'results/Sean GPU-relay/gpu_check_p0.005_s8.csv'
```

The CSV contains one summary row per decoder. Each matching NPZ contains CPU/GPU failure and seconds arrays with shape (shots, 2) in X, Z order, logical agreement flags, GPU iterations and per-basis GPU prior rebuild costs.
