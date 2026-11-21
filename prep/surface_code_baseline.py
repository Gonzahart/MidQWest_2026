"""
Surface code decoder baseline (MidQWest 2026 prep).

Simulates a rotated surface code memory experiment with circuit-level noise in
Stim, decodes it with minimum-weight perfect matching (PyMatching) via sinter,
and plots logical error rate per round vs physical error rate for d = 3, 5, 7.
Where the curves cross is the threshold (~0.5-1% for this noise model).

Run:   python surface_code_baseline.py
Output: threshold.png + a printed table.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sinter
import stim

DISTANCES = [3, 5, 7]
NOISE = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012]


def make_tasks():
    for d in DISTANCES:
        for p in NOISE:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_z",
                distance=d,
                rounds=d,  # d rounds of syndrome extraction (standard choice)
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p,
                before_round_data_depolarization=p,
            )
            yield sinter.Task(circuit=circuit, json_metadata={"d": d, "p": p, "r": d})


def main():
    stats = sinter.collect(
        num_workers=4,
        tasks=list(make_tasks()),
        decoders=["pymatching"],
        max_shots=200_000,
        max_errors=500,
        print_progress=True,
    )

    print(f"\n{'d':>3} {'p':>7} {'shots':>9} {'errors':>7} {'LER/round':>11}")
    for s in sorted(stats, key=lambda s: (s.json_metadata["d"], s.json_metadata["p"])):
        ler = sinter.shot_error_rate_to_piece_error_rate(
            s.errors / s.shots, pieces=s.json_metadata["r"]
        )
        print(f"{s.json_metadata['d']:>3} {s.json_metadata['p']:>7.3f} "
              f"{s.shots:>9} {s.errors:>7} {ler:>11.2e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda s: s.json_metadata["p"],
        group_func=lambda s: f"d={s.json_metadata['d']}",
        failure_units_per_shot_func=lambda s: s.json_metadata["r"],
    )
    ax.loglog()
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate per round")
    ax.set_title("Rotated surface code, circuit noise, MWPM (PyMatching)")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    fig.savefig("threshold.png", dpi=150, bbox_inches="tight")
    print("\nSaved threshold.png")


if __name__ == "__main__":  # needed: sinter uses multiprocessing (macOS spawn)
    main()
