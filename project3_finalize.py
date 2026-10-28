from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from project3_pinn import create_reference_field


# ============================================================
# Project 3 Final Diagnostics
# P4: Spectral analysis
# P5: Computational-cost comparison
# ============================================================

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

PINN_RESULTS_FILE = Path(
    "project3_sensor_sweep.csv"
)

BASELINE_RESULTS_FILE = Path(
    "project3_baseline_results.csv"
)

PINN_PREDICTION_FILE = Path(
    "project3_pinn_2_prediction.npy"
)

COMBINED_RESULTS_FILE = Path(
    "project3_combined_summary.csv"
)


# ============================================================
# Read PINN Results
# ============================================================

def read_pinn_results():

    results = {}

    with open(
        PINN_RESULTS_FILE,
        "r"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            sensors = int(
                row["Sensors"]
            )

            results[sensors] = {

                "sparsity":
                    float(
                        row["Sparsity S"]
                    ),

                "relative_l2":
                    float(
                        row["Relative L2 Error"]
                    ),

                "percent_error":
                    float(
                        row["Percent Error"]
                    ),

                "training_time":
                    float(
                        row["Training Time (s)"]
                    ),

                "inference_time":
                    float(
                        row["Inference Time (s)"]
                    ),

                "pde_loss":
                    float(
                        row["Final PDE Loss"]
                    ),

                "bc_loss":
                    float(
                        row["Final Boundary Loss"]
                    ),

                "ic_loss":
                    float(
                        row["Final Initial Loss"]
                    ),

                "data_loss":
                    float(
                        row["Final Measurement Loss"]
                    )
            }

    return results


# ============================================================
# Read Conventional Baseline Results
# ============================================================

def read_baseline_results():

    results = {}

    with open(
        BASELINE_RESULTS_FILE,
        "r"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            sensors = int(
                row["Sensors"]
            )

            results[sensors] = {

                "sparsity":
                    float(
                        row["Sparsity S"]
                    ),

                "relative_l2":
                    float(
                        row["Relative L2 Error"]
                    ),

                "percent_error":
                    float(
                        row["Percent Error"]
                    ),

                "condition_number":
                    float(
                        row["Condition Number"]
                    ),

                "runtime":
                    float(
                        row["Reconstruction Time (s)"]
                    )
            }

    return results


# ============================================================
# Fourier Sine Spectrum
# ============================================================

def sine_spectrum(
    field,
    x,
    time_index=0,
    n_modes=12
):
    """
    Calculate Fourier sine-series amplitudes.

    The zero-value boundary conditions make a
    sine-series representation appropriate.
    """

    profile = field[
        time_index,
        :
    ]

    amplitudes = []

    for n in range(
        1,
        n_modes + 1
    ):

        basis = np.sin(
            n * np.pi * x
        )

        coefficient = (
            2.0
            * np.trapezoid(
                profile * basis,
                x
            )
        )

        amplitudes.append(
            abs(coefficient)
        )

    return np.array(
        amplitudes
    )


# ============================================================
# P4: Fourier Spectrum
# ============================================================

def plot_fourier_spectrum():

    (
        x,
        t,
        X,
        T,
        truth
    ) = create_reference_field()

    prediction = np.load(
        PINN_PREDICTION_FILE
    )

    modes = np.arange(
        1,
        13
    )

    true_spectrum = sine_spectrum(
        truth,
        x,
        time_index=0,
        n_modes=12
    )

    pinn_spectrum = sine_spectrum(
        prediction,
        x,
        time_index=0,
        n_modes=12
    )


    # --------------------------------------------------------
    # Print important mode amplitudes
    # --------------------------------------------------------

    print(
        "\nFOURIER SPECTRUM CHECK"
    )

    print(
        "---------------------------------------------"
    )

    print(
        f"True Mode 1 amplitude: "
        f"{true_spectrum[0]:.6f}"
    )

    print(
        f"PINN Mode 1 amplitude: "
        f"{pinn_spectrum[0]:.6f}"
    )

    print(
        f"True Mode 4 amplitude: "
        f"{true_spectrum[3]:.6f}"
    )

    print(
        f"PINN Mode 4 amplitude: "
        f"{pinn_spectrum[3]:.6f}"
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        modes,
        true_spectrum,
        marker="o",
        linewidth=2,
        label="Ground Truth"
    )

    ax.plot(
        modes,
        pinn_spectrum,
        marker="s",
        linewidth=2,
        label="2-Sensor PINN"
    )

    ax.set_xlabel(
        "Spatial Fourier Sine Mode"
    )

    ax.set_ylabel(
        "Mode Amplitude"
    )

    ax.set_title(
        "Fourier Spectrum of 2-Sensor PINN Reconstruction"
    )

    ax.set_xticks(
        modes
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "project3_p4_fourier_spectrum.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# P5: Runtime Comparison
# ============================================================

def plot_runtime_comparison(
    pinn_results,
    baseline_results
):

    sensors = [
        16,
        8,
        4,
        2
    ]

    sparsity = [
        pinn_results[s][
            "sparsity"
        ]
        for s in sensors
    ]

    training_time = [
        pinn_results[s][
            "training_time"
        ]
        for s in sensors
    ]

    inference_time = [
        pinn_results[s][
            "inference_time"
        ]
        for s in sensors
    ]

    baseline_time = [
        baseline_results[s][
            "runtime"
        ]
        for s in sensors
    ]


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        sparsity,
        training_time,
        marker="o",
        linewidth=2,
        label="PINN Training"
    )

    ax.semilogy(
        sparsity,
        inference_time,
        marker="s",
        linewidth=2,
        label="PINN Inference"
    )

    ax.semilogy(
        sparsity,
        baseline_time,
        marker="^",
        linewidth=2,
        label="Conventional Reconstruction"
    )

    ax.set_xlabel(
        "Measurement Sparsity, S = 16 / Ns"
    )

    ax.set_ylabel(
        "Runtime (seconds)"
    )

    ax.set_title(
        "Computational Cost Comparison"
    )

    ax.set_xticks(
        sparsity
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "project3_p5_runtime_comparison.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Save Combined Results
# ============================================================

def save_combined_results(
    pinn_results,
    baseline_results
):

    sensors_list = [
        16,
        8,
        4,
        2
    ]

    with open(
        COMBINED_RESULTS_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Sensors",
            "Sparsity",
            "PINN Error (%)",
            "Baseline Error (%)",
            "Condition Number",
            "PINN Training Time (s)",
            "PINN Inference Time (s)",
            "Baseline Runtime (s)",
            "Final PDE Loss",
            "Final Boundary Loss",
            "Final Initial Loss",
            "Final Measurement Loss"
        ])

        for sensors in sensors_list:

            writer.writerow([

                sensors,

                pinn_results[
                    sensors
                ]["sparsity"],

                pinn_results[
                    sensors
                ]["percent_error"],

                baseline_results[
                    sensors
                ]["percent_error"],

                baseline_results[
                    sensors
                ]["condition_number"],

                pinn_results[
                    sensors
                ]["training_time"],

                pinn_results[
                    sensors
                ]["inference_time"],

                baseline_results[
                    sensors
                ]["runtime"],

                pinn_results[
                    sensors
                ]["pde_loss"],

                pinn_results[
                    sensors
                ]["bc_loss"],

                pinn_results[
                    sensors
                ]["ic_loss"],

                pinn_results[
                    sensors
                ]["data_loss"]
            ])


# ============================================================
# Print Report-Ready Summary
# ============================================================

def print_summary(
    pinn_results,
    baseline_results
):

    print(
        "\n"
        "=========================================================================="
    )

    print(
        "PROJECT 3 FINAL NUMERICAL SUMMARY"
    )

    print(
        "=========================================================================="
    )

    print(
        f"{'Sensors':>8}"
        f"{'S':>8}"
        f"{'PINN %':>12}"
        f"{'Base %':>12}"
        f"{'Condition':>16}"
        f"{'Train(s)':>12}"
    )

    print(
        "-" * 70
    )

    for sensors in [
        16,
        8,
        4,
        2
    ]:

        pinn = pinn_results[
            sensors
        ]

        baseline = baseline_results[
            sensors
        ]

        print(
            f"{sensors:8d}"
            f"{pinn['sparsity']:8.1f}"
            f"{pinn['percent_error']:12.4f}"
            f"{baseline['percent_error']:12.6f}"
            f"{baseline['condition_number']:16.3e}"
            f"{pinn['training_time']:12.2f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\nProject 3 Final Diagnostics"
    )


    # --------------------------------------------------------
    # Check that required files exist
    # --------------------------------------------------------

    required_files = [
        PINN_RESULTS_FILE,
        BASELINE_RESULTS_FILE,
        PINN_PREDICTION_FILE
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: "
                f"{file_path}"
            )


    # --------------------------------------------------------
    # Read results
    # --------------------------------------------------------

    pinn_results = (
        read_pinn_results()
    )

    baseline_results = (
        read_baseline_results()
    )


    # --------------------------------------------------------
    # Generate P4
    # --------------------------------------------------------

    plot_fourier_spectrum()


    # --------------------------------------------------------
    # Generate P5
    # --------------------------------------------------------

    plot_runtime_comparison(
        pinn_results,
        baseline_results
    )


    # --------------------------------------------------------
    # Generate combined numerical table
    # --------------------------------------------------------

    save_combined_results(
        pinn_results,
        baseline_results
    )


    # --------------------------------------------------------
    # Print numerical summary
    # --------------------------------------------------------

    print_summary(
        pinn_results,
        baseline_results
    )


    print(
        "\nGenerated:"
    )

    print(
        "figures/project3_p4_fourier_spectrum.png"
    )

    print(
        "figures/project3_p5_runtime_comparison.png"
    )

    print(
        "project3_combined_summary.csv"
    )


if __name__ == "__main__":
    main()
