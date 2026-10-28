from pathlib import Path
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

from project3_pinn import (
    ALPHA,
    SENSOR_COUNTS,
    create_reference_field,
    create_sparse_measurements,
)


# ============================================================
# Project 3 Conventional Spectral Inverse Baseline
# ============================================================

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

BASELINE_RESULTS_FILE = Path(
    "project3_baseline_results.csv"
)

PINN_RESULTS_FILE = Path(
    "project3_sensor_sweep.csv"
)

# Number of Fourier sine modes used by the conventional model
N_MODES = 12

# Small regularization parameter used to stabilize
# poorly conditioned inverse problems
RIDGE_LAMBDA = 1e-8


# ============================================================
# Spectral Heat-Equation Model
# ============================================================

def build_design_matrix(x, t, n_modes=N_MODES):
    """
    Construct the heat-equation spectral basis:

        T(x,t) =
        sum a_n sin(n*pi*x)
        exp[-alpha*(n*pi)^2*t]
    """

    x = np.asarray(x).reshape(-1)
    t = np.asarray(t).reshape(-1)

    A = np.zeros(
        (len(x), n_modes)
    )

    for n in range(
        1,
        n_modes + 1
    ):

        A[:, n - 1] = (
            np.sin(n * np.pi * x)
            *
            np.exp(
                -ALPHA
                * (n * np.pi) ** 2
                * t
            )
        )

    return A


# ============================================================
# Conventional Reconstruction
# ============================================================

def spectral_reconstruction(
    n_sensors,
    X,
    T
):
    """
    Reconstruct the complete field using
    sparse measurements and a spectral
    heat-equation representation.
    """

    (
        X_sensor,
        T_sensor,
        temperature_sensor
    ) = create_sparse_measurements(
        n_sensors
    )

    x_measure = X_sensor.flatten()

    t_measure = T_sensor.flatten()

    y_measure = temperature_sensor.flatten()


    # --------------------------------------------------------
    # Build inverse-problem matrix
    # --------------------------------------------------------

    A = build_design_matrix(
        x_measure,
        t_measure
    )


    # --------------------------------------------------------
    # Measure conditioning
    # --------------------------------------------------------

    condition_number = np.linalg.cond(
        A
    )


    # --------------------------------------------------------
    # Solve regularized least squares
    #
    # (A^T A + lambda I)a = A^T y
    # --------------------------------------------------------

    start_time = time.perf_counter()

    ATA = A.T @ A

    regularized_matrix = (
        ATA
        +
        RIDGE_LAMBDA
        * np.eye(N_MODES)
    )

    rhs = A.T @ y_measure

    coefficients = np.linalg.solve(
        regularized_matrix,
        rhs
    )


    # --------------------------------------------------------
    # Evaluate on full reference grid
    # --------------------------------------------------------

    x_eval = X.flatten()

    t_eval = T.flatten()

    A_eval = build_design_matrix(
        x_eval,
        t_eval
    )

    prediction = (
        A_eval
        @ coefficients
    )

    prediction = prediction.reshape(
        X.shape
    )

    reconstruction_time = (
        time.perf_counter()
        - start_time
    )


    return (
        prediction,
        coefficients,
        condition_number,
        reconstruction_time
    )


# ============================================================
# Error Metric
# ============================================================

def relative_l2_error(
    prediction,
    truth
):

    return (
        np.linalg.norm(
            prediction - truth
        )
        /
        np.linalg.norm(
            truth
        )
    )


# ============================================================
# Individual Reconstruction Figure
# ============================================================

def plot_baseline_reconstruction(
    X,
    T,
    truth,
    prediction,
    n_sensors,
    error
):

    absolute_error = np.abs(
        prediction - truth
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        constrained_layout=True
    )


    im0 = axes[0].contourf(
        X,
        T,
        truth,
        levels=50
    )

    axes[0].set_title(
        "Ground Truth"
    )

    axes[0].set_xlabel(
        "Position, x"
    )

    axes[0].set_ylabel(
        "Time, t"
    )

    fig.colorbar(
        im0,
        ax=axes[0]
    )


    im1 = axes[1].contourf(
        X,
        T,
        prediction,
        levels=50
    )

    axes[1].set_title(
        "Spectral Reconstruction"
    )

    axes[1].set_xlabel(
        "Position, x"
    )

    axes[1].set_ylabel(
        "Time, t"
    )

    fig.colorbar(
        im1,
        ax=axes[1]
    )


    im2 = axes[2].contourf(
        X,
        T,
        absolute_error,
        levels=50
    )

    axes[2].set_title(
        "Absolute Error"
    )

    axes[2].set_xlabel(
        "Position, x"
    )

    axes[2].set_ylabel(
        "Time, t"
    )

    fig.colorbar(
        im2,
        ax=axes[2]
    )


    fig.suptitle(
        f"{n_sensors}-Sensor Conventional Reconstruction "
        f"| Relative L2 Error = {error:.4f}"
    )


    fig.savefig(
        FIGURE_DIR
        / f"project3_baseline_{n_sensors}_reconstruction.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Save Baseline Results
# ============================================================

def save_baseline_results(
    results
):

    with open(
        BASELINE_RESULTS_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Sensors",
            "Sparsity S",
            "Relative L2 Error",
            "Percent Error",
            "Condition Number",
            "Reconstruction Time (s)"
        ])

        for result in results:

            writer.writerow([
                result["sensors"],
                result["sparsity"],
                result["relative_l2"],
                result["percent_error"],
                result["condition_number"],
                result["runtime"]
            ])


# ============================================================
# Read PINN Sweep
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

            results[sensors] = float(
                row["Percent Error"]
            )

    return results


# ============================================================
# Compare PINN and Conventional Solver
# ============================================================

def plot_method_comparison(
    baseline_results
):

    pinn_results = (
        read_pinn_results()
    )

    sparsity = [
        result["sparsity"]
        for result
        in baseline_results
    ]

    baseline_error = [
        result["percent_error"]
        for result
        in baseline_results
    ]

    pinn_error = [
        pinn_results[
            result["sensors"]
        ]
        for result
        in baseline_results
    ]


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        sparsity,
        pinn_error,
        marker="o",
        linewidth=2,
        label="PINN"
    )

    ax.plot(
        sparsity,
        baseline_error,
        marker="s",
        linewidth=2,
        label="Conventional Spectral Solver"
    )

    ax.set_xlabel(
        "Measurement Sparsity, S = 16 / Ns"
    )

    ax.set_ylabel(
        "Relative L2 Error (%)"
    )

    ax.set_title(
        "PINN vs. Conventional Reconstruction"
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
        / "project3_pinn_vs_baseline.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Conditioning Plot
# ============================================================

def plot_conditioning(
    results
):

    sparsity = [
        result["sparsity"]
        for result
        in results
    ]

    condition_numbers = [
        result["condition_number"]
        for result
        in results
    ]


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        sparsity,
        condition_numbers,
        marker="o",
        linewidth=2
    )

    ax.set_xlabel(
        "Measurement Sparsity, S = 16 / Ns"
    )

    ax.set_ylabel(
        "Condition Number of Measurement Matrix"
    )

    ax.set_title(
        "Inverse-Problem Conditioning vs. Measurement Sparsity"
    )

    ax.set_xticks(
        sparsity
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "project3_baseline_conditioning.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\nProject 3: Conventional Spectral Baseline"
    )


    (
        x,
        t,
        X,
        T,
        true_temperature
    ) = create_reference_field()


    results = []


    for n_sensors in SENSOR_COUNTS:

        sparsity = (
            16 / n_sensors
        )


        print(
            "\n"
            "========================================"
        )

        print(
            f"Conventional solver: "
            f"{n_sensors} sensors"
        )

        print(
            "========================================"
        )


        (
            prediction,
            coefficients,
            condition_number,
            runtime
        ) = spectral_reconstruction(
            n_sensors,
            X,
            T
        )


        error = relative_l2_error(
            prediction,
            true_temperature
        )


        print(
            f"Relative L2 error: "
            f"{error:.6e}"
        )

        print(
            f"Percent error: "
            f"{100 * error:.4f}%"
        )

        print(
            f"Condition number: "
            f"{condition_number:.4e}"
        )

        print(
            f"Runtime: "
            f"{runtime:.6f} seconds"
        )


        results.append({

            "sensors":
                n_sensors,

            "sparsity":
                sparsity,

            "relative_l2":
                error,

            "percent_error":
                100 * error,

            "condition_number":
                condition_number,

            "runtime":
                runtime
        })


        plot_baseline_reconstruction(
            X,
            T,
            true_temperature,
            prediction,
            n_sensors,
            error
        )


    save_baseline_results(
        results
    )

    plot_method_comparison(
        results
    )

    plot_conditioning(
        results
    )


    print(
        "\nFiles created:"
    )

    print(
        "project3_baseline_results.csv"
    )

    print(
        "figures/project3_pinn_vs_baseline.png"
    )

    print(
        "figures/project3_baseline_conditioning.png"
    )


if __name__ == "__main__":
    main()
