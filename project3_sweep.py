from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from project3_pinn import (
    SENSOR_COUNTS,
    create_reference_field,
    train_pinn,
    evaluate_pinn,
    plot_loss_history,
    plot_reconstruction,
)


# ============================================================
# Project 3 Sensor-Sparsity Sweep
# ============================================================

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

RESULTS_FILE = Path("project3_sensor_sweep.csv")


def save_results_csv(results):
    """
    Save numerical results from all sensor cases.
    """

    with open(
        RESULTS_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Sensors",
            "Sparsity S",
            "Relative L2 Error",
            "Percent Error",
            "Training Time (s)",
            "Inference Time (s)",
            "Final PDE Loss",
            "Final Boundary Loss",
            "Final Initial Loss",
            "Final Measurement Loss"
        ])

        for result in results:

            writer.writerow([
                result["sensors"],
                result["sparsity"],
                result["relative_l2"],
                result["percent_error"],
                result["training_time"],
                result["inference_time"],
                result["final_pde"],
                result["final_bc"],
                result["final_ic"],
                result["final_data"]
            ])


def plot_error_vs_sparsity(results):
    """
    P3:
    Plot reconstruction error versus measurement sparsity.
    """

    sparsity = [
        result["sparsity"]
        for result in results
    ]

    percent_error = [
        result["percent_error"]
        for result in results
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        sparsity,
        percent_error,
        marker="o",
        linewidth=2
    )

    ax.set_xlabel(
        "Measurement Sparsity, S = 16 / Ns"
    )

    ax.set_ylabel(
        "Relative L2 Error (%)"
    )

    ax.set_title(
        "PINN Reconstruction Error vs. Measurement Sparsity"
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
        / "project3_error_vs_sparsity.png",
        dpi=300
    )

    plt.close(fig)


def plot_training_time(results):
    """
    P5:
    Plot PINN training time versus measurement sparsity.
    """

    sparsity = [
        result["sparsity"]
        for result in results
    ]

    training_time = [
        result["training_time"]
        for result in results
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        sparsity,
        training_time,
        marker="o",
        linewidth=2
    )

    ax.set_xlabel(
        "Measurement Sparsity, S = 16 / Ns"
    )

    ax.set_ylabel(
        "Training Time (seconds)"
    )

    ax.set_title(
        "PINN Training Time vs. Measurement Sparsity"
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
        / "project3_training_time_vs_sparsity.png",
        dpi=300
    )

    plt.close(fig)


def print_summary(results):
    """
    Print final sensor-sweep results.
    """

    print(
        "\n"
        "============================================================"
    )

    print(
        "PROJECT 3 SENSOR-SPARSITY SWEEP RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Sensors':>8} "
        f"{'S':>8} "
        f"{'L2 Error':>14} "
        f"{'Error (%)':>14} "
        f"{'Train (s)':>14}"
    )

    print(
        "-" * 65
    )

    for result in results:

        print(
            f"{result['sensors']:8d} "
            f"{result['sparsity']:8.1f} "
            f"{result['relative_l2']:14.6f} "
            f"{result['percent_error']:14.4f} "
            f"{result['training_time']:14.2f}"
        )


def main():

    print(
        "\nProject 3: Sensor-Sparsity Sweep"
    )

    # --------------------------------------------------------
    # Create common ground-truth field
    # --------------------------------------------------------

    (
        x,
        t,
        X,
        T,
        true_temperature
    ) = create_reference_field()

    results = []


    # ========================================================
    # Run all four sensor cases
    # ========================================================

    for n_sensors in SENSOR_COUNTS:

        sparsity = (
            16 / n_sensors
        )

        print(
            "\n"
            "============================================================"
        )

        print(
            f"CASE: {n_sensors} SENSORS"
        )

        print(
            f"SPARSITY: S = {sparsity:g}"
        )

        print(
            "============================================================"
        )


        # ----------------------------------------------------
        # Train PINN
        # ----------------------------------------------------

        (
            model,
            history,
            training_time
        ) = train_pinn(
            n_sensors
        )


        # ----------------------------------------------------
        # Evaluate PINN
        # ----------------------------------------------------

        (
            prediction,
            relative_l2,
            inference_time
        ) = evaluate_pinn(
            model,
            X,
            T,
            true_temperature
        )
        if n_sensors == 2:
    np.save(
        "project3_pinn_2_prediction.npy",
        prediction
    )


        # ----------------------------------------------------
        # Save individual-case plots
        # ----------------------------------------------------

        plot_loss_history(
            history,
            n_sensors
        )

        plot_reconstruction(
            X,
            T,
            true_temperature,
            prediction,
            n_sensors,
            relative_l2
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({

            "sensors":
                n_sensors,

            "sparsity":
                sparsity,

            "relative_l2":
                relative_l2,

            "percent_error":
                100.0 * relative_l2,

            "training_time":
                training_time,

            "inference_time":
                inference_time,

            "final_pde":
                history["pde"][-1],

            "final_bc":
                history["bc"][-1],

            "final_ic":
                history["ic"][-1],

            "final_data":
                history["data"][-1]
        })


    # ========================================================
    # Save summary outputs
    # ========================================================

    save_results_csv(
        results
    )

    plot_error_vs_sparsity(
        results
    )

    plot_training_time(
        results
    )

    print_summary(
        results
    )


    print(
        "\nFiles created:"
    )

    print(
        "project3_sensor_sweep.csv"
    )

    print(
        "figures/project3_error_vs_sparsity.png"
    )

    print(
        "figures/project3_training_time_vs_sparsity.png"
    )


if __name__ == "__main__":
    main()
