from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Project 3: Sparse-Data Heat-Equation Reconstruction
# Step 1: Generate analytical reference field and measurements
# ============================================================

ALPHA = 0.01

X_MIN = 0.0
X_MAX = 1.0

T_MIN = 0.0
T_MAX = 1.0

SENSOR_COUNTS = [16, 8, 4, 2]

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


def exact_temperature(x, t, alpha=ALPHA):
    """
    Analytical solution of the 1D heat equation:

        T_t = alpha * T_xx

    with

        T(0,t) = 0
        T(1,t) = 0

    and initial condition

        T(x,0) = sin(pi*x) + 0.5*sin(4*pi*x)
    """

    term_1 = np.sin(np.pi * x) * np.exp(
        -alpha * np.pi**2 * t
    )

    term_2 = 0.5 * np.sin(4.0 * np.pi * x) * np.exp(
        -16.0 * alpha * np.pi**2 * t
    )

    return term_1 + term_2


def create_reference_field():
    """
    Create a dense grid representing the true
    normalized temperature field.
    """

    x = np.linspace(X_MIN, X_MAX, 201)
    t = np.linspace(T_MIN, T_MAX, 101)

    X, T = np.meshgrid(x, t)

    temperature = exact_temperature(X, T)

    return x, t, X, T, temperature


def create_sparse_measurements(n_sensors, n_time_samples=21):
    """
    Generate synthetic measurements at a limited number
    of fixed spatial sensor locations.
    """

    # Interior sensor locations only
    x_sensors = np.linspace(
        X_MIN,
        X_MAX,
        n_sensors + 2
    )[1:-1]

    t_measurements = np.linspace(
        T_MIN,
        T_MAX,
        n_time_samples
    )

    X_sensor, T_sensor = np.meshgrid(
        x_sensors,
        t_measurements
    )

    temperature_sensor = exact_temperature(
        X_sensor,
        T_sensor
    )

    return X_sensor, T_sensor, temperature_sensor


def plot_reference_field(X, T, temperature):
    """
    Plot the complete analytical reference field.
    """

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    contour = ax.contourf(
        X,
        T,
        temperature,
        levels=50
    )

    cbar = fig.colorbar(
        contour,
        ax=ax,
        pad=0.03
    )

    cbar.set_label(
        "Normalized Temperature"
    )

    ax.set_xlabel(
        "Position, x"
    )

    ax.set_ylabel(
        "Time, t"
    )

    ax.set_title(
        "Reference Temperature Field"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "project3_reference_field.png",
        dpi=300
    )

    plt.close(fig)


def plot_sparse_measurements():
    """
    Compare measurement locations for the four
    sensor-sparsity cases.
    """

    # Leave dedicated space on the right for colorbar
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8)
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.84,
        bottom=0.09,
        top=0.88,
        wspace=0.30,
        hspace=0.35
    )

    axes = axes.flatten()

    # Fixed color scale shared by every subplot
    vmin = -0.2
    vmax = 1.45

    scatter = None

    for ax, n_sensors in zip(
        axes,
        SENSOR_COUNTS
    ):

        X_sensor, T_sensor, temperature_sensor = (
            create_sparse_measurements(n_sensors)
        )

        scatter = ax.scatter(
            X_sensor.flatten(),
            T_sensor.flatten(),
            c=temperature_sensor.flatten(),
            s=28,
            vmin=vmin,
            vmax=vmax
        )

        sparsity = 16 / n_sensors

        ax.set_title(
            f"{n_sensors} Sensors, S = {sparsity:g}"
        )

        ax.set_xlabel(
            "Position, x"
        )

        ax.set_ylabel(
            "Time, t"
        )

        ax.set_xlim(
            X_MIN,
            X_MAX
        )

        ax.set_ylim(
            T_MIN,
            T_MAX
        )

    # Create a completely separate colorbar axis
    cbar_axis = fig.add_axes(
        [0.88, 0.18, 0.025, 0.62]
    )

    cbar = fig.colorbar(
        scatter,
        cax=cbar_axis
    )

    cbar.set_label(
        "Measured Normalized Temperature"
    )

    fig.suptitle(
        "Sparse Temperature Measurements",
        fontsize=15
    )

    fig.savefig(
        FIGURE_DIR / "project3_sparse_measurements.png",
        dpi=300
    )

    plt.close(fig)


def print_case_information():
    """
    Print information for each measurement-sparsity case.
    """

    print(
        "\nPROJECT 3 SPARSE-DATA CASES"
    )

    print(
        "--------------------------------"
    )

    for n_sensors in SENSOR_COUNTS:

        X_sensor, T_sensor, temperature_sensor = (
            create_sparse_measurements(n_sensors)
        )

        n_measurements = temperature_sensor.size

        sparsity = 16 / n_sensors

        print(
            f"Sensors = {n_sensors:2d} | "
            f"S = {sparsity:4.1f} | "
            f"Measurements = {n_measurements}"
        )


def verify_reference_solution():
    """
    Perform simple checks on the analytical solution.
    """

    t_test = np.linspace(
        0.0,
        1.0,
        100
    )

    left_boundary = exact_temperature(
        np.zeros_like(t_test),
        t_test
    )

    right_boundary = exact_temperature(
        np.ones_like(t_test),
        t_test
    )

    print(
        "\nREFERENCE-SOLUTION CHECK"
    )

    print(
        "--------------------------------"
    )

    print(
        "Maximum left boundary error:",
        np.max(
            np.abs(left_boundary)
        )
    )

    print(
        "Maximum right boundary error:",
        np.max(
            np.abs(right_boundary)
        )
    )


def main():

    print(
        "Project 3: Sparse-Data PINN Reconstruction"
    )

    # --------------------------------------------------------
    # Generate analytical ground truth
    # --------------------------------------------------------

    x, t, X, T, temperature = (
        create_reference_field()
    )

    # --------------------------------------------------------
    # Verify analytical solution
    # --------------------------------------------------------

    verify_reference_solution()

    # --------------------------------------------------------
    # Print sparse-data cases
    # --------------------------------------------------------

    print_case_information()

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    plot_reference_field(
        X,
        T,
        temperature
    )

    plot_sparse_measurements()

    print(
        "\nFigures saved:"
    )

    print(
        "figures/project3_reference_field.png"
    )

    print(
        "figures/project3_sparse_measurements.png"
    )


if __name__ == "__main__":
    main()
