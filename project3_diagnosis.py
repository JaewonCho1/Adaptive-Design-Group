from pathlib import Path
import csv
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from project3_pinn import (
    PINN,
    DEVICE,
    SEED,
    create_reference_field,
    to_tensor,
)


# ============================================================
# Project 3 Final Path-B Diagnosis
#
# 1. P4 Fourier spectrum OF THE ERROR
# 2. Controlled experiment:
#    supervised regression with the same network architecture
#
# This separates network expressivity from PINN training failure.
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

PINN_PREDICTION_FILE = Path(
    "project3_pinn_2_prediction.npy"
)

DIAGNOSIS_RESULTS_FILE = Path(
    "project3_diagnosis_summary.csv"
)


# ------------------------------------------------------------
# Supervised-control settings
# ------------------------------------------------------------

SUPERVISED_STEPS = 3000

LEARNING_RATE = 1e-3

TRAINING_POINTS = 5000

BATCH_SIZE = 512


# ============================================================
# Fourier Sine Spectrum
# ============================================================

def sine_spectrum(
    profile,
    x,
    n_modes=12
):
    """
    Calculate absolute Fourier sine-mode amplitudes
    for a spatial profile.
    """

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
# P4: Fourier Spectrum of the ERROR
# ============================================================

def plot_error_spectrum(
    x,
    truth,
    pinn_prediction
):
    """
    Compute the Fourier sine spectrum of the signed
    reconstruction error at t = 0.
    """

    # First time row corresponds to t = 0
    error_profile = (
        pinn_prediction[0, :]
        - truth[0, :]
    )

    spectrum = sine_spectrum(
        error_profile,
        x,
        n_modes=12
    )

    modes = np.arange(
        1,
        13
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        modes,
        spectrum,
        marker="o",
        linewidth=2
    )

    ax.set_xlabel(
        "Spatial Fourier Sine Mode"
    )

    ax.set_ylabel(
        "Absolute Error-Mode Amplitude"
    )

    ax.set_title(
        "Fourier Spectrum of 2-Sensor PINN Error at t = 0"
    )

    ax.set_xticks(
        modes
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "project3_p4_error_spectrum.png",
        dpi=300
    )

    plt.close(fig)


    print(
        "\nERROR-SPECTRUM CHECK"
    )

    print(
        "-----------------------------------------"
    )

    for mode, amplitude in zip(
        modes,
        spectrum
    ):

        print(
            f"Mode {mode:2d}: "
            f"{amplitude:.6f}"
        )

    return spectrum


# ============================================================
# Create Supervised Training Data
# ============================================================

def create_supervised_training_data(
    X,
    T,
    truth
):
    """
    Draw training points from the known reference field.

    This is intentionally NOT a PINN.

    The goal is to test whether the same neural-network
    architecture is capable of representing the solution
    when directly shown reference data.
    """

    x_all = X.reshape(
        -1,
        1
    )

    t_all = T.reshape(
        -1,
        1
    )

    y_all = truth.reshape(
        -1,
        1
    )

    total_points = len(
        x_all
    )

    n_train = min(
        TRAINING_POINTS,
        total_points
    )

    indices = np.random.choice(
        total_points,
        size=n_train,
        replace=False
    )

    x_train = x_all[
        indices
    ]

    t_train = t_all[
        indices
    ]

    y_train = y_all[
        indices
    ]

    return (
        x_train,
        t_train,
        y_train
    )


# ============================================================
# Controlled Supervised Experiment
# ============================================================

def train_supervised_control(
    X,
    T,
    truth
):
    """
    Train the SAME PINN network architecture using
    ordinary supervised regression.

    If this model reconstructs the reference solution well,
    then the architecture has enough expressive capacity.
    The sparse PINN failure is therefore associated with
    the physics-informed/sparse training process rather
    than an inability of the network to represent the field.
    """

    (
        x_train_np,
        t_train_np,
        y_train_np
    ) = create_supervised_training_data(
        X,
        T,
        truth
    )


    x_train = to_tensor(
        x_train_np
    )

    t_train = to_tensor(
        t_train_np
    )

    y_train = to_tensor(
        y_train_np
    )


    model = PINN().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    mse = nn.MSELoss()

    n_train = x_train.shape[0]

    loss_history = []

    start_time = time.perf_counter()


    print(
        "\n============================================"
    )

    print(
        "CONTROLLED EXPERIMENT:"
    )

    print(
        "SAME NETWORK + SUPERVISED REFERENCE DATA"
    )

    print(
        "============================================"
    )


    for step in range(
        1,
        SUPERVISED_STEPS + 1
    ):

        # Random mini-batch
        batch_indices = torch.randint(
            low=0,
            high=n_train,
            size=(BATCH_SIZE,),
            device=DEVICE
        )

        x_batch = x_train[
            batch_indices
        ]

        t_batch = t_train[
            batch_indices
        ]

        y_batch = y_train[
            batch_indices
        ]


        optimizer.zero_grad()


        prediction = model(
            x_batch,
            t_batch
        )


        loss = mse(
            prediction,
            y_batch
        )


        loss.backward()

        optimizer.step()


        loss_history.append(
            loss.item()
        )


        if (
            step == 1
            or step % 500 == 0
            or step == SUPERVISED_STEPS
        ):

            print(
                f"Step {step:4d} | "
                f"Supervised Loss = "
                f"{loss.item():.3e}"
            )


    training_time = (
        time.perf_counter()
        - start_time
    )


    # --------------------------------------------------------
    # Evaluate on entire field
    # --------------------------------------------------------

    x_eval = to_tensor(
        X.reshape(
            -1,
            1
        )
    )

    t_eval = to_tensor(
        T.reshape(
            -1,
            1
        )
    )


    model.eval()

    with torch.no_grad():

        prediction = model(
            x_eval,
            t_eval
        )


    prediction = (
        prediction
        .cpu()
        .numpy()
        .reshape(
            truth.shape
        )
    )


    relative_l2 = (
        np.linalg.norm(
            prediction - truth
        )
        /
        np.linalg.norm(
            truth
        )
    )


    print(
        "\nSUPERVISED CONTROL RESULT"
    )

    print(
        "-----------------------------------------"
    )

    print(
        f"Relative L2 error: "
        f"{relative_l2:.6f}"
    )

    print(
        f"Percent error: "
        f"{100.0 * relative_l2:.4f}%"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )


    return (
        prediction,
        relative_l2,
        training_time,
        loss_history
    )


# ============================================================
# Plot Supervised Control
# ============================================================

def plot_supervised_control(
    X,
    T,
    truth,
    supervised_prediction,
    supervised_error
):

    absolute_error = np.abs(
        supervised_prediction
        - truth
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        constrained_layout=True
    )


    # Ground truth
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


    # Supervised network
    im1 = axes[1].contourf(
        X,
        T,
        supervised_prediction,
        levels=50
    )

    axes[1].set_title(
        "Supervised Network"
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


    # Absolute error
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
        "Controlled Expressivity Test "
        f"| Relative L2 Error = "
        f"{supervised_error:.4f}"
    )


    fig.savefig(
        FIGURE_DIR
        / "project3_supervised_control.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Plot Supervised Loss
# ============================================================

def plot_supervised_loss(
    loss_history
):

    steps = np.arange(
        1,
        len(loss_history) + 1
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        steps,
        loss_history
    )

    ax.set_xlabel(
        "Optimization Step"
    )

    ax.set_ylabel(
        "Supervised MSE Loss"
    )

    ax.set_title(
        "Supervised-Control Training Loss"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / "project3_supervised_control_loss.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Save Diagnostic Summary
# ============================================================

def save_diagnosis_summary(
    pinn_error,
    supervised_error,
    supervised_training_time,
    error_spectrum
):

    with open(
        DIAGNOSIS_RESULTS_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "2-Sensor PINN Error (%)",
            "Supervised Same-Network Error (%)",
            "Supervised Training Time (s)",
            "Error Spectrum Mode 1",
            "Error Spectrum Mode 4"
        ])

        writer.writerow([
            100.0 * pinn_error,
            100.0 * supervised_error,
            supervised_training_time,
            error_spectrum[0],
            error_spectrum[3]
        ])


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\nProject 3 Final Path-B Diagnosis"
    )

    print(
        "Using device:",
        DEVICE
    )


    if not PINN_PREDICTION_FILE.exists():

        raise FileNotFoundError(
            "project3_pinn_2_prediction.npy "
            "was not found."
        )


    (
        x,
        t,
        X,
        T,
        truth
    ) = create_reference_field()


    # --------------------------------------------------------
    # Existing sparse PINN result
    # --------------------------------------------------------

    pinn_prediction = np.load(
        PINN_PREDICTION_FILE
    )

    pinn_relative_l2 = (
        np.linalg.norm(
            pinn_prediction - truth
        )
        /
        np.linalg.norm(
            truth
        )
    )


    print(
        "\nExisting 2-sensor PINN error:"
    )

    print(
        f"{100.0 * pinn_relative_l2:.4f}%"
    )


    # --------------------------------------------------------
    # P4 required Fourier spectrum OF ERROR
    # --------------------------------------------------------

    error_spectrum = plot_error_spectrum(
        x,
        truth,
        pinn_prediction
    )


    # --------------------------------------------------------
    # Path-B controlled experiment
    # --------------------------------------------------------

    (
        supervised_prediction,
        supervised_error,
        supervised_training_time,
        loss_history
    ) = train_supervised_control(
        X,
        T,
        truth
    )


    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    plot_supervised_control(
        X,
        T,
        truth,
        supervised_prediction,
        supervised_error
    )

    plot_supervised_loss(
        loss_history
    )


    # --------------------------------------------------------
    # Summary file
    # --------------------------------------------------------

    save_diagnosis_summary(
        pinn_relative_l2,
        supervised_error,
        supervised_training_time,
        error_spectrum
    )


    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "PATH-B CONTROLLED DIAGNOSIS"
    )

    print(
        "============================================"
    )

    print(
        f"2-sensor PINN error: "
        f"{100.0 * pinn_relative_l2:.4f}%"
    )

    print(
        f"Same-network supervised error: "
        f"{100.0 * supervised_error:.4f}%"
    )


    if supervised_error < pinn_relative_l2:

        print(
            "\nThe same network architecture performs "
            "better when trained directly on reference data."
        )

        print(
            "This indicates that network expressivity alone "
            "does not explain the sparse PINN failure."
        )

        print(
            "The failure is associated with the sparse "
            "physics-informed training problem."
        )

    else:

        print(
            "\nThe supervised network did not substantially "
            "outperform the sparse PINN."
        )

        print(
            "Network expressivity may therefore contribute "
            "to the observed failure."
        )


    print(
        "\nGenerated:"
    )

    print(
        "figures/project3_p4_error_spectrum.png"
    )

    print(
        "figures/project3_supervised_control.png"
    )

    print(
        "figures/project3_supervised_control_loss.png"
    )

    print(
        "project3_diagnosis_summary.csv"
    )


if __name__ == "__main__":
    main()
