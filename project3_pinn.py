from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# Project 3: Sparse-Data Heat-Equation Reconstruction
# Step 2: First Physics-Informed Neural Network
# ============================================================

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------

ALPHA = 0.01

X_MIN = 0.0
X_MAX = 1.0

T_MIN = 0.0
T_MAX = 1.0

SENSOR_COUNTS = [16, 8, 4, 2]


# ------------------------------------------------------------
# PINN settings
# ------------------------------------------------------------

PINN_SENSOR_COUNT = 16

EPOCHS = 3000

LEARNING_RATE = 1e-3

N_COLLOCATION = 2000

N_BOUNDARY = 200


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# Analytical Reference Solution
# ============================================================

def exact_temperature(x, t, alpha=ALPHA):
    """
    Analytical solution of the 1D heat equation:

        T_t = alpha * T_xx

    with boundary conditions

        T(0,t) = 0
        T(1,t) = 0

    and initial condition

        T(x,0) = sin(pi*x) + 0.5*sin(4*pi*x)
    """

    term_1 = (
        np.sin(np.pi * x)
        * np.exp(-alpha * np.pi**2 * t)
    )

    term_2 = (
        0.5
        * np.sin(4.0 * np.pi * x)
        * np.exp(-16.0 * alpha * np.pi**2 * t)
    )

    return term_1 + term_2


# ============================================================
# Reference Field
# ============================================================

def create_reference_field():
    """
    Create the dense ground-truth field used only
    for evaluation.
    """

    x = np.linspace(
        X_MIN,
        X_MAX,
        201
    )

    t = np.linspace(
        T_MIN,
        T_MAX,
        101
    )

    X, T = np.meshgrid(
        x,
        t
    )

    temperature = exact_temperature(
        X,
        T
    )

    return x, t, X, T, temperature


# ============================================================
# Sparse Measurements
# ============================================================

def create_sparse_measurements(
    n_sensors,
    n_time_samples=21
):
    """
    Generate synthetic sparse measurements.
    """

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

    return (
        X_sensor,
        T_sensor,
        temperature_sensor
    )


# ============================================================
# Training Data
# ============================================================

def create_training_data(n_sensors):
    """
    Create the four groups of points used by the PINN:

    1. sparse measurement data
    2. sparse initial-condition data
    3. boundary-condition points
    4. PDE collocation points
    """

    # --------------------------------------------------------
    # Sparse measurement data
    # --------------------------------------------------------

    X_sensor, T_sensor, temp_sensor = (
        create_sparse_measurements(
            n_sensors
        )
    )

    # Exclude t = 0 from general measurement data.
    # Those points will be tracked separately as IC loss.
    mask = T_sensor > 0.0

    x_data = X_sensor[mask].reshape(-1, 1)

    t_data = T_sensor[mask].reshape(-1, 1)

    y_data = temp_sensor[mask].reshape(-1, 1)


    # --------------------------------------------------------
    # Sparse initial-condition points
    # --------------------------------------------------------

    x_ic = np.linspace(
        X_MIN,
        X_MAX,
        n_sensors + 2
    )[1:-1].reshape(-1, 1)

    t_ic = np.zeros_like(
        x_ic
    )

    y_ic = exact_temperature(
        x_ic,
        t_ic
    )


    # --------------------------------------------------------
    # Boundary-condition points
    # --------------------------------------------------------

    t_bc = np.linspace(
        T_MIN,
        T_MAX,
        N_BOUNDARY
    ).reshape(-1, 1)

    x_left = np.zeros_like(
        t_bc
    )

    x_right = np.ones_like(
        t_bc
    )

    y_bc = np.zeros_like(
        t_bc
    )


    # --------------------------------------------------------
    # PDE collocation points
    # --------------------------------------------------------

    x_f = np.random.uniform(
        X_MIN,
        X_MAX,
        (N_COLLOCATION, 1)
    )

    t_f = np.random.uniform(
        T_MIN,
        T_MAX,
        (N_COLLOCATION, 1)
    )


    return {
        "x_data": x_data,
        "t_data": t_data,
        "y_data": y_data,

        "x_ic": x_ic,
        "t_ic": t_ic,
        "y_ic": y_ic,

        "x_left": x_left,
        "x_right": x_right,
        "t_bc": t_bc,
        "y_bc": y_bc,

        "x_f": x_f,
        "t_f": t_f
    }


# ============================================================
# Tensor Helper
# ============================================================

def to_tensor(array, requires_grad=False):
    """
    Convert NumPy arrays into PyTorch tensors.
    """

    return torch.tensor(
        array,
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=requires_grad
    )


# ============================================================
# PINN Network
# ============================================================

class PINN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(2, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 1)
        )

    def forward(self, x, t):

        inputs = torch.cat(
            (x, t),
            dim=1
        )

        return self.network(
            inputs
        )


# ============================================================
# PDE Residual
# ============================================================

def calculate_pde_residual(
    model,
    x,
    t
):
    """
    Calculate

        T_t - alpha*T_xx

    using automatic differentiation.
    """

    x.requires_grad_(True)
    t.requires_grad_(True)

    temperature = model(
        x,
        t
    )

    temperature_t = torch.autograd.grad(
        temperature,
        t,
        grad_outputs=torch.ones_like(temperature),
        create_graph=True
    )[0]

    temperature_x = torch.autograd.grad(
        temperature,
        x,
        grad_outputs=torch.ones_like(temperature),
        create_graph=True
    )[0]

    temperature_xx = torch.autograd.grad(
        temperature_x,
        x,
        grad_outputs=torch.ones_like(temperature_x),
        create_graph=True
    )[0]

    residual = (
        temperature_t
        - ALPHA * temperature_xx
    )

    return residual


# ============================================================
# PINN Training
# ============================================================

def train_pinn(n_sensors):

    print(
        "\n========================================"
    )

    print(
        f"Training PINN with {n_sensors} sensors"
    )

    print(
        "========================================"
    )

    data = create_training_data(
        n_sensors
    )


    # --------------------------------------------------------
    # Convert training arrays to tensors
    # --------------------------------------------------------

    x_data = to_tensor(
        data["x_data"]
    )

    t_data = to_tensor(
        data["t_data"]
    )

    y_data = to_tensor(
        data["y_data"]
    )


    x_ic = to_tensor(
        data["x_ic"]
    )

    t_ic = to_tensor(
        data["t_ic"]
    )

    y_ic = to_tensor(
        data["y_ic"]
    )


    x_left = to_tensor(
        data["x_left"]
    )

    x_right = to_tensor(
        data["x_right"]
    )

    t_bc = to_tensor(
        data["t_bc"]
    )

    y_bc = to_tensor(
        data["y_bc"]
    )


    x_f = to_tensor(
        data["x_f"]
    )

    t_f = to_tensor(
        data["t_f"]
    )


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = PINN().to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    mse = nn.MSELoss()


    # --------------------------------------------------------
    # Loss history
    # --------------------------------------------------------

    history = {
        "total": [],
        "pde": [],
        "bc": [],
        "ic": [],
        "data": []
    }


    # --------------------------------------------------------
    # Training timer
    # --------------------------------------------------------

    start_time = time.perf_counter()


    # ========================================================
    # Optimization Loop
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Measurement loss
        # ----------------------------------------------------

        prediction_data = model(
            x_data,
            t_data
        )

        loss_data = mse(
            prediction_data,
            y_data
        )


        # ----------------------------------------------------
        # Initial-condition loss
        # ----------------------------------------------------

        prediction_ic = model(
            x_ic,
            t_ic
        )

        loss_ic = mse(
            prediction_ic,
            y_ic
        )


        # ----------------------------------------------------
        # Boundary-condition loss
        # ----------------------------------------------------

        prediction_left = model(
            x_left,
            t_bc
        )

        prediction_right = model(
            x_right,
            t_bc
        )

        loss_bc = (
            mse(
                prediction_left,
                y_bc
            )
            +
            mse(
                prediction_right,
                y_bc
            )
        )


        # ----------------------------------------------------
        # PDE residual loss
        # ----------------------------------------------------

        residual = calculate_pde_residual(
            model,
            x_f,
            t_f
        )

        loss_pde = torch.mean(
            residual**2
        )


        # ----------------------------------------------------
        # Equal-weight initial PINN loss
        # ----------------------------------------------------

        total_loss = (
            loss_data
            + loss_ic
            + loss_bc
            + loss_pde
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        total_loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Save loss history
        # ----------------------------------------------------

        history["total"].append(
            total_loss.item()
        )

        history["data"].append(
            loss_data.item()
        )

        history["ic"].append(
            loss_ic.item()
        )

        history["bc"].append(
            loss_bc.item()
        )

        history["pde"].append(
            loss_pde.item()
        )


        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 250 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:4d} | "
                f"Total = {total_loss.item():.3e} | "
                f"PDE = {loss_pde.item():.3e} | "
                f"BC = {loss_bc.item():.3e} | "
                f"IC = {loss_ic.item():.3e} | "
                f"Data = {loss_data.item():.3e}"
            )


    training_time = (
        time.perf_counter()
        - start_time
    )


    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    return (
        model,
        history,
        training_time
    )


# ============================================================
# PINN Evaluation
# ============================================================

def evaluate_pinn(
    model,
    X,
    T,
    true_temperature
):

    x_eval = X.reshape(
        -1,
        1
    )

    t_eval = T.reshape(
        -1,
        1
    )

    x_tensor = to_tensor(
        x_eval
    )

    t_tensor = to_tensor(
        t_eval
    )


    # --------------------------------------------------------
    # Inference timer
    # --------------------------------------------------------

    start_time = time.perf_counter()

    model.eval()

    with torch.no_grad():

        prediction = model(
            x_tensor,
            t_tensor
        )

    inference_time = (
        time.perf_counter()
        - start_time
    )


    prediction = (
        prediction
        .cpu()
        .numpy()
        .reshape(
            true_temperature.shape
        )
    )


    # --------------------------------------------------------
    # Relative L2 error
    # --------------------------------------------------------

    relative_l2 = (
        np.linalg.norm(
            prediction
            - true_temperature
        )
        /
        np.linalg.norm(
            true_temperature
        )
    )


    print(
        "\nPINN EVALUATION"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Relative L2 error: "
        f"{relative_l2:.6f}"
    )

    print(
        f"Inference time: "
        f"{inference_time:.6f} seconds"
    )


    return (
        prediction,
        relative_l2,
        inference_time
    )


# ============================================================
# Plot Reference Field
# ============================================================

def plot_reference_field(
    X,
    T,
    temperature
):

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
        FIGURE_DIR
        / "project3_reference_field.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Plot Sparse Measurements
# ============================================================

def plot_sparse_measurements():

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

    vmin = -0.2
    vmax = 1.45

    scatter = None


    for ax, n_sensors in zip(
        axes,
        SENSOR_COUNTS
    ):

        (
            X_sensor,
            T_sensor,
            temperature_sensor
        ) = create_sparse_measurements(
            n_sensors
        )

        scatter = ax.scatter(
            X_sensor.flatten(),
            T_sensor.flatten(),
            c=temperature_sensor.flatten(),
            s=28,
            vmin=vmin,
            vmax=vmax
        )

        sparsity = (
            16 / n_sensors
        )

        ax.set_title(
            f"{n_sensors} Sensors, "
            f"S = {sparsity:g}"
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


    cbar_axis = fig.add_axes(
        [
            0.88,
            0.18,
            0.025,
            0.62
        ]
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
        FIGURE_DIR
        / "project3_sparse_measurements.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Plot Training Loss
# ============================================================

def plot_loss_history(
    history,
    n_sensors
):

    epochs = np.arange(
        1,
        len(history["total"]) + 1
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.semilogy(
        epochs,
        history["total"],
        label="Total"
    )

    ax.semilogy(
        epochs,
        history["pde"],
        label="PDE"
    )

    ax.semilogy(
        epochs,
        history["bc"],
        label="Boundary"
    )

    ax.semilogy(
        epochs,
        history["ic"],
        label="Initial"
    )

    ax.semilogy(
        epochs,
        history["data"],
        label="Measurement"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Mean-Squared Loss"
    )

    ax.set_title(
        f"PINN Loss History - "
        f"{n_sensors} Sensors"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR
        / f"project3_pinn_{n_sensors}_loss.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Plot Reconstruction
# ============================================================

def plot_reconstruction(
    X,
    T,
    true_temperature,
    prediction,
    n_sensors,
    relative_l2
):

    error = np.abs(
        prediction
        - true_temperature
    )

    vmin = np.min(
        true_temperature
    )

    vmax = np.max(
        true_temperature
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        constrained_layout=True
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    im0 = axes[0].contourf(
        X,
        T,
        true_temperature,
        levels=50,
        vmin=vmin,
        vmax=vmax
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


    # --------------------------------------------------------
    # PINN reconstruction
    # --------------------------------------------------------

    im1 = axes[1].contourf(
        X,
        T,
        prediction,
        levels=50,
        vmin=vmin,
        vmax=vmax
    )

    axes[1].set_title(
        "PINN Reconstruction"
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


    # --------------------------------------------------------
    # Absolute error
    # --------------------------------------------------------

    im2 = axes[2].contourf(
        X,
        T,
        error,
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
        f"{n_sensors}-Sensor PINN Reconstruction "
        f"| Relative L2 Error = "
        f"{relative_l2:.4f}"
    )


    fig.savefig(
        FIGURE_DIR
        / f"project3_pinn_{n_sensors}_reconstruction.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# Verification
# ============================================================

def verify_reference_solution():

    t_test = np.linspace(
        0.0,
        1.0,
        100
    )

    left_boundary = (
        exact_temperature(
            np.zeros_like(t_test),
            t_test
        )
    )

    right_boundary = (
        exact_temperature(
            np.ones_like(t_test),
            t_test
        )
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
            np.abs(
                left_boundary
            )
        )
    )

    print(
        "Maximum right boundary error:",
        np.max(
            np.abs(
                right_boundary
            )
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\nProject 3: "
        "Sparse-Data PINN Reconstruction"
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    (
        x,
        t,
        X,
        T,
        true_temperature
    ) = create_reference_field()


    verify_reference_solution()


    # --------------------------------------------------------
    # Existing figures
    # --------------------------------------------------------

    plot_reference_field(
        X,
        T,
        true_temperature
    )

    plot_sparse_measurements()


    # --------------------------------------------------------
    # Train first PINN
    # --------------------------------------------------------

    (
        model,
        history,
        training_time
    ) = train_pinn(
        PINN_SENSOR_COUNT
    )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PINN figures
    # --------------------------------------------------------

    plot_loss_history(
        history,
        PINN_SENSOR_COUNT
    )

    plot_reconstruction(
        X,
        T,
        true_temperature,
        prediction,
        PINN_SENSOR_COUNT,
        relative_l2
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "PROJECT 3 FIRST PINN SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Sensors: "
        f"{PINN_SENSOR_COUNT}"
    )

    print(
        f"Relative L2 error: "
        f"{relative_l2:.6f}"
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Inference time: "
        f"{inference_time:.6f} seconds"
    )

    print(
        "\nNew figures:"
    )

    print(
        f"figures/"
        f"project3_pinn_"
        f"{PINN_SENSOR_COUNT}_loss.png"
    )

    print(
        f"figures/"
        f"project3_pinn_"
        f"{PINN_SENSOR_COUNT}_reconstruction.png"
    )


if __name__ == "__main__":
    main()
