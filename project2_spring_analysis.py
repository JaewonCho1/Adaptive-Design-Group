import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Project II: Mixed-Stiffness Spring System
# Conditioning study
# --------------------------------------------------

# Soft spring stiffness is held constant
k_soft = 1.0

# Stiffness ratios to test
ratios = np.array([1, 10, 100, 1000, 10000], dtype=float)

# Lists for storing results
condition_numbers = []
smallest_eigenvalues = []
largest_eigenvalues = []
scaled_condition_numbers = []

print("Mixed-Stiffness Spring Conditioning Study")
print("-" * 50)

for r in ratios:

    # Define alternating soft and stiff springs
    k1 = k_soft
    k2 = r * k_soft
    k3 = k_soft
    k4 = r * k_soft

    # Assemble stiffness matrix
    K = np.array([
        [k1 + k2, -k2,       0.0],
        [-k2,      k2 + k3, -k3],
        [0.0,     -k3,       k3 + k4]
    ])

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(K)

    # Smallest and largest eigenvalues
    lambda_min = eigenvalues[0]
    lambda_max = eigenvalues[-1]

    # Compute 2-norm condition number
    kappa = np.linalg.cond(K, 2)

    # Jacobi / diagonal rescaling for intrinsic-conditioning test
    diagonal = np.diag(K)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(diagonal))

    K_scaled = D_inv_sqrt @ K @ D_inv_sqrt

    scaled_kappa = np.linalg.cond(K_scaled, 2)

    # Store results
    smallest_eigenvalues.append(lambda_min)
    largest_eigenvalues.append(lambda_max)
    condition_numbers.append(kappa)
    scaled_condition_numbers.append(scaled_kappa)

    # Print results for this stiffness ratio
    print(f"\nStiffness ratio r = {r:.0f}")
    print("Eigenvalues:", eigenvalues)
    print(f"Condition number = {kappa:.6f}")
    print(
        f"Condition number after diagonal scaling = "
        f"{scaled_kappa:.6f}"
    )

# Convert lists to NumPy arrays
condition_numbers = np.array(condition_numbers)
smallest_eigenvalues = np.array(smallest_eigenvalues)
largest_eigenvalues = np.array(largest_eigenvalues)
scaled_condition_numbers = np.array(scaled_condition_numbers)

# --------------------------------------------------
# Plot condition number versus stiffness ratio
# --------------------------------------------------

plt.figure()

plt.loglog(
    ratios,
    condition_numbers,
    marker="o"
)

plt.xlabel("Stiffness Ratio r = k_stiff / k_soft")
plt.ylabel("Condition Number kappa(K)")
plt.title("Effect of Stiffness Contrast on Conditioning")

plt.grid(True, which="both")

plt.tight_layout()
plt.savefig("figures/project2_condition_number.png", dpi=300)
plt.show()

# --------------------------------------------------
# Plot condition number before and after scaling
# --------------------------------------------------

plt.figure()

plt.loglog(
    ratios,
    condition_numbers,
    marker="o",
    label="Original K"
)

plt.loglog(
    ratios,
    scaled_condition_numbers,
    marker="s",
    label="After diagonal scaling"
)

plt.xlabel("Stiffness Ratio r = k_stiff / k_soft")
plt.ylabel("Condition Number")
plt.title("Intrinsic Conditioning Test")

plt.grid(True, which="both")
plt.legend()

plt.tight_layout()
plt.savefig(
    "figures/project2_intrinsic_conditioning.png",
    dpi=300
)
plt.show()


# --------------------------------------------------
# Gradient descent study
# --------------------------------------------------

def objective(x, K, f):
    return 0.5 * x @ K @ x - f @ x


def gradient(x, K, f):
    return K @ x - f


def gradient_descent(K, f, x0, tol=1e-8, max_iterations=200000):
    # Eigenvalues determine a stable step size for this quadratic
    eigenvalues = np.linalg.eigvalsh(K)

    lambda_min = eigenvalues[0]
    lambda_max = eigenvalues[-1]

    # Near-optimal constant step size for SPD quadratic
    alpha = 2.0 / (lambda_min + lambda_max)

    x = x0.copy()

    # Exact solution used only as a reference
    x_star = np.linalg.solve(K, f)

    objective_errors = []
    gradient_norms = []

    initial_gradient_norm = np.linalg.norm(gradient(x, K, f))

    for iteration in range(max_iterations):

        g = gradient(x, K, f)

        error = x - x_star
        objective_error = 0.5 * error @ K @ error
        gradient_norm = np.linalg.norm(g)
        
        objective_errors.append(max(objective_error, 1e-14))
        gradient_norms.append(max(gradient_norm, 1e-14))

        # Relative stopping condition
        if gradient_norm <= tol * initial_gradient_norm:
            break

        x = x - alpha * g

    return (
        x,
        np.array(objective_errors),
        np.array(gradient_norms),
        iteration + 1
    )


# Ratios used for the convergence comparison
gd_ratios = [1, 10, 100, 1000]

# Force applied to the middle node
f = np.array([0.0, 1.0, 0.0])

# Same starting point for every experiment
x0 = np.zeros(3)

# Store convergence histories
gd_results = {}

print("\nGradient Descent Convergence Study")
print("-" * 50)

for r in gd_ratios:

    k1 = k_soft
    k2 = r * k_soft
    k3 = k_soft
    k4 = r * k_soft

    K = np.array([
        [k1 + k2, -k2,       0.0],
        [-k2,      k2 + k3, -k3],
        [0.0,     -k3,       k3 + k4]
    ])

    x_final, objective_errors, gradient_norms, iterations = (
        gradient_descent(K, f, x0)
    )

    gd_results[r] = {
        "objective_errors": objective_errors,
        "gradient_norms": gradient_norms,
        "iterations": iterations
    }

    print(
        f"r = {r:4d} | "
        f"kappa = {np.linalg.cond(K, 2):10.2f} | "
        f"iterations = {iterations}"
    )


# --------------------------------------------------
# Plot objective error convergence
# --------------------------------------------------

plt.figure()

for r in gd_ratios:
    errors = gd_results[r]["objective_errors"]

    plt.semilogy(
        range(len(errors)),
        errors,
        label=f"r = {r}"
    )

plt.xlabel("Iteration")
plt.ylabel("Objective Error")
plt.title("Gradient Descent Convergence")

plt.ylim(bottom=1e-14)

plt.grid(True, which="both")
plt.legend()

plt.tight_layout()
plt.savefig(
    "figures/project2_gradient_descent_convergence.png",
    dpi=300
)
plt.show()


# --------------------------------------------------
# Plot gradient norm convergence
# --------------------------------------------------

plt.figure()

for r in gd_ratios:
    norms = gd_results[r]["gradient_norms"]

    plt.semilogy(
        range(len(norms)),
        norms,
        label=f"r = {r}"
    )

plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.title("Gradient Descent Gradient-Norm Convergence")

plt.ylim(bottom=1e-14)

plt.grid(True, which="both")
plt.legend()

plt.tight_layout()
plt.savefig(
    "figures/project2_gradient_norm_convergence.png",
    dpi=300
)
plt.show()
