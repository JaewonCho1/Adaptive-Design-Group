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

    # Store results
    smallest_eigenvalues.append(lambda_min)
    largest_eigenvalues.append(lambda_max)
    condition_numbers.append(kappa)

    # Print results for this stiffness ratio
    print(f"\nStiffness ratio r = {r:.0f}")
    print("Eigenvalues:", eigenvalues)
    print(f"Condition number = {kappa:.6f}")

# Convert lists to NumPy arrays
condition_numbers = np.array(condition_numbers)
smallest_eigenvalues = np.array(smallest_eigenvalues)
largest_eigenvalues = np.array(largest_eigenvalues)

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
