import numpy as np

# --------------------------------------------------
# Project II: Mixed-Stiffness Spring System
# Small 3-node / 4-spring verification case
# --------------------------------------------------

# Spring stiffness values
k_soft = 1.0
k_stiff = 10.0

# Alternating soft and stiff springs
k1 = k_soft
k2 = k_stiff
k3 = k_soft
k4 = k_stiff

# Assemble the 3x3 stiffness matrix
K = np.array([
    [k1 + k2,   -k2,          0.0],
    [-k2,        k2 + k3,    -k3],
    [0.0,       -k3,          k3 + k4]
])

# Simple external force vector
f = np.array([0.0, 1.0, 0.0])

# Compute eigenvalues of the stiffness matrix
eigenvalues = np.linalg.eigvalsh(K)

# Compute the 2-norm condition number
condition_number = np.linalg.cond(K, 2)

# Solve Kx = f for the equilibrium displacement
x_star = np.linalg.solve(K, f)

# Compute the gradient at the solution
gradient_at_solution = K @ x_star - f

# Display results
print("Stiffness matrix K:")
print(K)

print("\nEigenvalues of K:")
print(eigenvalues)

print("\nCondition number of K:")
print(condition_number)

print("\nEquilibrium displacement x*:")
print(x_star)

print("\nGradient at x*:")
print(gradient_at_solution)
