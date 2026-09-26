# Project II: Ill-Conditioning in a Mixed-Stiffness Spring System

## 1. Problem Identification and Motivation

Mechanical systems often contain components with significantly different stiffnesses. Examples include compliant mechanisms, structural supports, suspension components, and assemblies containing both flexible and rigid members. These differences in stiffness can create numerical difficulties when solving for the equilibrium configuration of the system.

For this project, we consider a one-dimensional chain of springs connected between two fixed supports. The system contains alternating soft and stiff springs, with movable nodes between the springs. External forces are applied to the nodes, causing the system to deform until it reaches mechanical equilibrium.

The equilibrium configuration can be found by minimizing the total potential energy of the spring system. When the difference between the stiff and soft spring constants becomes large, the resulting optimization problem becomes increasingly ill-conditioned. This provides a simple mechanical system for studying how ill-conditioning affects the convergence of optimization algorithms.

The primary parameter investigated in this project is the stiffness ratio

$$
r = \frac{k_{\text{stiff}}}{k_{\text{soft}}}.
$$

By increasing this ratio, we will investigate how stiffness contrast affects the condition number of the optimization problem and the convergence behavior of gradient descent. We will then apply an optimization method intended to reduce the negative effects of ill-conditioning and compare its performance with standard gradient descent.

## 2. Formulation

### 2.1 Physical Model and Design Variables

The system is modeled as a one-dimensional chain containing three movable nodes and four linear springs connected between two fixed supports. The springs alternate between soft and stiff members.

The displacement of each movable node is represented by

$$
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix},
$$

where $x_i$ is the horizontal displacement of node $i$ from its unloaded position. These nodal displacements are the design variables of the optimization problem.

The spring constants are defined as

$$
k_1 = k_3 = k_{\text{soft}},
$$

$$
k_2 = k_4 = k_{\text{stiff}},
$$

with the stiffness ratio

$$
r = \frac{k_{\text{stiff}}}{k_{\text{soft}}}.
$$

The spring constants are treated as fixed parameters during each optimization problem. The ratio $r$ will be varied between experiments to study its effect on numerical conditioning.

### 2.2 Objective Function

For a linear spring with stiffness $k$ and displacement $\Delta x$, the stored elastic potential energy is

$$
U_s = \frac{1}{2}k(\Delta x)^2.
$$

For the four-spring system, the total elastic energy is

$$
U_s(\mathbf{x}) =
\frac{1}{2}k_1x_1^2
+\frac{1}{2}k_2(x_2-x_1)^2
+\frac{1}{2}k_3(x_3-x_2)^2
+\frac{1}{2}k_4x_3^2.
$$

If external nodal forces are represented by

$$
\mathbf{f} =
\begin{bmatrix}
f_1 \\
f_2 \\
f_3
\end{bmatrix},
$$

the total potential energy of the system is

$$
\Pi(\mathbf{x}) = U_s(\mathbf{x})-\mathbf{f}^T\mathbf{x}.
$$

The optimization problem is therefore

$$
\boxed{
\min_{\mathbf{x}} \; \Pi(\mathbf{x})
}
$$

The minimum of the total potential energy corresponds to the mechanical equilibrium configuration of the spring system. Because the two end supports are fixed, the system has no rigid-body translation and no additional displacement constraints are required for this simplified model.

### 2.3 Matrix Formulation

The total potential energy can be written in quadratic matrix form as

$$\Pi(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T K\mathbf{x} - \mathbf{f}^T\mathbf{x}$$

where $K$ is the stiffness matrix of the spring system. For the three-node, four-spring model,

$$
K =
\begin{bmatrix}
k_1+k_2 & -k_2 & 0 \\
-k_2 & k_2+k_3 & -k_3 \\
0 & -k_3 & k_3+k_4
\end{bmatrix}
$$

The diagonal entries represent the sum of the spring stiffnesses connected to each movable node. The off-diagonal entries represent coupling between adjacent nodes.

Taking the gradient of the objective gives

$$
\nabla \Pi(\mathbf{x}) = K\mathbf{x}-\mathbf{f}
$$

At the minimum-energy configuration,

$$
\nabla \Pi(\mathbf{x}^*)=0
$$

which gives

$$
K\mathbf{x}^*=\mathbf{f}
$$

Therefore, minimizing the total potential energy is equivalent to solving the mechanical equilibrium equations for the spring system.

The Hessian of the objective is

$$
H = \nabla^2\Pi(\mathbf{x}) = K
$$

Because the Hessian is equal to the stiffness matrix, the numerical conditioning of the optimization problem can be studied directly through the eigenvalues of $K$. For a symmetric positive-definite stiffness matrix, the 2-norm condition number is

$$
\kappa(K) =
\frac{\lambda_{\max}(K)}
{\lambda_{\min}(K)}
$$

This relationship will be used to investigate how increasing the stiffness contrast affects the conditioning of the optimization problem.

## 3. Ill-Conditioning Mechanism

### 3.1 Small Numerical Verification Case

Before studying larger stiffness ratios, a small three-node system was used to verify the stiffness matrix formulation and numerical calculations.

For the initial case,

$$
k_{\text{soft}} = 1
$$

and

$$
k_{\text{stiff}} = 10
$$

giving a stiffness ratio of

$$
r = \frac{k_{\text{stiff}}}{k_{\text{soft}}} = 10.
$$

The resulting stiffness matrix was

$$
K =
\begin{bmatrix}
11 & -10 & 0 \\
-10 & 11 & -1 \\
0 & -1 & 11
\end{bmatrix}.
$$

The computed eigenvalues were approximately

$$
\lambda =
\begin{bmatrix}
0.9501,\ 11.0000,\ 21.0499
\end{bmatrix}.
$$

Using the largest and smallest eigenvalues, the 2-norm condition number was

$$
\kappa(K)
=
\frac{21.0499}{0.9501}
\approx 22.15.
$$

For an external force vector

$$
\mathbf{f} =
\begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix},
$$

the equilibrium displacement was

$$
\mathbf{x}^* =
\begin{bmatrix}
0.50 \\
0.55 \\
0.05
\end{bmatrix}.
$$

Substituting this solution into the gradient expression

$$
\nabla\Pi(\mathbf{x}) = K\mathbf{x}-\mathbf{f}
$$

produced values on the order of $10^{-16}$, which is effectively zero within floating-point precision. This verifies that the computed displacement corresponds to the minimum-energy equilibrium configuration.

This initial case confirms that the numerical implementation is consistent with the analytical stiffness matrix formulation. The next step is to systematically increase the stiffness ratio and measure how the eigenvalue spread and condition number change.

## 4. Effect of Ill-Conditioning

## 5. Proposed Solution and Demonstration

## 6. Assumptions and Simplifications
