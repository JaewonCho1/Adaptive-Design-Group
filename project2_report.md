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

## 3. Ill-Conditioning Mechanism

## 4. Effect of Ill-Conditioning

## 5. Proposed Solution and Demonstration

## 6. Assumptions and Simplifications
