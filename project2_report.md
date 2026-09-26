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

## 3. Ill-Conditioning Mechanism

## 4. Effect of Ill-Conditioning

## 5. Proposed Solution and Demonstration

## 6. Assumptions and Simplifications
