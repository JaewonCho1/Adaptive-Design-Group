# Adaptive Design Group 

## About Our Group

We are the **Adaptive Design Group**, a team working together as part of the Design Optimization course. Throughout the semester, we will explore how mathematical modeling, computational tools, and optimization methods can be used to make better engineering design decisions.

Our work will include formulating design problems by identifying **design variables, objectives, and constraints**, implementing optimization methods, and analyzing the resulting solutions. Through the course projects, we will apply these concepts to real design problems while developing our skills in optimization, modeling, computation, and collaborative engineering design.

This repository will serve as a shared workspace for our project files, code, results, and documentation throughout the course.

## Project I: Lightweight Quadcopter Frame Design Optimization

Project I focuses on formulating and solving a constrained mechanical design optimization problem for a lightweight quadcopter frame.

The design variables are the arm length, outer width, outer height, and wall thickness:

$$
\mathbf{x} = [L,\ b,\ h,\ t]^T
$$

The objective is to minimize the structural mass of the four quadcopter arms while satisfying geometric and structural requirements including:

- valid hollow-arm geometry
- propeller clearance
- allowable bending stress
- maximum tip deflection
- specified bounds on the design variables

The complete formulation and discussion are contained in:

`project1_report.md`

The numerical optimization code is contained in:

`solve.py`

### Running the Project I Analysis

Project I uses Python and numerical optimization tools.

Install the required Python packages with:

```bash
pip install numpy scipy matplotlib
```

Then run:

```bash
python solve.py
```

The script evaluates the quadcopter design model, performs the numerical optimization, checks the design constraints, and generates the numerical results and figures used in the Project I report.

Project I is formulated as a constrained continuous nonlinear optimization problem.

---

## Project II: Ill-Conditioned Spring Optimization

Project II investigates how stiffness contrast in a mixed-stiffness spring system creates an ill-conditioned optimization problem and affects the convergence of optimization algorithms.

The physical model is a one-dimensional chain of alternating soft and stiff springs connected between fixed supports. The movable node displacements are optimized by minimizing the total potential energy of the system.

The main structural parameter is the stiffness ratio

$$r = \frac{k_{\text{stiff}}}{k_{\text{soft}}}$$

which is varied to study how increasing stiffness contrast affects the Hessian condition number and optimization convergence.

The complete Project II report is contained in:

`project2_report.md`

The numerical analysis code is contained in:

`project2_spring_analysis.py`

### Running the Project II Analysis

Project II requires Python with NumPy and Matplotlib.

Install the required Python packages with:

```bash
pip install numpy matplotlib
```

Then run:

```bash
python project2_spring_analysis.py
```

The script reproduces the main numerical results used in the Project II report, including:

- stiffness-matrix construction
- eigenvalue and condition-number analysis
- stiffness-ratio conditioning study
- Jacobi diagonal-scaling test
- gradient-descent convergence study
- objective-error and gradient-norm convergence figures
- Newton-method comparison
- Project II figures saved in the `figures/` directory

The analysis is deterministic and does not use random sampling, so no random seed is required.

### Project II Optimization Model

The total potential energy is written as

$$\Pi(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T K\mathbf{x} - \mathbf{f}^T\mathbf{x}$$

where $K$ is the spring-system stiffness matrix.

The gradient is

$$\nabla\Pi(\mathbf{x}) = K\mathbf{x}-\mathbf{f}$$

and the Hessian is

$$H = K$$

This allows the numerical conditioning of the optimization problem to be studied directly using the eigenvalues of the mechanical stiffness matrix.

The condition number is

$$\kappa(K) = \frac{\lambda_{\max}(K)}{\lambda_{\min}(K)}$$

As the stiffness ratio increases, the condition number also increases, producing a more ill-conditioned optimization problem.

The Project II study demonstrates that increasing stiffness contrast causes standard gradient descent to converge much more slowly. Newton's method is then used as a curvature-aware comparison method for the quadratic system.

### Project II Output Figures

Running the analysis script generates the following Project II figures in the `figures/` directory:

- `project2_condition_number.png`
- `project2_intrinsic_conditioning.png`
- `project2_gradient_descent_convergence.png`
- `project2_gradient_norm_convergence.png`
- `project2_method_comparison.png`

These figures support the conditioning analysis, intrinsic-scaling test, gradient-descent convergence study, and comparison between gradient descent and Newton's method.
