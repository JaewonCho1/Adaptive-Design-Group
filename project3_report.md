# Project III: Sparse-Data Field Reconstruction Using a Physics-Informed Neural Network

## 1. Researcher Outreach and Interview

### 1.1 Outreach

For this project, a researcher working on physics-based reconstruction of physical fields was contacted to learn about a real computational problem that could potentially benefit from physics-informed machine learning.

Because scheduling did not allow for a live meeting, the interview was conducted asynchronously through email. The researcher provided written responses to questions concerning their current research problem, governing equations, computational methods, limitations, and possible simplified problems suitable for a physics-informed neural network (PINN).

### 1.2 Interview Summary

The researcher's current work focuses on reconstructing **three-dimensional, time-varying refractive-index fields** from a limited number of camera measurements. These refractive-index variations may be produced by physical phenomena such as heat, airflow, or turbulence.

The purpose of the reconstruction is to make otherwise transparent physical phenomena measurable in three dimensions.

The governing physics may involve the **Navier-Stokes equations**, conservation of mass, and energy or advection-diffusion equations. Additional models relate temperature or density variations to refractive index and ultimately to the bending of light observed by cameras.

Current reconstruction methods include:

- differentiable rendering,
- ray tracing,
- gradient-based optimization,
- GPU-based computation.

For large three-dimensional or time-dependent problems, reconstruction can require thousands of optimization iterations.

A major difficulty is that the reconstruction is an **inverse problem with limited measurements**. A complete three-dimensional physical field must be recovered using only a small number of two-dimensional observations. This makes the problem ill-posed because multiple fields may potentially produce similar observations.

Turbulent flow further increases the difficulty of the reconstruction.

The researcher also identified several challenges when physics-informed neural networks or related methods are used:

- sparse measurements can allow multiple possible reconstructed solutions,
- a model may match measured data without being fully physically correct,
- different loss terms can be difficult to balance,
- training can depend strongly on initialization,
- training can require significant computational time.

For a reduced classroom problem, the researcher recommended preserving the **inverse-problem and limited-data characteristics** rather than attempting to reproduce the complete three-dimensional turbulent-flow problem.

A suggested simplified problem was reconstruction of a temperature field governed by a **heat equation or advection-diffusion equation** using sparse measurements and a PINN.

---

## 2. Problem Brief

### 2.1 Real Research Problem

The full research problem is the reconstruction of a time-dependent three-dimensional refractive-index field from sparse camera observations.

Conceptually, the reconstruction can be represented as

$$\text{sparse camera observations}\rightarrow\text{unknown 3D physical field}.$$

Because the available measurements contain significantly less information than the complete field being reconstructed, this is an inverse problem.

Known physical laws can provide additional information that is not directly contained in the camera measurements. Physics-informed approaches therefore combine measurement agreement with governing physical equations.

### 2.2 Governing Physics

Depending on the specific experiment, the complete problem can involve fluid flow, heat transfer, and optical propagation.

Relevant governing equations include:

- conservation of mass,
- Navier-Stokes equations,
- energy conservation,
- advection-diffusion equations.

For a temperature-like scalar field \(T\), a general advection-diffusion equation may be written as

$$\frac{\partial T}{\partial t}+\mathbf{u}\cdot\nabla T=\alpha\nabla^2T.$$

where:

- \(T\) is the transported scalar field,
- \(\mathbf{u}\) is the velocity field,
- \(\alpha\) is the diffusion coefficient.

The full research problem also requires a relationship between density or temperature and refractive index, followed by an optical forward model describing how the refractive-index field affects measured light rays.

### 2.3 Current Computational Approach

The researcher's current computational approach uses differentiable rendering, ray tracing, and gradient-based optimization.

A forward model predicts the measurements that a particular physical-field estimate would produce. The reconstructed field is then adjusted iteratively until the predicted measurements agree with the experimental measurements.

This process may require thousands of optimization iterations for large three-dimensional and time-dependent problems.

Furthermore, reconstruction must often be repeated for different:

- time steps,
- camera arrangements,
- measurements,
- physical parameters,
- experimental conditions.

### 2.4 Primary Computational Challenge

The primary difficulty identified by the researcher is **measurement sparsity and the resulting ill-posed inverse problem**.

The available measurements do not uniquely describe every point in the unknown physical field.

Therefore,

$$\text{fewer measurements}\rightarrow\text{greater reconstruction ambiguity}.$$

A model may potentially reproduce the available measurements while predicting an incorrect field in regions where no measurements exist.

The governing PDE provides additional constraints that can help reduce this ambiguity.

### 2.5 Role of Physics-Informed Machine Learning

A physics-informed neural network represents the unknown physical field using a neural network while simultaneously enforcing measured data and known governing equations.

Instead of learning only from measurements, the model is penalized when its predicted field violates the governing PDE.

Conceptually,

$$\text{sparse measurements}+\text{physical governing equations}\rightarrow\text{full-field reconstruction}.$$

This property makes PINNs potentially useful for the researcher's problem because measurements are difficult to obtain while physical governing equations are already known.

---

## 3. Minimal Problem Formulation

### 3.1 Simplification of the Research Problem

Modeling the complete three-dimensional turbulent refractive-index reconstruction problem would require fluid dynamics, optical ray tracing, camera models, and a large three-dimensional neural representation.

For the class project, the system will therefore be reduced to a **one-dimensional transient heat-transfer inverse problem**.

The simplification is

$$\text{sparse temperature measurements}\rightarrow\text{reconstruction of the complete }T(x,t)\text{ field}.$$

Although the geometry and physics are simpler, the feature identified by the researcher as most important is preserved:

> Reconstructing a complete physical field using limited measurements while enforcing a governing PDE.

### 3.2 Governing Equation

The simplified system is governed by the one-dimensional transient heat equation,

$$\frac{\partial T}{\partial t}=\alpha\frac{\partial^2T}{\partial x^2}.$$

The spatial domain is

$$0\leq x\leq1,$$

and the time domain is

$$0\leq t\leq1.$$

Here:

- \(T(x,t)\) is temperature,
- \(x\) is spatial position,
- \(t\) is time,
- \(\alpha\) is thermal diffusivity.

An initial value of

$$\alpha=0.01$$

will be used for the numerical experiments.

### 3.3 Boundary Conditions

Homogeneous Dirichlet boundary conditions are used:

$$T(0,t)=0,$$

$$T(1,t)=0.$$

These conditions provide a simple domain while allowing the project to focus on the inverse reconstruction problem.

### 3.4 Reference Temperature Field

A known analytical solution will be used to generate synthetic measurements and evaluate reconstruction accuracy.

The initial temperature distribution is chosen as

$$T(x,0)=\sin(\pi x)+0.5\sin(4\pi x).$$

For the heat equation, the resulting analytical solution is

$$T(x,t)=\sin(\pi x)e^{-\alpha\pi^2t}+0.5\sin(4\pi x)e^{-16\alpha\pi^2t}.$$

This solution contains both a low-frequency and a higher-frequency spatial component.

The analytical solution will only be used to generate synthetic measurement data and calculate reconstruction error. The PINN will not be given the complete analytical field during training.

### 3.5 Inverse Reconstruction Problem

The PINN will receive temperature measurements at only a limited number of spatial locations.

It must reconstruct the complete function

$$T_\theta(x,t).$$

The model is therefore solving the inverse problem

$$\text{sparse samples of }T(x,t)\rightarrow T_\theta(x,t).$$

Instead of being given the complete temperature field, the PINN must combine limited measurement information with the governing heat equation.

### 3.6 Difficulty Parameter

The primary difficulty parameter will be **measurement sparsity**.

The number of available spatial measurement locations will be progressively reduced.

An initial test set will use:

| Case | Number of Sensors \(N_s\) | Difficulty |
|---|---:|---|
| 1 | 16 | Low |
| 2 | 8 | Moderate |
| 3 | 4 | High |
| 4 | 2 | Very High |

A sparsity parameter is defined relative to the 16-sensor reference case as

$$S=\frac{16}{N_s}.$$

Therefore,

$$S\in\{1,2,4,8\}.$$

Increasing \(S\) corresponds to increasingly sparse measurements and a more difficult inverse reconstruction problem.

This parameter provides a controlled method for testing how reconstruction performance changes as less measurement information becomes available.

### 3.7 Project Path

The initial project will follow **Path B: Diagnosis**.

The primary research question is:

> How does measurement sparsity affect the ability of a physics-informed neural network to reconstruct a transient temperature field, and what causes reconstruction failure when measurements become insufficient?

The study will determine whether reconstruction error increases because of:

1. insufficient measurement information,
2. neural-network optimization difficulty,
3. imbalance between PDE and measurement losses,
4. inability of the network to recover higher-frequency spatial features.

If a specific failure mechanism can be identified and corrected, the project may be extended to **Path C** by implementing and evaluating a targeted remedy.

---

## 4. Numerical Experiments

### 4.1 P1: Conventional Baseline

A conventional physics-based reconstruction method will be implemented to provide a baseline for comparison with the PINN.

The analytical heat-equation solution will be used as the ground-truth reference field.

The baseline method will reconstruct the temperature field from the same sparse measurements provided to the PINN.

Reconstruction accuracy will be evaluated using relative \(L^2\) error,

$$\text{Relative }L^2\text{ Error}=\frac{\|T_{\text{pred}}-T_{\text{true}}\|_2}{\|T_{\text{true}}\|_2}.$$

Computational runtime will also be recorded.

### 4.2 P2: PINN Loss Components

The PINN will be trained using a combination of measurement data and physical constraints.

A general loss function will be

$$L=\lambda_{\text{PDE}}L_{\text{PDE}}+\lambda_{\text{BC}}L_{\text{BC}}+\lambda_{\text{IC}}L_{\text{IC}}+\lambda_{\text{data}}L_{\text{data}}.$$

The PDE residual loss will enforce the heat equation,

$$L_{\text{PDE}}=\frac{1}{N_f}\sum_{i=1}^{N_f}\left|\frac{\partial T_\theta}{\partial t}-\alpha\frac{\partial^2T_\theta}{\partial x^2}\right|^2.$$

The boundary-condition loss will enforce

$$T_\theta(0,t)=0$$

and

$$T_\theta(1,t)=0.$$

The measurement loss will compare the model prediction against the sparse measurements,

$$L_{\text{data}}=\frac{1}{N_m}\sum_{i=1}^{N_m}\left|T_\theta(x_i,t_i)-T_i\right|^2.$$

Each loss component will be recorded separately throughout training to determine whether one component dominates the optimization.

### 4.3 P3: Reconstruction Error vs. Measurement Sparsity

The PINN will be trained separately for

$$N_s\in\{16,8,4,2\}.$$

The corresponding sparsity values are

$$S\in\{1,2,4,8\}.$$

Relative \(L^2\) reconstruction error will be calculated for each case.

The resulting plot will show how reconstruction accuracy changes as the measurement information becomes increasingly sparse.

### 4.4 P4: Spatial and Spectral Error Analysis

Pointwise reconstruction error will be calculated as

$$e(x,t)=|T_\theta(x,t)-T_{\text{true}}(x,t)|.$$

A heatmap of this error will be used to determine where the PINN reconstruction fails.

Because the analytical temperature field contains both low- and high-frequency spatial components, Fourier analysis will also be used to evaluate whether the PINN preferentially reconstructs lower-frequency features.

This analysis may reveal whether reconstruction difficulty results from spectral bias or insufficient measurement information.

### 4.5 P5: Computational Cost

Computational cost will be evaluated using:

- PINN training time,
- PINN inference time,
- conventional baseline reconstruction time,
- hardware used for the experiments.

This comparison will help determine whether a PINN could provide useful faster inference after the initial training cost.

---

## 5. Results and Discussion

This section will contain the numerical results after the conventional baseline and PINN experiments are completed.

Results will include:

- reference temperature field,
- sparse measurement locations,
- training-loss history,
- relative reconstruction error,
- error versus sensor sparsity,
- pointwise error fields,
- Fourier-spectrum comparison,
- computational runtime.

The results will be used to determine whether PINN failure at high sparsity is primarily caused by insufficient measurement information or by neural-network optimization behavior.

---

## 6. Recommendation to Researcher

A final recommendation will be prepared after completion of the numerical experiments.

The recommendation will address whether physics-informed neural networks appear useful for sparse field reconstruction and under what measurement conditions they remain reliable.

Particular attention will be given to the researcher's stated goals of reducing measurement requirements and potentially enabling faster inference.

---

## 7. Assumptions and Limitations

The class project intentionally simplifies the researcher's full physical reconstruction problem.

Major simplifications include:

- reducing the problem from three spatial dimensions to one,
- using the heat equation rather than the full Navier-Stokes equations,
- using synthetic measurements rather than experimental camera data,
- excluding optical ray tracing,
- excluding turbulence,
- assuming a known constant thermal diffusivity,
- using idealized boundary conditions,
- initially assuming noise-free measurements.

These simplifications allow the study to isolate the central research challenge identified by the researcher: reconstructing a complete physical field from limited measurements while enforcing known physics.

The results should therefore be interpreted as a controlled demonstration of the inverse-problem behavior rather than a direct simulation of the researcher's full experimental system.
