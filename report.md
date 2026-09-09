# Lightweight Quadcopter Frame Design Optimization

## 1. Problem Identification and Motivation
Quadcopters are widely used for applications such as aerial photography,
inspection, mapping, and payload transportation. One important consideration
in quadcopter design is the mass of the structural frame. A lighter frame
reduces the amount of structural mass that must be supported by the propulsion
system and can allow more of the vehicle's mass budget to be allocated to
components such as the battery or payload.

Reducing frame mass, however, creates a structural tradeoff. Removing material
from the frame can reduce its mass, but it can also increase stress and
deformation in the arms that support the motors. The frame must therefore be
lightweight while remaining sufficiently strong and stiff under the expected
motor loading.

In this project, we consider a symmetric X-configuration quadcopter with four
identical arms connecting a central body to four motors. The motor and
propeller system is assumed to be predetermined. The design problem is to
determine the geometry of the quadcopter arms that minimizes structural mass
while satisfying requirements for strength, stiffness, propeller clearance,
and practical geometric dimensions.

**Design question:** What combination of quadcopter arm dimensions minimizes
structural mass while satisfying the required strength, stiffness, propeller
clearance, and geometric constraints?

## 2. Decision Variables

The quadcopter frame is modeled using four identical hollow rectangular
arms. Four continuous decision variables are used to define the geometry
of each arm.

| Variable | Description | Units | Lower Bound | Upper Bound | Type |
|---|---|---|---:|---:|---|
| $L$ | Structural arm length from body to motor | mm | 140 | 220 | Continuous |
| $b$ | Outer width of the arm cross-section | mm | 10 | 30 | Continuous |
| $h$ | Outer height of the arm cross-section | mm | 10 | 40 | Continuous |
| $t$ | Wall thickness of the hollow arm | mm | 0.5 | 3.0 | Continuous |

The design vector is

$$
\mathbf{x} = [L,\ b,\ h,\ t]^T.
$$

Arm length $L$ affects the overall frame geometry, structural mass, and
bending loads. The dimensions $b$, $h$, and $t$ determine the amount and
distribution of material in each arm, which affects its mass, strength,
and stiffness.

The variables are restricted to the bounds shown above to maintain a
realistic design space for the selected quadcopter configuration.
Additional geometric constraints will be imposed to ensure that the hollow
cross-section remains physically valid and that adequate propeller
clearance is maintained.

### 2.1 Fixed Parameters

In addition to the decision variables, the optimization model contains
several fixed parameters describing the quadcopter configuration, loading,
material properties, and design requirements. Unlike the decision variables,
these quantities are not selected by the optimizer.

| Symbol | Description | Value | Units | Basis |
|---|---|---:|---|---|
| $B$ | Central body width | 100 | mm | Modeling assumption |
| $D_p$ | Propeller diameter | 241.3 | mm | Reference 9.5 in propeller |
| $F_{\mathrm{motor}}$ | Maximum thrust of one motor | TBD | N | Manufacturer data |
| $n$ | Design load factor | TBD | -- | Design assumption |
| $F_d$ | Structural design load | Calculated | N | $nF_{\mathrm{motor}}$ |
| $\rho$ | Arm material density | TBD | kg/m³ | Material data |
| $E$ | Effective Young's modulus | TBD | GPa | Material data |
| $\sigma_{\mathrm{allow}}$ | Allowable bending stress | TBD | MPa | Material/design requirement |
| $C_{\min}$ | Minimum propeller tip clearance | TBD | mm | Design requirement |
| $\delta_{\mathrm{allow}}$ | Maximum arm-tip deflection | TBD | mm | Design requirement |

Values marked **TBD** will be established using either published
manufacturer/material data or explicitly stated engineering assumptions.
This distinction is maintained so that sourced physical properties are not
confused with requirements selected as part of the modeling process.

## 3. Objective Function

The objective of this optimization problem is to minimize the structural
mass of the four quadcopter arms. Because all four arms are assumed to
have identical geometry and material, the total structural arm mass can
be calculated from the mass of a single arm.

For a hollow rectangular arm, the cross-sectional material area is

$$
A(b,h,t)=bh-(b-2t)(h-2t),
$$

where $bh$ represents the area of the outer rectangle and
$(b-2t)(h-2t)$ represents the hollow interior.

The volume of one arm is $AL$, and its mass is $\rho AL$, where $\rho$
is the material density. Therefore, the total mass of the four arms is

```math
m_{\mathrm{arms}}(L,b,h,t)
=
4\rho L\left[bh-(b-2t)(h-2t)\right]
```

The optimization objective is therefore

```math
\min_{L,b,h,t} \quad m_{\mathrm{arms}}(L,b,h,t)
```

Minimizing this objective reduces the structural mass of the quadcopter
frame while the constraints defined in the following section ensure that
the resulting design maintains the required structural and geometric
performance.

## 4. Constraints

### 4.1 Geometric Validity

Each arm is modeled as a hollow rectangular tube. The dimensions of the
hollow interior are

$$
b_i=b-2t
$$

and

$$
h_i=h-2t.
$$

For the hollow cross-section to remain physically valid, both internal
dimensions must remain positive. Therefore,

$$
2t<b
$$

and

$$
2t<h.
$$

These constraints prevent the optimizer from selecting a wall thickness
that is incompatible with the selected outer width or height.

### 4.2 Propeller Clearance

Adjacent propellers must have sufficient separation to prevent their swept
disks from overlapping. The central body is modeled as a square with fixed
width $B$, and each arm extends diagonally outward from a corner of the body.

The distance from the center of the vehicle to a body corner is

$$
r_b=\frac{B}{\sqrt{2}}.
$$

Because the structural arm has length $L$, the center-to-motor distance is

$$
r=L+\frac{B}{\sqrt{2}}.
$$

Adjacent motors are separated by an angle of $90^\circ$, giving an
adjacent motor spacing of

$$
d_{\mathrm{motor}}=\sqrt{2}r
=B+\sqrt{2}L.
$$

To prevent adjacent propeller disks from overlapping, the motor spacing
must be at least the propeller diameter $D_p$ plus a specified minimum
clearance $C_{\min}$. Therefore, the propeller-clearance constraint is

$$
B+\sqrt{2}L \geq D_p+C_{\min}.
$$

Equivalently, the minimum arm length required by propeller clearance is

$$
L \geq
\frac{D_p+C_{\min}-B}{\sqrt{2}}.
$$

For the reference configuration, a $100$ mm square central body and a
$241.3$ mm (9.5 in) propeller diameter are assumed. A minimum propeller
tip clearance will be specified as a fixed design parameter.

### 4.3 Structural Stress

Each quadcopter arm is approximated as a cantilever beam fixed at the
central body and subjected to a vertical design load $F_d$ at its free
end. The design load is defined as

$$
F_d=nF_{\mathrm{motor}}
$$

where $F_{\mathrm{motor}}$ is the reference maximum motor thrust and $n$
is a design load factor used to provide margin for loading not represented
by the simplified static beam model.

Under this loading condition, the maximum bending moment occurs at the
root of the arm and is

$$
M_{\max}=F_dL
$$

The maximum bending stress is calculated using the beam-bending relation

$$
\sigma_{\max}=\frac{M_{\max}c}{I}
$$

where $c=h/2$ is the distance from the neutral axis to the outer surface
of the arm and $I$ is the second moment of area of the cross-section.

For the hollow rectangular cross-section,

```math
I(b,h,t)
=
\frac{bh^3-(b-2t)(h-2t)^3}{12}
```

Therefore, the maximum bending stress can be expressed directly in terms
of the design variables as

```math
\sigma_{\max}(L,b,h,t)
=
\frac{F_dL(h/2)}{I(b,h,t)}
```

The structural strength constraint is

```math
\sigma_{\max}(L,b,h,t)
\leq
\sigma_{\mathrm{allow}}
```

where $\sigma_{\mathrm{allow}}$ is the allowable bending stress of the
selected arm material. This constraint prevents the optimizer from
reducing the arm dimensions to values that would produce unacceptable
bending stress under the specified motor loading.

### 4.4 Structural Deflection

Satisfying the stress constraint does not necessarily guarantee that an
arm is sufficiently stiff. Excessive deformation could alter the position
and orientation of the motor even if the material remains below its
allowable stress. Therefore, a separate deflection constraint is imposed.

Using the same cantilever-beam approximation and design load $F_d$ defined
for the stress constraint, the maximum vertical deflection occurs at the
motor end of the arm and is

```math
\delta_{\max}
=
\frac{F_dL^3}{3EI}
```

where $E$ is the Young's modulus of the selected arm material and $I$ is
the second moment of area of the hollow rectangular cross-section.

Using the previously defined expression $I(b,h,t)$, the tip deflection can
be written directly in terms of the design variables as

```math
\delta_{\max}(L,b,h,t)
=
\frac{F_dL^3}{3E I(b,h,t)}
```

The stiffness constraint is therefore

```math
\delta_{\max}(L,b,h,t)
\leq
\delta_{\mathrm{allow}}
```

where $\delta_{\mathrm{allow}}$ is the maximum allowable motor-end
deflection. This constraint prevents the optimizer from selecting a
minimum-mass arm geometry that satisfies the strength requirement but is
too flexible for the intended structural application.

where $\delta_{\mathrm{allow}}$ is the maximum allowable motor-end
deflection. This constraint prevents the optimizer from selecting a
minimum-mass arm geometry that satisfies the strength requirement but is
too flexible for the intended structural application.

## 5. Problem Classification

The quadcopter arm design problem is formulated as a **constrained
continuous nonlinear optimization problem (NLP)**.

All four decision variables, $\mathbf{x}=[L,b,h,t]^T$, are continuous
geometric quantities. The formulation does not contain integer or binary
decision variables, so it is not a mixed-integer optimization problem.

The problem is nonlinear because the objective function contains products
of the decision variables. The arm-mass objective is

```math
m_{\mathrm{arms}}(L,b,h,t)
=
4\rho L\left[bh-(b-2t)(h-2t)\right].
```

For example, terms involving products of $L$, $b$, $h$, and $t$ prevent
the objective from being a linear function of the decision variables.

The structural constraints also introduce nonlinearity. The second moment
of area of the hollow rectangular cross-section is

```math
I(b,h,t)
=
\frac{bh^3-(b-2t)(h-2t)^3}{12}.
```

This expression contains powers and products of the geometric decision
variables. Consequently, the bending-stress constraint contains the
nonlinear expression

```math
\sigma_{\max}(L,b,h,t)
=
\frac{F_dL(h/2)}{I(b,h,t)},
```

and the deflection constraint contains

```math
\delta_{\max}(L,b,h,t)
=
\frac{F_dL^3}{3E I(b,h,t)}.
```

The formulation is constrained because feasible designs must satisfy
variable bounds, hollow-section geometric requirements, propeller
clearance, maximum bending stress, and maximum tip deflection. These are
inequality constraints; the current formulation does not contain any
equality constraints.

Therefore, the overall problem is classified as a **constrained continuous
nonlinear program (NLP)**.

Although the formulation contains nonlinear expressions, nonlinearity
alone does not prove that an optimization problem is nonconvex. Convexity
has not been established for the current formulation over the selected
design domain, so the problem is not assumed to be convex without further
analysis.
## 6. Assumptions and Simplifications

The optimization model uses several assumptions to keep the formulation
tractable while retaining the primary structural tradeoffs relevant to the
quadcopter arm design.

1. **Symmetric frame:** The quadcopter is assumed to have four identical
   arms. All four arms therefore share the same values of $L$, $b$, $h$,
   and $t$. This reduces the number of independent geometric variables and
   is consistent with the symmetric X-configuration considered in this
   project.

2. **Cantilever-beam model:** Each arm is modeled as a cantilever beam with
   a fixed connection at the central body and a load applied at the motor
   end. The central body is assumed to be sufficiently rigid for this
   approximation. This allows standard beam-bending equations to be used
   for estimating stress and deflection.

3. **Equivalent static loading:** Motor loading is represented using the
   design load $F_d=nF_{\mathrm{motor}}$, where $F_{\mathrm{motor}}$ is the
   reference maximum motor thrust and $n$ is a design load factor. This
   provides a simplified margin for loading effects that are not explicitly
   represented by the static beam model.

4. **Effective material properties:** The arm material is modeled using a
   fixed Young's modulus $E$, density $\rho$, and allowable stress
   $\sigma_{\mathrm{allow}}$. Carbon-fiber composites are anisotropic in
   reality, so these values represent effective properties for the primary
   structural loading direction rather than a complete composite-material
   model.

5. **Constant cross-section:** Each arm is assumed to have a uniform hollow
   rectangular cross-section along its entire length. Therefore, the
   dimensions $b$, $h$, and $t$ do not vary along the arm. Tapered or
   spatially varying arm geometries are not considered.

6. **Fixed central body:** The central body is modeled as a square with a
   fixed width of $B=100$ mm. The dimensions and mass of the central body
   are not design variables and are not included in the arm-mass objective.
   The body geometry is used when determining motor spacing and propeller
   clearance.

7. **Fixed propulsion configuration:** The motor and propeller system is
   assumed to be predetermined. Motor thrust $F_{\mathrm{motor}}$ and
   propeller diameter $D_p$ are therefore treated as fixed parameters
   rather than decision variables. The optimization focuses only on the
   structural geometry of the quadcopter arms.

8. **Limited structural failure modes:** The structural model considers
   bending stress and vertical tip deflection. Other effects such as
   torsion, local buckling, fatigue, vibration, resonance, connection
   failure, composite delamination, and impact loading are not explicitly
   modeled. These effects could be considered in a higher-fidelity
   structural analysis.

These assumptions allow the optimization problem to focus on the
relationship between arm geometry, structural mass, strength, stiffness,
and propeller clearance while keeping the formulation manageable. A
higher-fidelity design study could relax these assumptions by including
dynamic loading, detailed composite-material behavior, additional failure
modes, and optimization of other quadcopter components.
   
## References
