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

## 3. Objective Function

## 4. Constraints

## 5. Problem Classification

## 6. Assumptions and Simplifications

## References
