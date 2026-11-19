"""
Project 4 -- Heavy-lift quadcopter airframe: specification, geometry, and the
design problem (objective, constraints and their adjoint gradients).

The airframe continues Project 1 (quadcopter arm sizing), now as a full 3D
space frame of aluminium tubes:

  * a central hub (an octagonal bottom ring, a square top ring and X-braced
    side panels) that carries the battery/payload at four hard points;
  * four identical triangular lattice arms (two bottom chords, one top chord,
    4 bays) running from the hub to the motor mounts;
  * a three-strut motor mount at each arm tip.

Design variables (35):
  * 31 member-group radii: 28 groups per arm (left/right mirror pairs share a
    size, and the four arms are identical because one arm design is built four
    times), plus 3 hub groups;
  * 4 shape parameters: arm depth and width at the root and at the tip.

Every member is split into two beam elements (requirement of the notes).
"""

import numpy as np

from project4_frame import (Material, FrameModel, Analysis, chs_section,
                            radius_of_gyration, response_mass, response_stress,
                            response_displacement, response_frequency,
                            response_buckling, CS_H)

G0 = 9.81

# ============================================================
# Specification
# ============================================================

SPEC = {
    # geometry (m)
    'R_hub': 0.15,          # hub radius to the arm root
    'L_arm': 0.50,          # root-to-tip length of each arm (motor radius 0.65 m)
    'n_bay': 4,
    'z_motor': 0.04,        # motor-mount height above the tip top node
    'prop_diameter': 0.762,  # 30-inch propellers
    # material: aluminium 6061-T6 (ASM / MatWeb data sheet)
    'E': 68.9e9, 'nu': 0.33, 'rho': 2700.0,
    'S_y': 276e6, 'S_u': 310e6, 'S_f': 96.5e6,   # S_f: fatigue strength at 5e8 cycles
    'k_f': 0.5,             # fatigue strength factor for welded / bonded tube joints
    'tw': 0.10,             # wall thickness / outer radius
    # mass budget and loads
    'MTOW': 25.0,           # maximum take-off mass (kg)
    'T_max': 120.0,         # maximum thrust per motor (N)
    'Q_max': 4.0,           # rotor reaction torque per motor at T_max (N m)
    'm_motor': 0.90,        # motor + propeller + ESC at each tip (kg)
    'rpm_max': 3200.0,      # rotor speed at T_max
    # requirements
    'SF_static': 1.5,       # R1: S_y / SF
    'n_fatigue': 2.0,       # R2: Goodman factor of safety
    'delta_max': 2.0e-3,    # R3: motor vertical deflection (m)
    'freq_margin': 1.2,     # R4: f_min >= margin * rotor 1P frequency
    'lam_req': 3.0,         # R5: linear buckling load factor
    'slender_max': 200.0,   # R6: L / r_gyr
    # bounds
    'r_min': 4.0e-3, 'r_max': 20.0e-3,
    # KS aggregation parameters
    'rho_stress': 60.0, 'rho_disp': 200.0, 'rho_freq': 60.0, 'rho_buck': 60.0,
}

SPEC['n_load'] = 4 * SPEC['T_max'] / (SPEC['MTOW'] * G0)          # 1.96 g climb
SPEC['f_1P'] = SPEC['rpm_max'] / 60.0
SPEC['f_req'] = SPEC['freq_margin'] * SPEC['f_1P']
SPEC['S_static'] = SPEC['S_y'] / SPEC['SF_static']
# zero-based cycling: sigma_a = sigma_m = sigma/2;  1/n = sigma/2 (1/S_f + 1/S_u)
SPEC['S_e'] = SPEC['k_f'] * SPEC['S_f']
SPEC['S_fatigue'] = 2.0 / (SPEC['n_fatigue'] * (1 / SPEC['S_e'] + 1 / SPEC['S_u']))

SHAPE_NAMES = ['h_root', 'h_tip', 'w_root', 'w_tip']
SHAPE_LB = np.array([0.04, 0.03, 0.04, 0.03])
SHAPE_UB = np.array([0.20, 0.12, 0.20, 0.12])
SHAPE0 = np.array([0.10, 0.06, 0.10, 0.06])
R0 = 8.0e-3

MAT = Material(SPEC['E'], SPEC['nu'], SPEC['rho'])


# ============================================================
# Topology
# ============================================================

def _kidx(k, j, c):
    """Key node index: arm k, station j, c = 0 (BL), 1 (BR), 2 (T)."""
    return 16 * k + 3 * j + c


def _midx(k):
    return 16 * k + 15


def build_topology():
    nb = SPEC['n_bay']
    groups = []          # group names
    gid = {}

    def g(name):
        if name not in gid:
            gid[name] = len(groups)
            groups.append(name)
        return gid[name]

    members = []         # (node_a, node_b, group, kind)
    for k in range(4):
        for j in range(nb):
            members.append((_kidx(k, j, 0), _kidx(k, j + 1, 0), g(f'bottom chord {j+1}'), 'arm'))
            members.append((_kidx(k, j, 1), _kidx(k, j + 1, 1), g(f'bottom chord {j+1}'), 'arm'))
        for j in range(nb):
            members.append((_kidx(k, j, 2), _kidx(k, j + 1, 2), g(f'top chord {j+1}'), 'arm'))
        for j in range(nb + 1):
            members.append((_kidx(k, j, 0), _kidx(k, j, 1), g(f'frame bottom {j}'), 'arm'))
        for j in range(nb + 1):
            members.append((_kidx(k, j, 0), _kidx(k, j, 2), g(f'frame side {j}'), 'arm'))
            members.append((_kidx(k, j, 1), _kidx(k, j, 2), g(f'frame side {j}'), 'arm'))
        for j in range(nb):
            members.append((_kidx(k, j, 0), _kidx(k, j + 1, 2), g(f'side diagonal {j+1}'), 'arm'))
            members.append((_kidx(k, j, 1), _kidx(k, j + 1, 2), g(f'side diagonal {j+1}'), 'arm'))
        for j in range(nb):
            members.append((_kidx(k, j, 0), _kidx(k, j + 1, 1), g(f'bottom diagonal {j+1}'), 'arm'))
        members.append((_kidx(k, nb, 0), _midx(k), g('motor strut bottom'), 'arm'))
        members.append((_kidx(k, nb, 1), _midx(k), g('motor strut bottom'), 'arm'))
        members.append((_kidx(k, nb, 2), _midx(k), g('motor strut top'), 'arm'))
    for k in range(4):
        k1 = (k + 1) % 4
        members.append((_kidx(k, 0, 0), _kidx(k1, 0, 1), g('hub bottom ring'), 'hub'))
        members.append((_kidx(k, 0, 2), _kidx(k1, 0, 2), g('hub top ring'), 'hub'))
        members.append((_kidx(k, 0, 2), _kidx(k1, 0, 1), g('hub panel brace'), 'hub'))
        members.append((_kidx(k1, 0, 2), _kidx(k, 0, 0), g('hub panel brace'), 'hub'))
    return members, groups


MEMBERS, GROUPS = build_topology()
N_KEY = 64
N_MEM = len(MEMBERS)
N_GROUP = len(GROUPS)
N_SHAPE = 4
N_VAR = N_GROUP + N_SHAPE
MEM_A = np.array([m[0] for m in MEMBERS])
MEM_B = np.array([m[1] for m in MEMBERS])
MEM_GROUP = np.array([m[2] for m in MEMBERS])
MEM_KIND = [m[3] for m in MEMBERS]
N_NODE = N_KEY + N_MEM            # one mid-node per member

# elements: member m -> (a, mid), (mid, b)
CONN = np.zeros((2 * N_MEM, 2), int)
CONN[0::2, 0] = MEM_A
CONN[0::2, 1] = N_KEY + np.arange(N_MEM)
CONN[1::2, 0] = N_KEY + np.arange(N_MEM)
CONN[1::2, 1] = MEM_B
ELEM_MEMBER = np.repeat(np.arange(N_MEM), 2)
ELEM_GROUP = MEM_GROUP[ELEM_MEMBER]
N_ELEM = CONN.shape[0]

MOTOR_NODES = np.array([_midx(k) for k in range(4)])
HUB_BOTTOM_MEMBERS = [i for i, m in enumerate(MEMBERS) if GROUPS[m[2]] == 'hub bottom ring']
SUPPORT_NODES = np.array([N_KEY + i for i in HUB_BOTTOM_MEMBERS])
YAW_SIGN = np.array([1.0, -1.0, 1.0, -1.0])     # alternating CW/CCW rotors


# ============================================================
# Geometry  (complex-safe in the shape vector)
# ============================================================

def geometry(shape):
    """Node coordinates (N_NODE,3) for shape = [h_root, h_tip, w_root, w_tip]."""
    hr, ht, wr, wt = shape[0], shape[1], shape[2], shape[3]
    nb = SPEC['n_bay']
    dtype = np.result_type(shape, float)
    X = np.zeros((N_NODE, 3), dtype=dtype)
    for k in range(4):
        phi = np.pi / 4 + k * np.pi / 2
        d = np.array([np.cos(phi), np.sin(phi), 0.0])
        nvec = np.array([-np.sin(phi), np.cos(phi), 0.0])
        for j in range(nb + 1):
            t = j / nb
            rad = SPEC['R_hub'] + t * SPEC['L_arm']
            h = hr + (ht - hr) * t
            w = wr + (wt - wr) * t
            X[_kidx(k, j, 0)] = rad * d + 0.5 * w * nvec
            X[_kidx(k, j, 1)] = rad * d - 0.5 * w * nvec
            X[_kidx(k, j, 2)] = rad * d + np.array([0, 0, 1.0]) * h
        X[_midx(k)] = (SPEC['R_hub'] + SPEC['L_arm']) * d + np.array([0, 0, 1.0]) * (ht + SPEC['z_motor'])
    X[N_KEY:] = 0.5 * (X[MEM_A] + X[MEM_B])
    return X


def geometry_jacobian(shape):
    """dX/dshape (N_NODE,3,4) by complex step."""
    Jx = np.zeros((N_NODE, 3, N_SHAPE))
    for i in range(N_SHAPE):
        s = shape.astype(complex)
        s[i] += 1j * CS_H
        Jx[:, :, i] = np.imag(geometry(s)) / CS_H
    return Jx


# ============================================================
# Design-variable scaling:  x in [0,1]^N_VAR
#   radii: r = r_min * (r_max/r_min)^x  (log scale, sizes span 4x)
#   shape: linear between bounds
# ============================================================

def unscale(x):
    x = np.asarray(x, float)
    lr = np.log(SPEC['r_max'] / SPEC['r_min'])
    rg = SPEC['r_min'] * np.exp(lr * x[:N_GROUP])
    sh = SHAPE_LB + (SHAPE_UB - SHAPE_LB) * x[N_GROUP:]
    return rg, sh


def scale(rg, sh):
    lr = np.log(SPEC['r_max'] / SPEC['r_min'])
    return np.concatenate([np.log(np.asarray(rg) / SPEC['r_min']) / lr,
                           (np.asarray(sh) - SHAPE_LB) / (SHAPE_UB - SHAPE_LB)])


def dphys_dx(x):
    rg, sh = unscale(x)
    lr = np.log(SPEC['r_max'] / SPEC['r_min'])
    return rg * lr, (SHAPE_UB - SHAPE_LB)


def x_initial():
    return scale(np.full(N_GROUP, R0), SHAPE0)


# ============================================================
# Model assembly for one design
# ============================================================

def build_model(rg, sh, supports='mounts'):
    X = geometry(sh)
    r = np.asarray(rg)[ELEM_GROUP]
    fixed = []
    if supports == 'mounts':
        for nd in SUPPORT_NODES:
            fixed += [6 * nd + c for c in range(6)]
    pm = np.zeros(N_NODE)
    pm[MOTOR_NODES] = SPEC['m_motor']
    f = np.zeros(6 * N_NODE)
    nl = SPEC['n_load']
    for k, nd in enumerate(MOTOR_NODES):
        f[6 * nd + 2] += SPEC['T_max'] - nl * G0 * SPEC['m_motor']
        f[6 * nd + 5] += YAW_SIGN[k] * SPEC['Q_max']
    return FrameModel(X, CONN, r, MAT, fixed, tw=SPEC['tw'], point_mass=pm,
                      f_ext=f, accel=(0.0, 0.0, -nl * G0))


def motor_dofs():
    return 6 * MOTOR_NODES + 2


def member_lengths(sh):
    X = geometry(sh)
    return np.sqrt(np.sum((X[MEM_B] - X[MEM_A])**2, axis=1))


def slenderness(rg, sh):
    """L / r_gyr for every member (explicit formula)."""
    L = member_lengths(sh)
    return L / radius_of_gyration(np.asarray(rg)[MEM_GROUP], SPEC['tw'])


def slenderness_jac_x(x):
    """d(slenderness/limit)/dx  (N_MEM, N_VAR) by complex step on the explicit formula."""
    Jm = np.zeros((N_MEM, N_VAR))
    for i in range(N_VAR):
        xc = np.asarray(x, complex)
        xc[i] += 1j * CS_H
        lr = np.log(SPEC['r_max'] / SPEC['r_min'])
        rg = SPEC['r_min'] * np.exp(lr * xc[:N_GROUP])
        sh = SHAPE_LB + (SHAPE_UB - SHAPE_LB) * xc[N_GROUP:]
        Jm[:, i] = np.imag(slenderness(rg, sh)) / CS_H / SPEC['slender_max']
    return Jm


# ============================================================
# Evaluate all requirements (and gradients w.r.t. scaled x)
# ============================================================

def _chain(model, sh, grad, x):
    """(dX, dr_elem) -> gradient w.r.t. scaled design vector x."""
    dX, dr = grad
    dRg = np.bincount(ELEM_GROUP, weights=dr, minlength=N_GROUP)
    dSh = np.einsum('nc,nci->i', dX, geometry_jacobian(sh))
    sr, ss = dphys_dx(x)
    return np.concatenate([dRg * sr, dSh * ss])


class Evaluation:
    """All responses of one design. Constraint values g are written as g <= 1."""

    def __init__(self, x, grads=True, buckling=True, nm=10, nb=8):
        self.x = np.asarray(x, float)
        self.rg, self.sh = unscale(self.x)
        self.model = build_model(self.rg, self.sh)
        self.an = Analysis(self.model)
        m = self.model
        S = SPEC
        mass, gm = response_mass(m)
        self.mass_frame = mass
        self.mass_motors = 4 * S['m_motor']
        st = response_stress(self.an, S['S_fatigue'], S['rho_stress'], grads)
        dp = response_displacement(self.an, motor_dofs(), S['delta_max'], S['rho_disp'], grads)
        fq = response_frequency(self.an, S['f_req'], S['rho_freq'], nm, grads)
        self.vm = st['vm']
        self.sigma_max = st['vm'].max()
        self.delta = dp['u']
        self.freqs = fq['f']
        self.g = {'R2': st['value'], 'R3': dp['value'], 'R4': fq['value']}
        self.gtrue = {'R2': st['max'], 'R3': dp['max'], 'R4': fq['max']}
        if buckling:
            bk = response_buckling(self.an, S['lam_req'], S['rho_buck'], nb, grads)
            self.lams = bk['lam']
            self.g['R5'] = bk['value']
            self.gtrue['R5'] = bk['max']
        self.slender = slenderness(self.rg, self.sh)
        if grads:
            self.dmass = _chain(m, self.sh, gm, self.x)
            self.dg = {'R2': _chain(m, self.sh, st['grad'], self.x),
                       'R3': _chain(m, self.sh, dp['grad'], self.x),
                       'R4': _chain(m, self.sh, fq['grad'], self.x)}
            if buckling:
                self.dg['R5'] = _chain(m, self.sh, bk['grad'], self.x)
            self.dslender = slenderness_jac_x(self.x)

    # ---- human-readable requirement table ----
    def metrics(self):
        S = SPEC
        smax = self.sigma_max
        nf = 1.0 / (0.5 * smax * (1 / S['S_e'] + 1 / S['S_u']))
        out = [
            dict(id='OBJ', name='Frame mass (tubes)', value=self.mass_frame, unit='kg', limit=None, ok=None),
            dict(id='R1', name='Static strength: max von Mises', value=smax / 1e6, unit='MPa',
                 limit=f"<= {S['S_static']/1e6:.0f}", ok=bool(smax <= S['S_static'])),
            dict(id='R2', name='Fatigue: Goodman factor of safety', value=nf, unit='',
                 limit=f">= {S['n_fatigue']:.1f}", ok=bool(nf >= S['n_fatigue'] * (1 - 1e-4))),
            dict(id='R3', name='Max motor deflection', value=np.abs(self.delta).max() * 1e3, unit='mm',
                 limit=f"<= {S['delta_max']*1e3:.1f}", ok=bool(np.abs(self.delta).max() <= S['delta_max'] * (1 + 1e-4))),
            dict(id='R4', name='Lowest natural frequency', value=self.freqs.min(), unit='Hz',
                 limit=f">= {S['f_req']:.1f}", ok=bool(self.freqs.min() >= S['f_req'] * (1 - 1e-4))),
        ]
        if hasattr(self, 'lams'):
            out.append(dict(id='R5', name='Buckling load factor', value=self.lams.min(), unit='',
                            limit=f">= {S['lam_req']:.1f}", ok=bool(self.lams.min() >= S['lam_req'] * (1 - 1e-4))))
        out.append(dict(id='R6', name='Max member slenderness L/r', value=self.slender.max(), unit='',
                        limit=f"<= {S['slender_max']:.0f}", ok=bool(self.slender.max() <= S['slender_max'])))
        return out
