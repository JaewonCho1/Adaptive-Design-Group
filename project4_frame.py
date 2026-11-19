"""
Project 4 -- 3D frame finite element code with adjoint design sensitivities.

Everything here is generic (no drone-specific data). project4_drone.py builds a
model and calls these functions.

Element: 2-node, 12-DOF Euler-Bernoulli space-frame element (cubic Hermite
bending, linear axial and torsion), circular hollow section (CHS) with wall
thickness t = TW * r (one size variable per member).

Matrices per element (global frame):
    k_e  linear stiffness
    m_e  consistent mass
    kg_e geometric stiffness for a unit axial force (multiply by N_e)
    f_e  consistent nodal loads of a uniform body force (self-weight)

Sensitivities. Every response gradient is computed by the adjoint method with
the global derivative matrices never formed. The element-level partial
derivatives d(.)/d(x_i, x_j, r) are obtained by the complex-step method
(Squire & Trapp 1998), which is exact to machine precision for these real-
analytic element functions. That is implementation (ii) of the course notes,
with complex step in place of an AD framework: no dependency beyond numpy/scipy.

Units: SI (m, N, kg, Pa).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

CS_H = 1e-30  # complex-step size


# ============================================================
# Section and element geometry (complex-safe)
# ============================================================

def chs_section(r, tw):
    """Area, second moment, polar moment of a CHS with outer radius r, wall tw*r."""
    ro = r
    ri = r * (1.0 - tw)
    A = np.pi * (ro**2 - ri**2)
    I = 0.25 * np.pi * (ro**4 - ri**4)
    J = 2.0 * I
    return A, I, J


def radius_of_gyration(r, tw):
    ri = r * (1.0 - tw)
    return 0.5 * np.sqrt(r**2 + ri**2)


def element_frames(Xi, Xj):
    """Length and rotation matrices R (ne,3,3); rows are local axes (x along member).

    For a CHS (Iy = Iz) the stiffness is invariant to the choice of the local
    y/z axes, so the switch of reference vector for near-vertical members does
    not create a discontinuity in any response.
    """
    d = Xj - Xi
    L = np.sqrt(np.sum(d * d, axis=1))
    e1 = d / L[:, None]
    vert = np.abs(np.real(e1[:, 2])) > 0.95
    ref = np.zeros(e1.shape, dtype=e1.dtype)
    ref[:, 2] = 1.0
    ref[vert, 2] = 0.0
    ref[vert, 0] = 1.0
    e2 = np.cross(ref, e1)
    e2 = e2 / np.sqrt(np.sum(e2 * e2, axis=1))[:, None]
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=1)
    return L, R


def transformation(R):
    ne = R.shape[0]
    T = np.zeros((ne, 12, 12), dtype=R.dtype)
    for b in range(4):
        T[:, 3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
    return T


# Local DOF order: [u1 v1 w1 tx1 ty1 tz1 u2 v2 w2 tx2 ty2 tz2]
_BZ = [1, 5, 7, 11]   # bending in local x-y plane (v, theta_z)
_BY = [2, 4, 8, 10]   # bending in local x-z plane (w, theta_y)


def _put(M, idx, block):
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            M[:, ia, ib] = M[:, ia, ib] + block[a][b]


def k_local(L, A, I, J, E, G):
    ne = L.shape[0]
    k = np.zeros((ne, 12, 12), dtype=np.result_type(L, A))
    a = E * A / L
    _put(k, [0, 6], [[a, -a], [-a, a]])
    t = G * J / L
    _put(k, [3, 9], [[t, -t], [-t, t]])
    c = E * I / L**3
    L2 = L * L
    kz = [[12 * c, 6 * L * c, -12 * c, 6 * L * c],
          [6 * L * c, 4 * L2 * c, -6 * L * c, 2 * L2 * c],
          [-12 * c, -6 * L * c, 12 * c, -6 * L * c],
          [6 * L * c, 2 * L2 * c, -6 * L * c, 4 * L2 * c]]
    ky = [[12 * c, -6 * L * c, -12 * c, -6 * L * c],
          [-6 * L * c, 4 * L2 * c, 6 * L * c, 2 * L2 * c],
          [-12 * c, 6 * L * c, 12 * c, 6 * L * c],
          [-6 * L * c, 2 * L2 * c, 6 * L * c, 4 * L2 * c]]
    _put(k, _BZ, kz)
    _put(k, _BY, ky)
    return k


def m_local(L, A, J, rho):
    ne = L.shape[0]
    m = np.zeros((ne, 12, 12), dtype=np.result_type(L, A))
    ma = rho * A * L / 6.0
    _put(m, [0, 6], [[2 * ma, ma], [ma, 2 * ma]])
    mt = rho * J * L / 6.0
    _put(m, [3, 9], [[2 * mt, mt], [mt, 2 * mt]])
    c = rho * A * L / 420.0
    L2 = L * L
    mz = [[156 * c, 22 * L * c, 54 * c, -13 * L * c],
          [22 * L * c, 4 * L2 * c, 13 * L * c, -3 * L2 * c],
          [54 * c, 13 * L * c, 156 * c, -22 * L * c],
          [-13 * L * c, -3 * L2 * c, -22 * L * c, 4 * L2 * c]]
    my = [[156 * c, -22 * L * c, 54 * c, 13 * L * c],
          [-22 * L * c, 4 * L2 * c, -13 * L * c, -3 * L2 * c],
          [54 * c, -13 * L * c, 156 * c, 22 * L * c],
          [13 * L * c, -3 * L2 * c, 22 * L * c, 4 * L2 * c]]
    _put(m, _BZ, mz)
    _put(m, _BY, my)
    return m


def kg_local_unit(L, A, J):
    """Geometric stiffness for unit axial force N = +1 (tension positive)."""
    ne = L.shape[0]
    k = np.zeros((ne, 12, 12), dtype=np.result_type(L, A))
    c = 1.0 / L
    L2 = L * L
    gz = [[1.2 * c, 0.1 * L * c, -1.2 * c, 0.1 * L * c],
          [0.1 * L * c, 2 * L2 / 15 * c, -0.1 * L * c, -L2 / 30 * c],
          [-1.2 * c, -0.1 * L * c, 1.2 * c, -0.1 * L * c],
          [0.1 * L * c, -L2 / 30 * c, -0.1 * L * c, 2 * L2 / 15 * c]]
    gy = [[1.2 * c, -0.1 * L * c, -1.2 * c, -0.1 * L * c],
          [-0.1 * L * c, 2 * L2 / 15 * c, 0.1 * L * c, -L2 / 30 * c],
          [-1.2 * c, 0.1 * L * c, 1.2 * c, 0.1 * L * c],
          [-0.1 * L * c, -L2 / 30 * c, 0.1 * L * c, 2 * L2 / 15 * c]]
    _put(k, _BZ, gz)
    _put(k, _BY, gy)
    t = J / (A * L)
    _put(k, [3, 9], [[t, -t], [-t, t]])
    return k


def feq_local(L, q):
    """Consistent nodal loads (local) of a uniform load q (ne,3) given in local axes."""
    ne = L.shape[0]
    f = np.zeros((ne, 12), dtype=np.result_type(L, q))
    qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
    L2 = L * L
    f[:, 0] = qx * L / 2
    f[:, 1] = qy * L / 2
    f[:, 2] = qz * L / 2
    f[:, 4] = -qz * L2 / 12
    f[:, 5] = qy * L2 / 12
    f[:, 6] = qx * L / 2
    f[:, 7] = qy * L / 2
    f[:, 8] = qz * L / 2
    f[:, 10] = qz * L2 / 12
    f[:, 11] = -qy * L2 / 12
    return f


def to_global(T, kl):
    return np.einsum('eai,eab,ebj->eij', T, kl, T)


# ============================================================
# Model container
# ============================================================

class Material:
    def __init__(self, E, nu, rho):
        self.E = E
        self.nu = nu
        self.rho = rho
        self.G = E / (2.0 * (1.0 + nu))


class FrameModel:
    """A frame: nodes X (n,3), elements conn (ne,2), radii r (ne,).

    fixed      : array of constrained global DOF indices
    point_mass : (n,) translational lumped masses (kg)
    f_ext      : (6n,) design-independent nodal loads (N, N*m)
    accel      : (3,) body-force acceleration applied to the frame's own mass
                 (e.g. (0,0,-n*g)); its consistent loads depend on the design.
    """

    def __init__(self, X, conn, r, mat, fixed, tw=0.1, point_mass=None,
                 f_ext=None, accel=(0.0, 0.0, 0.0)):
        self.X = np.asarray(X, float)
        self.conn = np.asarray(conn, int)
        self.r = np.asarray(r, float)
        self.mat = mat
        self.tw = tw
        self.n = self.X.shape[0]
        self.ndof = 6 * self.n
        self.ne = self.conn.shape[0]
        self.fixed = np.unique(np.asarray(fixed, int))
        mask = np.ones(self.ndof, bool)
        mask[self.fixed] = False
        self.free = np.nonzero(mask)[0]
        self.point_mass = np.zeros(self.n) if point_mass is None else np.asarray(point_mass, float)
        self.f_ext = np.zeros(self.ndof) if f_ext is None else np.asarray(f_ext, float)
        self.accel = np.asarray(accel, float)
        # element DOF map
        d = np.zeros((self.ne, 12), int)
        for a in range(2):
            for c in range(6):
                d[:, 6 * a + c] = 6 * self.conn[:, a] + c
        self.edofs = d
        self._rows = np.repeat(d, 12, axis=1).ravel()
        self._cols = np.tile(d, (1, 12)).ravel()

    # ---------- element functions (complex-safe in Xi, Xj, r) ----------
    def elem(self, Xi=None, Xj=None, r=None):
        if Xi is None:
            Xi = self.X[self.conn[:, 0]]
        if Xj is None:
            Xj = self.X[self.conn[:, 1]]
        if r is None:
            r = self.r
        return ElementSet(Xi, Xj, r, self.mat, self.tw, self.accel)

    def assemble(self, ke):
        return sp.coo_matrix((ke.reshape(-1), (self._rows, self._cols)),
                             shape=(self.ndof, self.ndof)).tocsc()

    def assemble_vec(self, fe):
        F = np.zeros(self.ndof)
        np.add.at(F, self.edofs.ravel(), fe.reshape(-1))
        return F

    def mass_point_matrix(self):
        diag = np.zeros(self.ndof)
        for c in range(3):
            diag[c::6] = self.point_mass
        return sp.diags(diag).tocsc()

    def node_grad(self, g7):
        """Map element-level gradient (ne,7) to (dX (n,3), dr (ne,))."""
        dX = np.zeros((self.n, 3))
        np.add.at(dX, self.conn[:, 0], g7[:, 0:3])
        np.add.at(dX, self.conn[:, 1], g7[:, 3:6])
        return dX, g7[:, 6].copy()

    def cs_grad(self, fun):
        """Complex-step derivative of a per-element scalar fun(Xi,Xj,r)->(ne,).

        Returns (ne,7): derivative with respect to (xi,yi,zi,xj,yj,zj,r).
        """
        Xi = self.X[self.conn[:, 0]].astype(complex)
        Xj = self.X[self.conn[:, 1]].astype(complex)
        r = self.r.astype(complex)
        out = np.zeros((self.ne, 7))
        for k in range(3):
            Xp = Xi.copy()
            Xp[:, k] += 1j * CS_H
            out[:, k] = np.imag(fun(Xp, Xj, r)) / CS_H
            Xp = Xj.copy()
            Xp[:, k] += 1j * CS_H
            out[:, 3 + k] = np.imag(fun(Xi, Xp, r)) / CS_H
        rp = r + 1j * CS_H
        out[:, 6] = np.imag(fun(Xi, Xj, rp)) / CS_H
        return out


class ElementSet:
    """All element quantities for one design (real or complex)."""

    def __init__(self, Xi, Xj, r, mat, tw, accel):
        self.L, self.R = element_frames(Xi, Xj)
        self.T = transformation(self.R)
        self.A, self.I, self.J = chs_section(r, tw)
        self.r = r
        self.mat = mat
        self.accel = accel
        self._kl = None

    @property
    def kl(self):
        if self._kl is None:
            m = self.mat
            self._kl = k_local(self.L, self.A, self.I, self.J, m.E, m.G)
        return self._kl

    def k(self):
        return to_global(self.T, self.kl)

    def m(self):
        return to_global(self.T, m_local(self.L, self.A, self.J, self.mat.rho))

    def kg_unit(self):
        return to_global(self.T, kg_local_unit(self.L, self.A, self.J))

    def feq_loc(self):
        q_glob = self.mat.rho * self.A[:, None] * self.accel[None, :]
        q_loc = np.einsum('eij,ej->ei', self.R, q_glob)
        return feq_local(self.L, q_loc)

    def feq(self):
        return np.einsum('eji,ej->ei', self.T, self.feq_loc())

    def mass(self):
        return self.mat.rho * self.A * self.L

    def end_forces(self, ue):
        """Local end forces f = k_loc T u_e - f_eq (ne,12)."""
        ul = np.einsum('eij,ej->ei', self.T, ue)
        return np.einsum('eij,ej->ei', self.kl, ul) - self.feq_loc()

    def axial_force(self, ue):
        f = self.end_forces(ue)
        return 0.5 * (f[:, 6] - f[:, 0])

    def vm_ends(self, ue, eps_n=1e-6, eps_m=1e-9):
        """Peak von Mises stress at both ends (ne,2), smooth in the forces."""
        f = self.end_forces(ue)
        r = self.r
        out = []
        for (iN, iT, iMy, iMz) in [(0, 3, 4, 5), (6, 9, 10, 11)]:
            N = f[:, iN]
            Tq = f[:, iT]
            M2 = f[:, iMy]**2 + f[:, iMz]**2
            s = np.sqrt(N**2 + eps_n**2) / self.A + np.sqrt(M2 + eps_m**2) * r / self.I
            tau = Tq * r / self.J
            out.append(np.sqrt(s**2 + 3.0 * tau**2))
        return np.stack(out, axis=1)


# ============================================================
# Analysis: statics, eigenproblems
# ============================================================

class Analysis:
    def __init__(self, model):
        self.model = model
        self.es = model.elem()
        self.ke = self.es.k()
        self.K = model.assemble(self.ke)
        self.F = model.f_ext + model.assemble_vec(self.es.feq())
        fr = model.free
        self.Kff = self.K[fr][:, fr].tocsc()
        self.lu = spla.splu(self.Kff)
        self.u = np.zeros(model.ndof)
        self.u[fr] = self.lu.solve(self.F[fr])
        self.ue = self.u[model.edofs]
        self._M = None
        self._modes = None
        self._buck = None

    # ----- convenience -----
    def solve_adjoint(self, rhs):
        """Solve K lam = rhs on free DOFs (K symmetric)."""
        lam = np.zeros(self.model.ndof)
        fr = self.model.free
        lam[fr] = self.lu.solve(rhs[fr])
        return lam

    def reactions(self):
        return self.K @ self.u - self.F

    @property
    def M(self):
        if self._M is None:
            self._M = self.model.assemble(self.es.m()) + self.model.mass_point_matrix()
        return self._M

    def modes(self, nm=10):
        if self._modes is None or self._modes[0].size < nm:
            fr = self.model.free
            Mff = self.M[fr][:, fr].tocsc()
            # shift-invert about 0 with the already factored K
            if self.Kff.shape[0] < 200:
                w2, V = sla.eigh(self.Kff.toarray(), Mff.toarray())
                w2, V = w2[:nm], V[:, :nm]
            else:
                op = spla.LinearOperator(self.Kff.shape, matvec=self.lu.solve, dtype=float)
                w2, V = spla.eigsh(self.Kff, k=nm, M=Mff, sigma=0.0, OPinv=op, which='LM')
            nm = w2.size
            idx = np.argsort(w2)
            w2, V = w2[idx], V[:, idx]
            Phi = np.zeros((self.model.ndof, nm))
            Phi[fr] = V
            # mass-normalize (eigsh already does; enforce)
            for i in range(nm):
                Phi[:, i] /= np.sqrt(Phi[:, i] @ (self.M @ Phi[:, i]))
            self._modes = (w2, Phi)
        return self._modes

    def KG(self):
        N = self.es.axial_force(self.ue)
        kg = self.es.kg_unit() * N[:, None, None]
        return self.model.assemble(kg), N

    def buckling(self, nb=8):
        """Smallest positive load factors lam: (K + lam KG) psi = 0, psi^T K psi = 1."""
        if self._buck is None or self._buck[0].size < nb:
            fr = self.model.free
            KG, N = self.KG()
            KGff = KG[fr][:, fr].tocsc()
            Minv = spla.LinearOperator(self.Kff.shape, matvec=self.lu.solve, dtype=float)
            # -KG psi = gamma K psi ; gamma = 1/lam ; largest positive gamma
            if self.Kff.shape[0] < 200:
                gam, V = sla.eigh(-KGff.toarray(), self.Kff.toarray())
            else:
                gam, V = spla.eigsh(-KGff, k=nb + 4, M=self.Kff, Minv=Minv, which='LA',
                                    tol=1e-12, maxiter=20000)
            idx = np.argsort(-gam)
            gam, V = gam[idx], V[:, idx]
            keep = gam > 0
            gam, V = gam[keep][:nb], V[:, keep][:, :nb]
            Psi = np.zeros((self.model.ndof, gam.size))
            Psi[fr] = V
            for i in range(gam.size):
                Psi[:, i] /= np.sqrt(Psi[:, i] @ (self.K @ Psi[:, i]))
            self._buck = (1.0 / gam, Psi, N)
        return self._buck


# ============================================================
# Aggregation
# ============================================================

def ks(values, rho):
    """Kreisselmeier-Steinhauser smooth max and its weights (dKS/dvalue)."""
    v = np.asarray(values, float).ravel()
    vmax = v.max()
    e = np.exp(rho * (v - vmax))
    s = e.sum()
    return vmax + np.log(s) / rho, e / s


# ============================================================
# Responses with adjoint gradients.
# Each returns value, (dX (n,3), dr (ne,)).
# ============================================================

def _residual_term(an, lam):
    """Gradient of lam^T R(q, u) with u fixed, R = K u - F.  Returns (ne,7)."""
    model = an.model
    lam_e = lam[model.edofs]
    ue = an.ue
    accel = model.accel

    def fun(Xi, Xj, r):
        es = ElementSet(Xi, Xj, r, model.mat, model.tw, accel)
        ke = es.k()
        return np.einsum('ei,eij,ej->e', lam_e, ke, ue) - np.einsum('ei,ei->e', lam_e, es.feq())
    return model.cs_grad(fun)


def _cs_wrt_ue(an, fun_u):
    """Complex-step derivative of per-element scalar fun_u(ue)->(ne,) w.r.t. ue (ne,12)."""
    ue = an.ue.astype(complex)
    out = np.zeros(ue.shape)
    for c in range(12):
        up = ue.copy()
        up[:, c] += 1j * CS_H
        out[:, c] = np.imag(fun_u(up)) / CS_H
    return out


def response_mass(model):
    es = model.elem()
    val = float(np.sum(es.mass()))

    def fun(Xi, Xj, r):
        return ElementSet(Xi, Xj, r, model.mat, model.tw, model.accel).mass()
    return val, model.node_grad(model.cs_grad(fun))


def response_stress(an, s_lim, rho_ks, want_grad=True):
    """KS of peak von Mises / s_lim over both ends of every element."""
    model = an.model
    vm = an.es.vm_ends(an.ue)
    val, w = ks(vm / s_lim, rho_ks)
    out = {'value': val, 'max': vm.max() / s_lim, 'vm': vm}
    if not want_grad:
        return out
    W = w.reshape(vm.shape) / s_lim
    # explicit part (u fixed)

    def fexp(Xi, Xj, r):
        es = ElementSet(Xi, Xj, r, model.mat, model.tw, model.accel)
        return np.sum(W * es.vm_ends(an.ue), axis=1)
    g_exp = model.cs_grad(fexp)
    # dg/du
    es_c = model.elem()

    def fu(ue):
        return np.sum(W * es_c.vm_ends(ue), axis=1)
    gu_e = _cs_wrt_ue(an, fu)
    gu = model.assemble_vec(gu_e)
    lam = an.solve_adjoint(gu)
    g = g_exp - _residual_term(an, lam)
    out['grad'] = model.node_grad(g)
    return out


def response_displacement(an, dofs, d_lim, rho_ks, want_grad=True):
    """KS of |u_d| / d_lim over the given DOFs (smooth absolute value)."""
    model = an.model
    eps = 1e-12
    ud = an.u[dofs]
    a = np.sqrt(ud**2 + eps**2) / d_lim
    val, w = ks(a, rho_ks)
    out = {'value': val, 'max': np.abs(ud).max() / d_lim, 'u': ud}
    if not want_grad:
        return out
    gu = np.zeros(model.ndof)
    gu[dofs] = w * ud / np.sqrt(ud**2 + eps**2) / d_lim
    lam = an.solve_adjoint(gu)
    g = -_residual_term(an, lam)
    out['grad'] = model.node_grad(g)
    return out


def response_frequency(an, f_req, rho_ks, nm=10, want_grad=True):
    """KS over the lowest nm modes of f_req / f_i  (constraint: <= 1)."""
    model = an.model
    w2, Phi = an.modes(nm)
    f = np.sqrt(np.maximum(w2, 0)) / (2 * np.pi)
    ratio = f_req / f
    val, w = ks(ratio, rho_ks)
    out = {'value': val, 'max': ratio.max(), 'f': f}
    if not want_grad:
        return out
    # d ratio_i / d w2_i = -0.5 ratio_i / w2_i
    c = w * (-0.5 * ratio / w2)
    Pe = Phi[model.edofs]          # (ne,12,nm)

    def fun(Xi, Xj, r):
        es = ElementSet(Xi, Xj, r, model.mat, model.tw, model.accel)
        ke, me = es.k(), es.m()
        a = np.einsum('ein,eij,ejn->en', Pe, ke, Pe)
        b = np.einsum('ein,eij,ejn->en', Pe, me, Pe)
        return np.sum(c[None, :] * (a - w2[None, :] * b), axis=1)
    out['grad'] = model.node_grad(model.cs_grad(fun))
    return out


def response_buckling(an, lam_req, rho_ks, nb=8, want_grad=True):
    """KS over the lowest nb positive buckling factors of lam_req / lam_i (<= 1)."""
    model = an.model
    lam, Psi, N = an.buckling(nb)
    ratio = lam_req / lam
    val, w = ks(ratio, rho_ks)
    out = {'value': val, 'max': ratio.max(), 'lam': lam}
    if not want_grad:
        return out
    # dlam_i = lam_i * psi^T (dK + lam_i dKG) psi    (psi^T K psi = 1)
    c = w * (-lam_req / lam**2)                       # dKS/dlam_i
    Pe = Psi[model.edofs]
    ue0 = an.ue

    def h(Xi, Xj, r, ue):
        es = ElementSet(Xi, Xj, r, model.mat, model.tw, model.accel)
        ke = es.k()
        Ne = es.axial_force(ue)
        kgu = es.kg_unit()
        a = np.einsum('ein,eij,ejn->en', Pe, ke, Pe)
        b = np.einsum('ein,eij,ejn->en', Pe, kgu, Pe) * Ne[:, None]
        return np.sum((c * lam)[None, :] * (a + lam[None, :] * b), axis=1)

    g_exp = model.cs_grad(lambda Xi, Xj, r: h(Xi, Xj, r, ue0))
    Xi = model.X[model.conn[:, 0]]
    Xj = model.X[model.conn[:, 1]]
    gu_e = _cs_wrt_ue(an, lambda ue: h(Xi, Xj, model.r, ue))
    mu = an.solve_adjoint(model.assemble_vec(gu_e))
    g = g_exp - _residual_term(an, mu)
    out['grad'] = model.node_grad(g)
    return out
