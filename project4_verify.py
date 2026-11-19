"""
Project 4 -- Verification ladder (V1-V6) for the 3D frame code.

Every expected value below is a closed-form textbook result, written here
before the solver is called:

  V1  cantilever: tip load in each bending plane  delta = P L^3 / (3 E I),
      tip torque  phi = T L / (G J)
  V2  uniformly loaded beams:  simply supported midspan 5 q L^4 / (384 E I),
      cantilever tip q L^4 / (8 E I); then the cantilever rotated by an
      arbitrary 3D rotation (result must be invariant)
  V3  simply supported beam frequency  f1 = (pi / L)^2 sqrt(E I / (rho A)) / (2 pi),
      mesh refinement 1..32 elements; expected slope 4 (cubic Hermite, consistent mass)
  V4  pinned-pinned Euler column  P_cr = pi^2 E I / L^2  with 1, 2, 4 elements
  V5  global checks on the quadcopter frame: free-free model has exactly six
      zero-frequency modes; support reactions balance the applied load (force and
      moment); sparse Lanczos eigenvalues equal dense eigenvalues
  V6  adjoint gradients vs central finite differences for every response and both
      kinds of design variable (sizes, shape), plus a Taylor test (error vs step)

Run:  python project4_verify.py
Output: project4_verification.json, figures/project4_v3_convergence.png,
        figures/project4_taylor_test.png
"""

import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project4_frame import Material, FrameModel, Analysis, chs_section
import project4_drone as D

FIG = Path('figures')
FIG.mkdir(exist_ok=True)
OUT = 'project4_verification.json'

MAT = Material(68.9e9, 0.33, 2700.0)
R_TEST = 0.02
TW = 0.1
A_T, I_T, J_T = chs_section(R_TEST, TW)

INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'


def style(ax):
    ax.grid(True, which='major', color=GRID, lw=0.6)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def line_beam(ne, L=1.0, direction=(1, 0, 0)):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    X = np.outer(np.linspace(0, L, ne + 1), d)
    conn = np.c_[np.arange(ne), np.arange(1, ne + 1)]
    return X, conn


def rel(a, b):
    return abs(a - b) / abs(b)


rows = []


def add(test, quantity, ours, exact, tol=1e-6):
    e = rel(ours, exact)
    rows.append({'test': test, 'quantity': quantity, 'ours': float(ours), 'exact': float(exact),
                 'rel_error': float(e), 'pass': bool(e < tol)})
    print(f'{test:4s} {quantity:48s} ours={ours: .6e} exact={exact: .6e} err={e:.2e}')


# ------------------------------------------------------------------ V1
def v1():
    L, P, T, ne = 1.0, 100.0, 10.0, 4
    X, conn = line_beam(ne, L)
    F = np.zeros(6 * (ne + 1))
    F[6 * ne + 1] = P
    F[6 * ne + 2] = P
    F[6 * ne + 3] = T
    an = Analysis(FrameModel(X, conn, np.full(ne, R_TEST), MAT, range(6), tw=TW, f_ext=F))
    add('V1', 'cantilever tip deflection, x-y plane', an.u[6 * ne + 1], P * L**3 / (3 * MAT.E * I_T))
    add('V1', 'cantilever tip deflection, x-z plane', an.u[6 * ne + 2], P * L**3 / (3 * MAT.E * I_T))
    add('V1', 'cantilever tip rotation, x-y plane', an.u[6 * ne + 5], P * L**2 / (2 * MAT.E * I_T))
    add('V1', 'cantilever tip twist', an.u[6 * ne + 3], T * L / (MAT.G * J_T))


# ------------------------------------------------------------------ V2
def v2():
    L, ne, a = 1.0, 4, 50.0
    q = MAT.rho * A_T * a
    X, conn = line_beam(ne, L)
    fixed = [0, 1, 2, 3, 6 * ne + 1, 6 * ne + 2]
    an = Analysis(FrameModel(X, conn, np.full(ne, R_TEST), MAT, fixed, tw=TW, accel=(0, 0, -a)))
    add('V2', 'simply supported midspan deflection (uniform q)', -an.u[6 * (ne // 2) + 2],
        5 * q * L**4 / (384 * MAT.E * I_T))
    an = Analysis(FrameModel(X, conn, np.full(ne, R_TEST), MAT, range(6), tw=TW, accel=(0, 0, -a)))
    tip = -an.u[6 * ne + 2]
    add('V2', 'cantilever tip deflection (uniform q)', tip, q * L**4 / (8 * MAT.E * I_T))
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    Xr = X @ Q.T
    accr = Q @ np.array([0, 0, -a])
    an2 = Analysis(FrameModel(Xr, conn, np.full(ne, R_TEST), MAT, range(6), tw=TW, accel=accr))
    add('V2', 'rotated cantilever: magnitude of tip displacement', np.linalg.norm(an2.u[6 * ne:6 * ne + 3]), tip)


# ------------------------------------------------------------------ V3
def v3():
    L = 1.0
    f_ex = (np.pi / L)**2 * np.sqrt(MAT.E * I_T / (MAT.rho * A_T)) / (2 * np.pi)
    nes = [1, 2, 4, 8, 16, 32]
    errs = []
    for ne in nes:
        X, conn = line_beam(ne, L)
        fixed = [0, 1, 2, 3, 6 * ne, 6 * ne + 1, 6 * ne + 2]
        an = Analysis(FrameModel(X, conn, np.full(ne, R_TEST), MAT, fixed, tw=TW))
        w2, _ = an.modes(2)
        f1 = np.sqrt(w2[0]) / (2 * np.pi)
        errs.append(rel(f1, f_ex))
        if ne == 8:
            add('V3', 'SS beam first frequency, 8 elements (Hz)', f1, f_ex, tol=1e-4)
    errs = np.array(errs)
    h = L / np.array(nes)
    slope = np.polyfit(np.log(h[1:5]), np.log(errs[1:5]), 1)[0]
    rows.append({'test': 'V3', 'quantity': 'convergence slope of frequency error vs h',
                 'ours': float(slope), 'exact': 4.0, 'rel_error': float(abs(slope - 4) / 4),
                 'pass': bool(abs(slope - 4) < 0.15)})
    print(f'V3 slope {slope:.3f}')
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.loglog(h, errs, '-o', color=C1, lw=2, ms=7, label='3D frame code')
    ax.loglog(h, errs[1] * (h / h[1])**4, '--', color=INK2, lw=1, label='slope 4 reference')
    ax.set_xlabel('element length h / L', color=INK)
    ax.set_ylabel('relative error in first frequency', color=INK)
    ax.set_title(f'V3: mesh convergence (fitted slope {slope:.2f})', color=INK, fontsize=11)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'project4_v3_convergence.png', dpi=160)
    plt.close(fig)
    return {'h': h.tolist(), 'err': errs.tolist(), 'slope': float(slope)}


# ------------------------------------------------------------------ V4
def v4():
    L = 1.0
    pcr = np.pi**2 * MAT.E * I_T / L**2
    textbook = {1: 0.2159, 2: 0.0075, 4: 0.0005}
    for ne in [1, 2, 4]:
        X, conn = line_beam(ne, L)
        F = np.zeros(6 * (ne + 1))
        F[6 * ne] = -1.0
        fixed = [0, 1, 2, 3, 6 * ne + 1, 6 * ne + 2]
        an = Analysis(FrameModel(X, conn, np.full(ne, R_TEST), MAT, fixed, tw=TW, f_ext=F))
        lam = an.buckling(2)[0][0]
        e = rel(lam, pcr)
        rows.append({'test': 'V4', 'quantity': f'Euler column P_cr, {ne} element(s) (N)', 'ours': float(lam),
                     'exact': float(pcr), 'rel_error': float(e),
                     'pass': bool(abs(e - textbook[ne]) < 1e-3 + 0.02 * textbook[ne]),
                     'note': f'expected discretization error {textbook[ne]*100:.2f}% (cubic element)'})
        print(f'V4 ne={ne} P={lam:.4e} exact={pcr:.4e} err={e:.4%}')


# ------------------------------------------------------------------ V5
def v5():
    x0 = D.x_initial()
    rg, sh = D.unscale(x0)
    mf = D.build_model(rg, sh, supports='none')
    an = type('A', (), {})()
    from project4_frame import ElementSet
    es = mf.elem()
    K = mf.assemble(es.k()).toarray()
    M = (mf.assemble(es.m()) + mf.mass_point_matrix()).toarray()
    w2 = sla.eigh(K, M, eigvals_only=True, subset_by_index=[0, 9])
    f = np.sqrt(np.abs(w2)) / (2 * np.pi)
    nzero = int(np.sum(f < 1e-3 * f[6]))   # round-off floor of the dense solver
    rows.append({'test': 'V5', 'quantity': 'free-free frame: number of zero-frequency modes',
                 'ours': nzero, 'exact': 6, 'rel_error': float(abs(nzero - 6) / 6), 'pass': nzero == 6,
                 'note': f'7th frequency {f[6]:.2f} Hz; largest of the six {f[5]:.1e} Hz'})
    print('V5 free-free first freqs', f[:8])
    # supported: reactions balance applied loads
    m = D.build_model(rg, sh)
    a = Analysis(m)
    R = a.reactions()
    Fa = a.F
    X = m.X
    add('V5', 'sum of vertical reactions = applied vertical load (N)', -R[2::6].sum(), Fa[2::6].sum(), tol=1e-9)
    # moment about origin
    def moment(vec):
        Fn = vec.reshape(-1, 6)
        return np.sum(np.cross(X, Fn[:, :3]), axis=0) + Fn[:, 3:].sum(axis=0)
    Mr, Ma = moment(R), moment(Fa)
    scale = np.abs(Fa[2::6]).sum() * 0.65
    e = np.linalg.norm(Mr + Ma) / scale
    rows.append({'test': 'V5', 'quantity': 'moment balance: norm(M_react + M_applied) / (F L)', 'ours': float(e),
                 'exact': 0.0, 'rel_error': float(e), 'pass': bool(e < 1e-10)})
    # sparse Lanczos vs dense eigenvalues (supported model)
    fr = m.free
    Kd = a.K[fr][:, fr].toarray()
    Md = a.M[fr][:, fr].toarray()
    wd = sla.eigh(Kd, Md, eigvals_only=True, subset_by_index=[0, 4])
    ws, _ = a.modes(5)
    add('V5', 'lowest frequency: sparse shift-invert vs dense (Hz)', np.sqrt(ws[0]) / (2 * np.pi),
        np.sqrt(wd[0]) / (2 * np.pi), tol=1e-7)


# ------------------------------------------------------------------ V6
def v6():
    rng = np.random.default_rng(7)
    x0 = D.x_initial()
    x0 = x0 + 0.05 * rng.uniform(-1, 1, x0.size)     # a generic, non-symmetric point
    t0 = time.time()
    ev = D.Evaluation(x0, grads=True)
    t_adj = time.time() - t0
    t0 = time.time()
    D.Evaluation(x0, grads=False)
    t_an = time.time() - t0
    keys = ['mass', 'R2', 'R3', 'R4', 'R5', 'R6']
    # R6 is checked on the most slender member whose length depends on the shape
    dep = np.abs(ev.dslender[:, D.N_GROUP:]).sum(axis=1) > 0
    i6 = int(np.argmax(np.where(dep, ev.slender, -1)))

    def values(e):
        return {'mass': e.mass_frame, 'R2': e.g['R2'], 'R3': e.g['R3'], 'R4': e.g['R4'],
                'R5': e.g['R5'], 'R6': e.slender[i6] / D.SPEC['slender_max']}
    grads = {'mass': ev.dmass, 'R2': ev.dg['R2'], 'R3': ev.dg['R3'], 'R4': ev.dg['R4'],
             'R5': ev.dg['R5'], 'R6': ev.dslender[i6]}
    checks = []
    h = 1e-6
    for kind in ['size', 'shape']:
        for trial in range(2):
            p = np.zeros(D.N_VAR)
            if kind == 'size':
                p[:D.N_GROUP] = rng.standard_normal(D.N_GROUP)
            else:
                p[D.N_GROUP:] = rng.standard_normal(D.N_SHAPE)
            p /= np.linalg.norm(p)
            vp = values(D.Evaluation(x0 + h * p, grads=False))
            vm = values(D.Evaluation(x0 - h * p, grads=False))
            for k in keys:
                fd = (vp[k] - vm[k]) / (2 * h)
                ad = float(grads[k] @ p)
                checks.append({'response': k, 'kind': kind, 'trial': trial, 'adjoint': ad, 'fd': float(fd),
                               'rel_error': float(abs(ad - fd) / max(abs(fd), 1e-14))})
                print(f'V6 {k:4s} {kind:5s} adj={ad: .8e} fd={fd: .8e} err={checks[-1]["rel_error"]:.1e}')
    worst = max(c['rel_error'] for c in checks)
    rows.append({'test': 'V6', 'quantity': 'adjoint vs central FD, worst of 24 checks', 'ours': worst,
                 'exact': 0.0, 'rel_error': worst, 'pass': bool(worst < 1e-5)})
    # Taylor test
    p = rng.standard_normal(D.N_VAR)
    p /= np.linalg.norm(p)
    hs = np.logspace(-1, -9, 17)
    tay = {k: [] for k in ['R2', 'R3', 'R4', 'R5']}
    for hh in hs:
        vp = values(D.Evaluation(x0 + hh * p, grads=False))
        vm = values(D.Evaluation(x0 - hh * p, grads=False))
        for k in tay:
            fd = (vp[k] - vm[k]) / (2 * hh)
            ad = grads[k] @ p
            tay[k].append(abs(fd - ad) / abs(ad))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    labels = {'R2': 'R2 fatigue (stress KS)', 'R3': 'R3 deflection', 'R4': 'R4 frequency',
              'R5': 'R5 buckling'}
    for k, c, mk in zip(tay, [C1, C2, C3, C4], ['o', 's', '^', 'D']):
        ax.loglog(hs, tay[k], '-' + mk, color=c, lw=2, ms=6, label=labels[k])
    ax.loglog(hs[:8], 0.5 * tay['R4'][0] * (hs[:8] / hs[0])**2, '--', color=INK2, lw=1, label='slope 2 reference')
    ax.set_xlabel('finite-difference step h (scaled design units)', color=INK)
    ax.set_ylabel('|FD - adjoint| / |adjoint|', color=INK)
    ax.set_title('V6: Taylor test of the adjoint gradients', color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'project4_taylor_test.png', dpi=160)
    plt.close(fig)
    n = D.N_VAR
    cost = {'t_analysis_s': t_an, 't_analysis_plus_all_gradients_s': t_adj,
            'n_var': n, 't_forward_fd_s': (n + 1) * t_an, 't_central_fd_s': 2 * n * t_an}
    print('cost', cost)
    return checks, {'h': hs.tolist(), **{k: [float(v) for v in tay[k]] for k in tay}}, cost


if __name__ == '__main__':
    v1()
    v2()
    conv = v3()
    v4()
    v5()
    checks, taylor, cost = v6()
    json.dump({'ladder': rows, 'convergence': conv, 'gradient_checks': checks, 'taylor': taylor,
               'cost': cost}, open(OUT, 'w'), indent=1)
    print('all pass:', all(r['pass'] for r in rows))
