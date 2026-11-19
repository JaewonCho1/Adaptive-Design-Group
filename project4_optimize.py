"""
Project 4 -- Optimization of the quadcopter airframe.

  * SLSQP on the scaled design vector x in [0,1]^35 (log-scaled radii, linear shape)
  * objective: frame mass / initial frame mass
  * constraints (all written g <= 1, passed to SciPy as 1 - g >= 0):
        R2 fatigue (KS-aggregated stress), R3 motor deflection (KS),
        R4 frequency (KS over the lowest 10 modes), R5 buckling (KS over the
        lowest 8 positive load factors; switched off for the "without R5" run),
        R6 slenderness (one explicit constraint per member)
    R1 (static strength) is implied by R2 and only reported.
  * runs: the main design (with R5), the same problem without R5, and several
    random starts to estimate the spread of local optima.
  * also: the lightest uniform design (all tubes one size) that meets every
    requirement, found by bisection on the common radius.

Output: project4_optimization.json (designs, histories, KKT multipliers).
Run:    python project4_optimize.py
"""

import json
import time

import numpy as np
from scipy.optimize import minimize, nnls

import project4_drone as D

OUT_FILE = 'project4_optimization.json'
PDE_KEYS_ALL = ['R2', 'R3', 'R4', 'R5']


class Problem:
    def __init__(self, use_buckling=True):
        self.use_buckling = use_buckling
        self.keys = PDE_KEYS_ALL if use_buckling else ['R2', 'R3', 'R4']
        self._x = None
        self._ev = None
        self.m0 = D.Evaluation(D.x_initial(), grads=False, buckling=False).mass_frame
        self.history = []
        self.n_eval = 0

    def ev(self, x):
        if self._x is None or not np.array_equal(x, self._x):
            self._ev = D.Evaluation(x, grads=True, buckling=self.use_buckling)
            self._x = np.array(x, copy=True)
            self.n_eval += 1
        return self._ev

    def f(self, x):
        return self.ev(x).mass_frame / self.m0

    def df(self, x):
        return self.ev(x).dmass / self.m0

    def c(self, x):
        e = self.ev(x)
        return np.concatenate([[1.0 - e.g[k] for k in self.keys],
                               1.0 - e.slender / D.SPEC['slender_max']])

    def dc(self, x):
        e = self.ev(x)
        return np.vstack([-e.dg[k] for k in self.keys] + [-e.dslender])

    def callback(self, x):
        e = self.ev(x)
        self.history.append({
            'mass': float(e.mass_frame),
            'g': {k: float(e.g[k]) for k in self.keys},
            'gtrue': {k: float(e.gtrue[k]) for k in self.keys},
            'x': [float(v) for v in x],
        })


def run(x0, use_buckling=True, maxiter=300, ftol=1e-7, label=''):
    P = Problem(use_buckling)
    P.callback(np.asarray(x0, float))
    t0 = time.time()
    res = minimize(P.f, x0, jac=P.df, method='SLSQP',
                   bounds=[(0.0, 1.0)] * D.N_VAR,
                   constraints=[{'type': 'ineq', 'fun': P.c, 'jac': P.dc}],
                   callback=P.callback,
                   options={'maxiter': maxiter, 'ftol': ftol})
    wall = time.time() - t0
    x = np.clip(res.x, 0, 1)
    print(f'[{label}] {res.message} | it={res.nit} evals={P.n_eval} '
          f'mass={P.f(x)*P.m0:.4f} kg | {wall:.1f} s | g=' +
          ', '.join(f'{k}:{P.ev(x).g[k]:.4f}' for k in P.keys))
    return {'x': [float(v) for v in x], 'success': bool(res.success), 'message': str(res.message),
            'nit': int(res.nit), 'n_eval': P.n_eval, 'wall_s': wall,
            'mass': float(P.f(x) * P.m0), 'history': P.history,
            'use_buckling': use_buckling, 'kkt': kkt(P, x)}


def kkt(P, x, tol_act=2e-3, tol_bnd=1e-6):
    """Lagrange multipliers from the stationarity condition on the active set.

    grad f = sum_i mu_i grad c_i + bound multipliers, mu_i >= 0 (NNLS).
    In physical terms d(mass)/d(limit relaxation) for constraint i is
    mu_i * m0 (kg per unit of g), because g = response / limit.
    """
    e = P.ev(x)
    gf = P.df(x)
    names = list(P.keys) + [f'R6[{i}]' for i in range(D.N_MEM)]
    cval = P.c(x)
    J = -P.dc(x)                    # gradients of g (g <= 1)
    act = np.nonzero(cval < tol_act)[0]
    lo = np.nonzero(x < tol_bnd)[0]
    hi = np.nonzero(x > 1 - tol_bnd)[0]
    # columns: -grad g_i (constraint pushes mass up), +e_lo, -e_hi
    cols, labels = [], []
    for i in act:
        cols.append(-J[i])
        labels.append(names[i])
    for i in lo:
        v = np.zeros(D.N_VAR)
        v[i] = 1.0
        cols.append(v)
        labels.append(f'lower bound x[{i}]')
    for i in hi:
        v = np.zeros(D.N_VAR)
        v[i] = -1.0
        cols.append(v)
        labels.append(f'upper bound x[{i}]')
    if cols:
        A = np.array(cols).T
        mu, resid = nnls(A, gf)
    else:
        mu, resid = np.zeros(0), float(np.linalg.norm(gf))
    out = {'active': [], 'stationarity_residual': float(resid),
           'grad_norm': float(np.linalg.norm(gf)), 'n_lower': int(lo.size), 'n_upper': int(hi.size)}
    for lab, m in zip(labels, mu):
        if lab.startswith('R'):
            out['active'].append({'name': lab, 'mu': float(m), 'kg_per_unit': float(m * P.m0)})
    return out


def lightest_uniform(sh=D.SHAPE0, lo=3e-3, hi=20e-3):
    def ok(r):
        e = D.Evaluation(D.scale(np.full(D.N_GROUP, r), sh), grads=False)
        return all(m['ok'] for m in e.metrics() if m['ok'] is not None)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def main(n_starts=3, seed=4):
    out = {}
    x0 = D.x_initial()
    r_u = lightest_uniform()
    out['uniform'] = {'r': r_u, 'x': [float(v) for v in D.scale(np.full(D.N_GROUP, r_u), D.SHAPE0)]}
    print(f'lightest uniform radius: {r_u*1e3:.3f} mm')
    out['main'] = run(x0, True, label='with R5')
    out['no_buckling'] = run(x0, False, label='without R5')
    rng = np.random.default_rng(seed)
    out['starts'] = []
    for s in range(n_starts):
        xs = rng.uniform(0.2, 0.8, D.N_VAR)
        r = run(xs, True, label=f'start {s+1}')
        r.pop('history')
        out['starts'].append(r)
    with open(OUT_FILE, 'w') as f:
        json.dump(out, f)
    print('wrote', OUT_FILE)


if __name__ == '__main__':
    main()
