"""
Project 4 -- Build the results file, figures, and report.

Reads   project4_verification.json   (from project4_verify.py)
        project4_optimization.json   (from project4_optimize.py)
Writes  assets/project4_results.json (read by the 3D viewer)
        figures/project4_*.png
        project4_report.md           (filled in from project4_report_template.md,
                                      so no number in the report is typed by hand)

Run everything with:  python project4_run_all.py
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

import project4_drone as D

FIG = Path('figures')
ASSETS = Path('assets')
FIG.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)

INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def rnd(a, sig=5):
    a = np.asarray(a, float)
    return [float(f'{v:.{sig}g}') for v in a.ravel()]


def mode_field(vec, n):
    v = vec.reshape(n, 6)
    s = np.abs(v[:, :3]).max()
    return v / s


def export_design(name, label, x):
    ev = D.Evaluation(np.asarray(x), grads=False)
    m = ev.model
    an = ev.an
    n = m.n
    w2, Phi = an.modes(10)
    lam, Psi, N = an.buckling(8)
    es = an.es
    util = ev.vm.max(axis=1) / D.SPEC['S_fatigue']
    mem_slender = ev.slender
    metrics = []
    for q in ev.metrics():
        q = dict(q)
        q['value'] = float(q['value'])
        metrics.append(q)
    return ev, {
        'name': name, 'label': label,
        'nodes': rnd(m.X, 6),
        'u': rnd(an.u.reshape(n, 6), 5),
        'mode': rnd(mode_field(Phi[:, 0], n), 4),
        'mode_freq': float(np.sqrt(w2[0]) / (2 * np.pi)),
        'buck': rnd(mode_field(Psi[:, 0], n), 4),
        'buck_lam': float(lam[0]),
        'r': rnd(m.r, 5),
        'R': rnd(es.R, 6),
        'vm': rnd(ev.vm / 1e6, 4),
        'util': rnd(util, 4),
        'N': rnd(N, 4),
        'slender': rnd(mem_slender[D.ELEM_MEMBER], 4),
        'L_member': rnd(D.member_lengths(ev.sh), 4),
        'metrics': metrics,
        'freqs': rnd(ev.freqs, 5),
        'lams': rnd(lam, 5),
        'radii_groups_mm': rnd(ev.rg * 1e3, 4),
        'shape': rnd(ev.sh, 5),
    }


def fig_history(hist):
    it = np.arange(len(hist))
    mass = [h['mass'] for h in hist]
    fig, axs = plt.subplots(2, 1, figsize=(6.0, 5.6), sharex=True)
    axs[0].plot(it, mass, '-o', color=C1, lw=2, ms=4)
    axs[0].set_ylabel('frame mass (kg)', color=INK)
    axs[0].set_title('Optimization history (SLSQP)', color=INK, fontsize=11)
    names = {'R2': 'R2 fatigue', 'R3': 'R3 deflection', 'R4': 'R4 frequency', 'R5': 'R5 buckling'}
    for k, c, mk in zip(['R2', 'R3', 'R4', 'R5'], [C1, C2, C3, C4], ['o', 's', '^', 'D']):
        axs[1].plot(it, [h['g'][k] for h in hist], '-' + mk, color=c, lw=2, ms=4, label=names[k])
    axs[1].axhline(1.0, color=INK2, lw=1, ls='--')
    axs[1].text(it[-1], 1.03, 'limit', color=INK2, ha='right', va='bottom', fontsize=8)
    axs[1].set_yscale('log')
    axs[1].set_ylabel('constraint value g (feasible if <= 1)', color=INK)
    axs[1].set_xlabel('iteration', color=INK)
    axs[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axs[1].legend(frameon=False, fontsize=8, ncol=2)
    for a in axs:
        style(a)
    fig.tight_layout()
    fig.savefig(FIG / 'project4_convergence.png', dpi=160)
    plt.close(fig)


def fig_radii(r_opt, r_uni):
    g = D.GROUPS
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    ax.barh(y, r_opt, height=0.7, color=C1, label='optimized')
    ax.axvline(D.R0 * 1e3, color=C2, lw=1.5, ls='--', label=f'initial (all {D.R0*1e3:.0f} mm)')
    ax.axvline(r_uni, color=C3, lw=1.5, ls=':', label=f'lightest uniform ({r_uni:.1f} mm)')
    ax.axvline(D.SPEC['r_min'] * 1e3, color=INK2, lw=1, label='minimum gauge')
    ax.set_yticks(y)
    ax.set_yticklabels(g, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('outer radius r (mm); wall t = r/10', color=INK)
    ax.set_title('Member sizes by group (one arm design, built four times)', color=INK, fontsize=10)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, framealpha=0.95, edgecolor=GRID, fontsize=8, loc='center right')
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'project4_radii.png', dpi=160)
    plt.close(fig)


def fig_modes(ev_init, ev_opt, ev_uni):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    k = np.arange(1, 11)
    for ev, c, mk, lab in [(ev_init, C2, 's', 'initial'), (ev_uni, C3, '^', 'lightest uniform'),
                           (ev_opt, C1, 'o', 'optimized')]:
        ax.plot(k, ev.freqs[:10], '-' + mk, color=c, lw=2, ms=6, label=lab)
    ax.axhline(D.SPEC['f_req'], color=INK2, ls='--', lw=1)
    ax.text(10, D.SPEC['f_req'] + 2, f"required {D.SPEC['f_req']:.0f} Hz", color=INK2, ha='right', fontsize=8)
    ax.set_xlabel('mode number', color=INK)
    ax.set_ylabel('natural frequency (Hz)', color=INK)
    ax.set_title('Lowest ten natural frequencies', color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'project4_frequencies.png', dpi=160)
    plt.close(fig)


def fmt(v, d=2):
    return f'{v:,.{d}f}'


def main():
    ver = json.load(open('project4_verification.json'))
    opt = json.load(open('project4_optimization.json'))
    x_init = D.x_initial()
    x_uni = np.array(opt['uniform']['x'])
    x_opt = np.array(opt['main']['x'])
    x_nob = np.array(opt['no_buckling']['x'])
    ev_i, d_i = export_design('initial', f'Initial (all tubes r = {D.R0*1e3:.0f} mm)', x_init)
    ev_u, d_u = export_design('uniform', f"Lightest uniform (r = {opt['uniform']['r']*1e3:.1f} mm)", x_uni)
    ev_o, d_o = export_design('optimized', 'Optimized', x_opt)
    S = D.SPEC
    results = {
        'title': 'Adaptive Design Group - Project 4: quadcopter airframe',
        'spec': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in S.items()},
        'elements': D.CONN.tolist(),
        'elem_member': D.ELEM_MEMBER.tolist(),
        'member_group': D.MEM_GROUP.tolist(),
        'groups': D.GROUPS,
        'supports': D.SUPPORT_NODES.tolist(),
        'motors': D.MOTOR_NODES.tolist(),
        'yaw_sign': D.YAW_SIGN.tolist(),
        'designs': [d_i, d_o, d_u],
        'history': [{'mass': h['mass'], 'g': h['g']} for h in opt['main']['history']],
    }
    with open(ASSETS / 'project4_results.json', 'w') as f:
        json.dump(results, f, separators=(',', ':'))
    fig_history(opt['main']['history'])
    fig_radii(ev_o.rg * 1e3, opt['uniform']['r'] * 1e3)
    fig_modes(ev_i, ev_o, ev_u)

    # ---------- numbers for the report ----------
    V = {}
    for tag, ev in [('i', ev_i), ('u', ev_u), ('o', ev_o)]:
        for q in ev.metrics():
            val = q['value']
            dec = {'OBJ': 3, 'R1': 1, 'R2': 2, 'R3': 2, 'R4': 1, 'R5': 1, 'R6': 0}[q['id']]
            mark = '' if q['ok'] is None else (' ✓' if q['ok'] else ' ✗')
            V[f'{q["id"]}_{tag}'] = fmt(val, dec) + mark
        V[f'mass_{tag}'] = fmt(ev.mass_frame, 3)
        V[f'h_root_{tag}'] = fmt(ev.sh[0] * 1e3, 0)
        V[f'h_tip_{tag}'] = fmt(ev.sh[1] * 1e3, 0)
        V[f'w_root_{tag}'] = fmt(ev.sh[2] * 1e3, 0)
        V[f'w_tip_{tag}'] = fmt(ev.sh[3] * 1e3, 0)
        V[f'auw_{tag}'] = fmt(ev.mass_frame / S['MTOW'] * 100, 1)
    V['f1_i'] = fmt(ev_i.freqs.min(), 1)
    V['r_uni'] = fmt(opt['uniform']['r'] * 1e3, 2)
    V['save_vs_uni'] = fmt((1 - ev_o.mass_frame / ev_u.mass_frame) * 100, 0)
    V['save_vs_init'] = fmt((1 - ev_o.mass_frame / ev_i.mass_frame) * 100, 0)
    m = opt['main']
    V['nit'] = str(m['nit'])
    V['neval'] = str(m['n_eval'])
    V['wall'] = fmt(m['wall_s'], 0)
    V['ks_true_R2'] = fmt(ev_o.gtrue['R2'], 3)
    V['ks_R2'] = fmt(ev_o.g['R2'], 3)
    V['sigma_opt'] = fmt(ev_o.sigma_max / 1e6, 1)
    V['f1_ks'] = fmt(S['f_req'] / ev_o.g['R4'], 1)
    for a in m['kkt']['active']:
        V[f"mu_{a['name']}"] = fmt(a['mu'], 4)
        V[f"kg_{a['name']}"] = fmt(a['kg_per_unit'], 3)
        if a['name'] == 'R4':
            V['kg_per_Hz'] = fmt(a['kg_per_unit'] / S['f_req'] * 1000, 0)     # g per Hz
            V['pct_per_Hz'] = fmt(a['kg_per_unit'] / S['f_req'] / ev_o.mass_frame * 100, 1)
    V['kkt_resid'] = f"{m['kkt']['stationarity_residual']:.1e}"
    V['kkt_grad'] = fmt(m['kkt']['grad_norm'], 3)
    V['n_lower'] = str(m['kkt']['n_lower'])
    V['n_upper'] = str(m['kkt']['n_upper'])
    starts = opt['starts']
    V['n_starts'] = str(len(starts))
    sm = [s['mass'] for s in starts] + [m['mass']]
    spread_g = (max(sm) - min(sm)) * 1000
    V['start_spread'] = 'less than 0.01 g' if spread_g < 0.01 else f'{spread_g:.2f} g'
    V['util_max'] = fmt(ev_o.vm.max() / S['S_fatigue'] * 100, 0)
    V['start_dx'] = f"{max(np.abs(np.array(s['x']) - x_opt).max() for s in starts):.1e}"
    V['nob_dx'] = f'{np.abs(x_nob - x_opt).max():.1e}'
    V['nob_mass'] = fmt(opt['no_buckling']['mass'], 3)
    V['nob_nit'] = str(opt['no_buckling']['nit'])
    V['lam_o'] = fmt(ev_o.lams.min(), 1)
    V['lam_i'] = fmt(ev_i.lams.min(), 1)
    V['lam_u'] = fmt(ev_u.lams.min(), 1)
    # buckling: compressive members at the optimum
    lam, Psi, N = ev_o.an.buckling(8)
    V['bow_amp'] = fmt(1 / (1 - 1 / ev_o.lams.min()), 3)
    V['nmin_members'] = str(int(np.sum(ev_o.rg <= S['r_min'] * 1.0001)))
    # requirements constants
    V['S_fat'] = fmt(S['S_fatigue'] / 1e6, 1)
    V['S_e'] = fmt(S['S_e'] / 1e6, 2)
    V['S_static'] = fmt(S['S_static'] / 1e6, 0)
    V['n_load'] = fmt(S['n_load'], 2)
    V['f_1P'] = fmt(S['f_1P'], 1)
    V['f_req'] = fmt(S['f_req'], 0)
    # verification table
    rows = []
    for r in ver['ladder']:
        if r['test'] in ('V4',):
            err = f"{r['rel_error']*100:.2f}% (textbook value for cubic elements)"
        elif r['test'] == 'V3' and 'slope' in r['quantity']:
            err = f"slope {r['ours']:.2f} vs 4"
        elif r['test'] == 'V5' and 'zero' in r['quantity']:
            err = r.get('note', '')
        else:
            err = f"{r['rel_error']:.1e}"
        ours = f"{r['ours']:.6g}"
        exact = f"{r['exact']:.6g}"
        rows.append(f"| {r['test']} | {r['quantity']} | {ours} | {exact} | {err} | {'✓' if r['pass'] else '✗'} |")
    V['ladder_table'] = '\n'.join(rows)
    g_rows = []
    for c in ver['gradient_checks']:
        if c['trial'] == 0:
            g_rows.append(f"| {c['response']} | {c['kind']} | {c['adjoint']:.8e} | {c['fd']:.8e} | {c['rel_error']:.1e} |")
    V['grad_table'] = '\n'.join(g_rows)
    V['grad_worst'] = f"{max(c['rel_error'] for c in ver['gradient_checks']):.1e}"
    cost = ver['cost']
    V['t_an'] = fmt(cost['t_analysis_s'], 2)
    V['t_adj'] = fmt(cost['t_analysis_plus_all_gradients_s'], 2)
    V['t_ffd'] = fmt(cost['t_forward_fd_s'], 1)
    V['t_cfd'] = fmt(cost['t_central_fd_s'], 1)
    V['speedup'] = fmt(cost['t_central_fd_s'] / cost['t_analysis_plus_all_gradients_s'], 1)
    V['adj_ratio'] = fmt(cost['t_analysis_plus_all_gradients_s'] / cost['t_analysis_s'], 0)
    V['t_fd_ungrouped'] = fmt(2 * (D.N_MEM + D.N_SHAPE) * cost['t_analysis_s'], 0)
    V['N_comp'] = fmt(-N.min(), 0)
    V['conv_slope'] = fmt(ver['convergence']['slope'], 2)
    # group radii table
    gr = []
    for gname, ri in zip(D.GROUPS, ev_o.rg):
        gr.append(f'| {gname} | {ri*1e3:.2f} |' + (' minimum gauge |' if ri <= S['r_min'] * 1.0001 else ' |'))
    V['radii_table'] = '\n'.join(gr)
    V['freqs_o'] = ', '.join(f'{f:.1f}' for f in ev_o.freqs[:6])
    V['freqs_i'] = ', '.join(f'{f:.1f}' for f in ev_i.freqs[1:6])

    tpl = open('project4_report_template.md').read()
    missing = set()

    def sub(mo):
        k = mo.group(1)
        if k not in V:
            missing.add(k)
            return mo.group(0)
        return V[k]
    out = re.sub(r'@@([A-Za-z0-9_]+)@@', sub, tpl)
    if missing:
        raise KeyError(f'report template keys without values: {sorted(missing)}')
    open('project4_report.md', 'w').write(out)
    print('wrote assets/project4_results.json, figures, project4_report.md')
    return V


if __name__ == '__main__':
    main()
