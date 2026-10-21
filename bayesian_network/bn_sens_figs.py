# -*- coding: utf-8 -*-
"""Sensitivity (tornado) and figures of the v3 BN: DAG, annual profile, breakdown by cause, tornado"""
import copy, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_bn as bn

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 8.5,
                     'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#898781', 'ytick.color': '#898781',
                     'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2 = '#0b0b0b', '#52514e'
os.makedirs('fig', exist_ok=True); os.makedirs('out', exist_ok=True)
LAB = {'stop': 'downstream stop + flare failure', 'demand': 'ignition failure on flare demand', 'foam': 'foam blocks the outlet', 'block': 'condensate / freezing / XV blockage', 'wv': 'vent valve stuck'}

inp = bn.load_inputs()

# ----------------------------------------------------------------------------- sensitivity (Case 1)
def total_with(mod):
    prm = copy.deepcopy(bn.PARAMS)
    for k, v in mod.items():
        prm[k]['v'] = v
    df = bn.run_year(1, inp, prm)
    s = bn.summarize(df, prm)['total']
    return s['openings_per_yr'], s['vent_m3_per_yr']

base_n, base_v = total_with({})
sens = [
    ('flare failure on demand 0.01 / 0.10', {'pfd_flare': 0.01}, {'pfd_flare': 0.10}),
    ('downstream stop frequency 6 / 24 per year', {'lam_stop': 6.0}, {'lam_stop': 24.0}),
    ('foaming 1 / 3 per year', {'lam_foam': 1.0}, {'lam_foam': 3.0}),
    ('probability that foam fouls the PVRV 0.03 / 0.3', {'p_pv_fouled': 0.3}, {'p_pv_fouled': 0.03}),
    ('operator response: always daytime / always night', {'R_night': {0.5: 0.6, 2.0: 0.3, 8.0: 0.1}}, {'R_day': {0.5: 0.2, 2.0: 0.4, 8.0: 0.4}}),
    ('8-h response share 0 (alarm trusted) / doubled', {'R_day': {0.5: 0.65, 2.0: 0.35, 8.0: 0.0}, 'R_night': {0.5: 0.3, 2.0: 0.7, 8.0: 0.0}},
                                           {'R_day': {0.5: 0.5, 2.0: 0.3, 8.0: 0.2}, 'R_night': {0.5: 0.1, 2.0: 0.2, 8.0: 0.7}}),
    ('vent valve stuck 0.02 / 0.5 per year', {'lam_wv': 0.02}, {'lam_wv': 0.5}),
    ('stop duration: shorter / longer', {'D_dist': {0.5: 0.6, 2.0: 0.3, 6.0: 0.08, 24.0: 0.02}}, {'D_dist': {0.5: 0.3, 2.0: 0.3, 6.0: 0.25, 24.0: 0.15}}),
]
rows = []
for name, lo, hi in sens:
    n_lo, v_lo = total_with(lo); n_hi, v_hi = total_with(hi)
    rows.append(dict(parameter=name, n_low=n_lo, n_high=n_hi, v_low=v_lo, v_high=v_hi))
sdf = pd.DataFrame(rows); sdf.to_csv('out/bn_sensitivity_case1.csv', index=False, float_format='%.3f')
# alternative operation: downstream consumes all gas (no flare demands)
prm = copy.deepcopy(bn.PARAMS); inp2 = dict(inp); inp2['demand'] = np.zeros_like(inp['demand'], dtype=float); inp2['fill'] = np.full_like(inp['fill'], 0.5)
df_all = bn.run_year(1, inp2, prm); s_all = bn.summarize(df_all, prm)
json.dump({'base_openings': base_n, 'base_vent': base_v, 'all_consumed_fill50': s_all}, open('out/bn_alt_operation.json', 'w'), indent=1, default=float)
print('base', round(base_n, 2), round(base_v), '| all consumed, fill 50%:', {k: round(float(v), 3) for k, v in s_all['total'].items()})
print(sdf.round(3).to_string(index=False))

# ----------------------------------------------------------------------------- figure 1: DAG
G = nx.DiGraph()
nodes = {
    'Qcls': ('gas production class\n(ADM1)', 0, 3), 'Fcls': ('holder fill class\n(operating model)', 1, 3), 'DN': ('day / night', 2, 3), 'W': ('winter', 3, 3), 'Patm': ('7-day atmospheric ratchet\n(KMA)', 4, 3),
    'I': ('initiating event I\nnone/stop/demand/foam/block/wv', 1.5, 2), 'D': ('stop duration D', 0, 2), 'F': ('flare works F', 3, 2), 'R': ('response time R', 2.5, 1.2), 'B': ('PVRV fouled by foam B', 4, 2),
    'k': ('system stiffness k', 0, 1.2), 'M': ('effective margin M\n(case)', 1, 1.2),
    'Open': ('PVRV open (t_open < end)', 1.5, 0.4), 'Vent': ('vented volume, methane', 3, 0.4), 'Chat': ('chattering cycles', 4.2, 0.4),
}
for k, (lab, x, y) in nodes.items():
    G.add_node(k, label=lab, pos=(x, y))
edges = [('Qcls', 'Open'), ('Fcls', 'Open'), ('DN', 'R'), ('W', 'I'), ('Patm', 'Open'), ('I', 'Open'), ('I', 'k'), ('D', 'Open'), ('F', 'Open'), ('R', 'Open'),
         ('B', 'Open'), ('k', 'Open'), ('M', 'Open'), ('Open', 'Vent'), ('Qcls', 'Vent'), ('R', 'Vent'), ('D', 'Vent'), ('Open', 'Chat'), ('Qcls', 'Chat')]
G.add_edges_from(edges)
pos = {k: v['pos'] for k, v in G.nodes(data=True)}
fig, ax = plt.subplots(figsize=(13, 7)); ax.axis('off'); ax.grid(False)
col = {k: (C[0] if k in ('Qcls', 'Fcls', 'DN', 'W', 'Patm') else C[1] if k in ('I', 'D', 'F', 'R', 'B') else C[2] if k in ('k', 'M') else C[3]) for k in G.nodes}
nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=12, edge_color='#898781', width=0.9, node_size=3800, connectionstyle='arc3,rad=0.08')
for k, (x, y) in pos.items():
    ax.text(x, y, G.nodes[k]['label'], ha='center', va='center', fontsize=8, color=INK,
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=col[k], lw=1.6))
for i, (c, t) in enumerate(((C[0], 'evidence (hourly inputs)'), (C[1], 'operating / fault state'), (C[2], 'physics-derived'), (C[3], 'outcome'))):
    ax.text(0.0 + i * 1.1, 3.75, '■ ' + t, color=c, fontsize=9, fontweight='bold')
ax.set_xlim(-0.6, 4.9); ax.set_ylim(-0.1, 4.0)
ax.set_title('PVRV opening Bayesian network (one-hour slice)', loc='left', color=INK, fontweight='bold')
fig.savefig('fig/bn_dag.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# ----------------------------------------------------------------------------- figure 2: annual profile (Case 1, weekly sums)
df = pd.read_csv('out/bn_hourly_case1.csv.gz')
t = inp['t'].iloc[:len(df)].reset_index(drop=True)
wk = df.set_index(t).resample('W').sum()
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw=dict(hspace=0.4))
bottom = np.zeros(len(wk))
for i, k in enumerate(('demand', 'stop', 'foam', 'block', 'wv')):
    ax[0].bar(wk.index, wk[f'p_{k}'], width=6, bottom=bottom, color=C[i], label=LAB[k]); bottom += wk[f'p_{k}'].to_numpy()
ax[0].set_ylabel('expected openings (per week)'); ax[0].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
ax[0].set_title('Case 1: expected weekly PVRV openings by cause', loc='left', color=INK, fontweight='bold', pad=40)
qw = pd.Series(inp['q'][:len(df)], index=t).resample('W').mean()
fw = pd.Series(inp['fill'][:len(df)], index=t).resample('W').mean()
ax[1].plot(qw.index, qw, color=C[0], label='gas production (m³/h)')
ax[1].plot(fw.index, fw * 100, color=C[1], label='holder fill (%)')
ax[1].set_ylabel('m³/h  |  %'); ax[1].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
fig.savefig('fig/bn_profile_case1.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# ----------------------------------------------------------------------------- figure 3: breakdown by cause (Cases 1-3) + tornado
S = json.load(open('out/bn_summary.json'))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1.3]))
x = np.arange(3); bottom = np.zeros(3)
for i, k in enumerate(('demand', 'stop', 'foam', 'block', 'wv')):
    v = np.array([S[str(c)][k]['openings_per_yr'] for c in (1, 2, 3)])
    ax[0].bar(x, v, bottom=bottom, color=C[i], width=0.55, label=LAB[k]); bottom += v
for xi, b in zip(x, bottom):
    ax[0].text(xi, b + 0.1, f'{b:.1f}', ha='center', fontsize=9, color=INK)
ax[0].set_xticks(x); ax[0].set_xticklabels([f'Case {c}\np_op {bn.ph.P_OP[c]} kPag' for c in (1, 2, 3)])
ax[0].set_ylabel('expected openings (per year)'); ax[0].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7.5)
ax[0].set_title('expected annual openings by cause', loc='left', color=INK, fontweight='bold', pad=46)
sd = sdf.iloc[::-1]
y = np.arange(len(sd))
ax[1].barh(y, sd['n_low'] - base_n, left=base_n, color=C[0], height=0.55, label='lower value')
ax[1].barh(y, sd['n_high'] - base_n, left=base_n, color=C[1], height=0.55, label='upper value')
ax[1].axvline(base_n, color=INK2, lw=0.8); ax[1].set_yticks(y); ax[1].set_yticklabels(sd['parameter'], fontsize=7.5)
ax[1].set_xlabel('expected openings (per year)'); ax[1].legend(loc='lower right', fontsize=7.5)
ax[1].set_title(f'Case 1 sensitivity (base {base_n:.1f} per year)', loc='left', color=INK, fontweight='bold')
fig.savefig('fig/bn_breakdown_tornado.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('figs ok')
