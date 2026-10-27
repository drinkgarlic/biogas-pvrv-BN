# -*- coding: utf-8 -*-
"""on/off 9-case BN figures: breakdown by initiating event, variant sensitivity, comparison with the 1-s pressure model, Case 1 weekly profile"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 8.5, 'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False,
                     'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#898781', 'ytick.color': '#898781'})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']; INK, INK2 = '#0b0b0b', '#52514e'
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE); os.makedirs('fig', exist_ok=True)
r = pd.read_csv('out/bn_onoff_9cases.csv')
INIT = [('open_demand', 'ignition failure on flare demand'), ('open_stop', 'downstream stop'), ('open_block', 'condensate / freezing / XV blockage'), ('open_foam', 'foam blockage'), ('open_wv', 'vent valve stuck')]
ICOL = [C[1], C[0], C[6], C[2], C[3]]
labels = [f'Case {c}\n{int(f*100)} %' for f in (1.0, 0.8, 0.6) for c in (1, 2, 3)]
order = [(c, f) for f in (1.0, 0.8, 0.6) for c in (1, 2, 3)]

def stacked(ax, sub, ylim=None, fmt='{:.1f}'):
    x = np.arange(len(sub)); bottom = np.zeros(len(sub))
    for (col, name), colr in zip(INIT, ICOL):
        v = sub[col].to_numpy(); ax.bar(x, v, 0.6, bottom=bottom, color=colr, label=name); bottom += v
    for xi, t in zip(x, bottom): ax.text(xi, t + (ylim or bottom.max()) * 0.015, fmt.format(t), ha='center', fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f'Case {c}\n{int(f*100)} %' for c, f in zip(sub['case'], sub['f_on'])])
    if ylim: ax.set_ylim(0, ylim)

main = pd.concat([r[(r.variant == 'retry30') & (r.case == c) & (r.f_on == f)] for c, f in order])
fig = plt.figure(figsize=(14, 9)); gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.25, height_ratios=[1.1, 1])
a = fig.add_subplot(gs[0, :]); stacked(a, main)
a.set_ylabel('PVRV openings (per year)'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7.5)
a.set_title('on/off operation, 9-case BN (v4.3): expected annual openings by initiating event — automatic retry 30 min, ignition failure 0.05, day/night operator response', loc='left', color=INK, fontweight='bold', pad=24)
a.axvspan(2.5, 8.5, color=C[2], alpha=0.05, lw=0); a.text(5.5, main.openings.max() * 0.8, 'flare start 80 % / 60 %: demand and downstream-stop branches = 0, residual 1.2~1.3 per year', ha='center', fontsize=8, color=INK2)
b = fig.add_subplot(gs[1, 0]); stacked(b, main[main.f_on < 1.0], ylim=1.6, fmt='{:.2f}')
b.set_ylabel('per year'); b.set_title('flare start 80 / 60 % zoomed: residual initiating events unrelated to the flare', loc='left', color=INK, fontweight='bold')
c_ = fig.add_subplot(gs[1, 1]); x = np.arange(9); w = 0.26
for j, (var, name, colr) in enumerate([('retry30', 'automatic retry 30 min (base)', C[0]), ('hard30', '30 % of failures are hard faults', C[3]), ('manual', 'no automatic retry (v3 style)', C[7])]):
    sub = pd.concat([r[(r.variant == var) & (r.case == c) & (r.f_on == f)] for c, f in order])
    c_.bar(x + (j - 1) * w, sub.openings.to_numpy(), w, color=colr, label=name)
c_.set_yscale('log'); c_.set_ylim(0.5, 200); c_.set_xticks(x); c_.set_xticklabels(labels, fontsize=7.5); c_.set_ylabel('per year (log)')
c_.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=1, fontsize=7.5); c_.set_title('flare recovery model sensitivity', loc='left', color=INK, fontweight='bold', pad=40)
fig.savefig('fig/bn_onoff_9cases.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# --- BN demand branch vs 1-s pressure model (ignition failure -> opening episodes)
_pm_path = os.path.join(HERE, '..', 'pressure_model', 'out', 'onoff_summary.csv')          # produced by pressure_model/sum_onoff.py
if not os.path.exists(_pm_path): _pm_path = os.path.join(HERE, '..', 'results', 'pressure_model', 'onoff_summary.csv')   # shipped reference results
pm = pd.read_csv(_pm_path); pm = pm[pm.pfd > 0]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.28))
bnv = [float(main[(main.case == c) & (main.f_on == f)].open_demand.iloc[0]) for c, f in order]
pmv = [float(pm[(pm.case == c) & (np.isclose(pm.f_on, f))].pv_episodes.iloc[0]) for c, f in order]
bnx = [float(main[(main.case == c) & (main.f_on == f)].vent_demand.iloc[0]) for c, f in order]
pmx = [float(pm[(pm.case == c) & (np.isclose(pm.f_on, f))].vent_m3.iloc[0]) for c, f in order]
# BN without the operator response (automatic retries only) = the condition of the 1-s model; same input series (v4.3)
import copy, pv_bn as _bnv
nov, nox = [], []
for c, f in order:
    if f < 1.0: nov.append(0.0); nox.append(0.0); continue
    prm = copy.deepcopy(_bnv.PARAMS); prm['f_on']['v'] = f; prm['t_retry_h']['v'] = 0.5; prm['n_retry']['v'] = 4; prm['p_hard']['v'] = 0.0; prm['pressure_backup']['v'] = False
    tag = f'onoff_c{c}_f{int(round(f * 100))}'; prm['fill_series']['v'] = f'fill_hourly_{tag}.csv'; prm['demand_series']['v'] = f'flare_demands_{tag}.csv'
    prm['R_day']['v'] = {2.5: 1.0}; prm['R_night']['v'] = {2.5: 1.0}
    d_ = _bnv.run_year(c, _bnv.load_inputs(prm=prm), prm); nov.append(float(d_.p_demand.sum())); nox.append(float(d_.v_demand.sum()))
print('BN demand branch without operator (100 %):', [round(v, 2) for v in nov[:3]], 'per year,', [round(v) for v in nox[:3]], 'm3')
w = 0.27
for a, bv, nv, pv, t, yl in [(ax[0], bnv, nov, pmv, '(a) openings caused by ignition failures', 'per year'), (ax[1], bnx, nox, pmx, '(b) vented volume of those openings', 'm³/year')]:
    a.bar(x - w, bv, w, color=C[0], label='BN demand branch, with operator response (expected)'); a.bar(x, nv, w, color='#9cc3ee', label='BN, retries only (expected)')
    a.bar(x + w, pv, w, color=C[1], label='1-s pressure model (one random realisation)')
    top = max(bv + nv + pv)
    for xs, vs, col, integer in ((x - w, bv, C[0], False), (x, nv, '#5f8fc4', False), (x + w, pv, C[1], True)):
        for xi, v in zip(xs, vs):
            if v >= 0.05: a.text(xi, v + top * 0.012, f'{v:.0f}' if (v >= 10 or integer) else f'{v:.1f}', ha='center', fontsize=6.5, color=col)
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=7.5); a.set_ylabel(yl)
    a.set_title(t, loc='left', color=INK, fontweight='bold', pad=8)
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=8)
fig.suptitle('BN demand branch vs 1-s pressure model, both run on the same 1-s input series (v4.3)', x=0.01, y=1.06, ha='left', color=INK, fontweight='bold')
fig.savefig('fig/bn_vs_pressure_onoff.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# --- Case 1 weekly profile (three flare-start fill levels), stacked by initiating event
import pv_bn as _bn
t0 = pd.read_csv(_bn.ADM1_CSV, parse_dates=['datetime_interval_end'])['datetime_interval_end']
fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, gridspec_kw=dict(hspace=0.5))
for j, fo in enumerate((100, 80, 60)):
    d = pd.read_csv(f'out/bn_hourly_case1_f{fo}_retry30.csv.gz'); d.index = t0.iloc[:len(d)].to_numpy()
    wk = d[['p_demand', 'p_stop', 'p_block', 'p_foam', 'p_wv']].resample('W').sum(); bottom = np.zeros(len(wk))
    for (col, name), colr in zip(INIT, ICOL):
        v = wk[col.replace('open_', 'p_')].to_numpy(); ax[j].bar(wk.index, v, width=6, bottom=bottom, color=colr, label=name); bottom += v
    q = d['q'].resample('W').mean(); a2 = ax[j].twinx(); a2.plot(q.index, q, color=INK2, lw=0.8, ls='--'); a2.set_ylim(0, 130); a2.set_ylabel('gas production (m³/h, dashed)', color=INK2); a2.grid(False)
    a2.axhline(60 + 0.94 / (0.0803 * 0.5) if fo == 100 else np.nan, color=C[1], lw=0.6, ls=':')
    tot = d['p_open'].sum()
    ax[j].set_ylabel('per week'); ax[j].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7.5)
    ax[j].set_title(f'Case 1 (p_op 4.0 kPag), flare start {fo} % — expected weekly openings, annual total {tot:.1f}' + ('  (dotted horizontal line: 30-min opening threshold 83.4 m³/h)' if fo == 100 else ''), loc='left', color=INK, fontweight='bold', pad=22)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.savefig('fig/bn_onoff_case1_weekly.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('figs ok')
