# -*- coding: utf-8 -*-
"""Comparison of buffer operating bands (flare on/off fill levels) + optional-node sensitivity (indicator error, stuck PVRV, reseat failure) + state-probability figures"""
import sys, os, copy, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_bn as bn, physics as ph
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 8.5, 'axes.grid': True, 'grid.color': '#e1e0d9',
                     'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False,
                     'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#898781', 'ytick.color': '#898781'})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']; INK, INK2 = '#0b0b0b', '#52514e'

inp = bn.load_inputs()
d = pd.read_csv(bn.ADM1_CSV); qh = d['Q_gas_wet_m3_d_20C'].to_numpy() / 24
rows = []
for on, off in [(0.85, 0.70), (0.75, 0.60), (0.65, 0.50), (0.55, 0.40)]:
    t, fills, starts, st = ph.operating_year(qh, on=on, off=off)
    fh = fills.reshape(-1, 60).mean(1)
    inp2 = dict(inp); inp2['fill'] = np.concatenate([fh, [fh[-1]]])
    dem = np.zeros(len(inp['q'])); np.add.at(dem, np.minimum((starts // 3600).astype(int), len(dem) - 1), 1.0); inp2['demand'] = dem
    # the fill at the flare demand is the on value, so the demand branch uses on instead of 0.85 through the bias parameter
    prm = copy.deepcopy(bn.PARAMS); prm['level_bias']['v'] = {on - 0.85: 1.0}
    for case in (1, 3):
        df = bn.run_year(case, inp2, prm); s = bn.summarize(df, prm)
        rows.append(dict(band=f'{int(off*100)}~{int(on*100)} %', case=case, flare_starts=st['flare_starts'], flare_share=st['flared_m3'] / st['production_m3'],
                         derate_h=st['derate_h'], aux_m3=st['aux_m3'], fill_mean=float(fh.mean()), fill_p90=float(np.percentile(fh, 90)),
                         openings=s['total']['openings_per_yr'], demand_branch=s['demand']['openings_per_yr'], vent_m3=s['total']['vent_m3_per_yr']))
bb = pd.DataFrame(rows); bb.to_csv('out/buffer_band.csv', index=False, float_format='%.3f'); print(bb.round(3).to_string(index=False))

# optional-node sensitivity (Case 1, base band)
sens = []
for name, mod in [('base', {}), ('indicator error ±5 %', {'level_bias': {-0.05: 0.25, 0.0: 0.5, 0.05: 0.25}}), ('indicator reads 5 % low', {'level_bias': {0.05: 1.0}}),
                  ('PVRV stuck 0.01', {'p_pv_stuck': 0.01}), ('reseat failure 0.05', {'p_reseat': 0.05})]:
    prm = copy.deepcopy(bn.PARAMS)
    for k, v in mod.items(): prm[k]['v'] = v
    df = bn.run_year(1, inp, prm); s = bn.summarize(df, prm)['total']
    sens.append(dict(case=name, openings=s['openings_per_yr'], vent=s['vent_m3_per_yr'], stuck_overpressure=s['pv_stuck_overpressure_per_yr'],
                     reseat=s['reseat_fail_per_yr'], reseat_leak=s['reseat_leak_m3_per_yr']))
sd = pd.DataFrame(sens); sd.to_csv('out/bn_optional_nodes.csv', index=False, float_format='%.4f'); print(sd.round(4).to_string(index=False))

# state-probability figure (Case 1): start-hour attribution vs probability of being open at that hour
df = pd.read_csv('out/bn_hourly_case1.csv.gz'); t = inp['t'].iloc[:len(df)].reset_index(drop=True)
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw=dict(hspace=0.45))
w = pd.DataFrame({'init': df['p_open'].to_numpy(), 'state': df['p_state_open'].to_numpy()}, index=t)
ww = w.resample('W').sum()
ax[0].bar(ww.index, ww['init'], width=6, color=C[0], label='(2) start-hour attribution: sum of the probabilities that events starting in that week eventually open the valve (per week)')
ax[0].set_ylabel('per week'); ax[0].legend(loc='lower left', bbox_to_anchor=(0, 1.0))
ax[0].set_title('Case 1: two kinds of hourly probability', loc='left', color=INK, fontweight='bold', pad=26)
ax[1].bar(ww.index, ww['state'], width=6, color=C[1], label='(3) state probability: probability of being open at that hour summed over the week = expected open time (h/week)')
ax[1].set_ylabel('h/week'); ax[1].legend(loc='lower left', bbox_to_anchor=(0, 1.0))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.savefig('fig/bn_state_vs_init.png', dpi=150, bbox_inches='tight'); plt.close(fig)
# distribution within the day (day / night)
hod = w.groupby(w.index.hour).mean()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(hod.index, hod['init'] * 1000, color=C[0], marker='o', ms=3, label='(2) start-hour attribution (x 1/1000)')
ax.plot(hod.index, hod['state'] * 1000, color=C[1], marker='o', ms=3, label='(3) state probability (x 1/1000)')
ax.set_xlabel('hour of day'); ax.set_ylabel('hourly probability (x 1/1000)'); ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
ax.set_title('Distribution within the day: delayed night response pushes the state probability into the early morning', loc='left', color=INK, fontweight='bold', pad=22)
fig.savefig('fig/bn_state_vs_init_hour.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('figs ok; state prob max', round(df['p_state_open'].max(), 4), 'mean', round(df['p_state_open'].mean(), 5))
