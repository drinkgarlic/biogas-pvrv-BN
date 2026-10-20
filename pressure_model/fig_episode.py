# -*- coding: utf-8 -*-
"""Case 1, three flare-start fill levels: a real PVRV opening episode (2020-11-04) at 1 min + ignition failure vs opening table"""
import pandas as pd, numpy as np
import os; os.makedirs('fig', exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 8.5, 'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False,
                     'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#898781', 'ytick.color': '#898781'})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']; INK, INK2 = '#0b0b0b', '#52514e'
t0, t1 = pd.Timestamp('2020-11-04 10:00'), pd.Timestamp('2020-11-04 20:00')
fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, gridspec_kw=dict(hspace=0.5))
for j, fo in enumerate((100, 80, 60)):
    d = pd.read_csv(f'out/onoff_case1_f{fo}_pfd5_1min.csv.gz', parse_dates=['datetime'])
    s = d[(d.datetime >= t0) & (d.datetime <= t1)]; a = ax[j]
    a.fill_between(s.datetime, 0, s['flare_on_frac'] * 100, color=C[3], alpha=0.18, lw=0, label='flare on (on=100)')
    a.plot(s.datetime, s['V_holder_gas_m3'] / 5, color=C[2], lw=1.4, label='holder fill (%)')
    a.plot(s.datetime, s['PT101A_max_kPag'] * 20, color=C[0], lw=0.9, label='PT-101A 1-min max ×20 (kPag)')
    a.plot(s.datetime, s['Q_PV_m3h'], color=C[7], lw=1.0, label='PVRV 1-min mean flow (m³/h)')
    a.plot(s.datetime, s['Q_gen_m3h'], color=INK2, lw=0.8, ls='--', label='gas production (m³/h)')
    a.axhline(60, color=INK2, lw=0.5, ls=':'); a.text(t0, 60, ' consumer 60 m³/h', fontsize=7, va='bottom', color=INK2)
    a.set_ylim(0, 112); a.set_ylabel('%  |  kPag×20  |  m³/h'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=5, fontsize=7.5)
    a.set_title(f'Case 1 (p_op 4.0 kPag), flare start {fo} % — 2020-11-04 10:00~20:00 (1 min)  ·  vented in window {s["Q_PV_m3h"].sum()/60:.1f} m³', loc='left', color=INK, fontweight='bold', pad=22)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.savefig('fig/onoff_case1_episode.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# per ignition failure (holder 100 % + flare off for 2 min or more): did the PVRV open, and the production at that time
K_FULL = 0.0803; rows_all = []
for case, pop in [(1, 4.0), (2, 3.0), (3, 2.0)]:
    d = pd.read_csv(f'out/onoff_case{case}_f100_pfd5_1min.csv.gz', parse_dates=['datetime'])
    fail = (d['V_holder_gas_m3'] >= 499.5) & (d['flare_on_frac'] == 0)
    for i in np.where(fail.astype(int).diff().fillna(0) == 1)[0]:
        j = i
        while j < len(d) and fail.iloc[j]: j += 1
        if j - i < 2: continue
        seg = d.iloc[i:j]
        rows_all.append(dict(case=case, start=d['datetime'].iloc[i], dur_min=j - i, q_gen=seg['Q_gen_m3h'].mean(), surplus=seg['Q_gen_m3h'].mean() - 60,
                             pt_max=seg['PT101A_max_kPag'].max(), opened=bool(seg['PV_open_frac'].sum() > 0), vent_m3=seg['Q_PV_m3h'].sum() / 60,
                             q_thr_30min=60 + (5.05 - pop - 0.11) / (K_FULL * 0.5)))
r = pd.DataFrame(rows_all); r.to_csv('out/onoff_f100_failures.csv', index=False, float_format='%.2f')
print(r.groupby(['case', 'opened']).agg(n=('start', 'size'), q_gen_mean=('q_gen', 'mean'), q_gen_min=('q_gen', 'min'), q_gen_max=('q_gen', 'max'), pt_max=('pt_max', 'max')).round(2))
