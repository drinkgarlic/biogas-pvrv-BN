# -*- coding: utf-8 -*-
"""Summary table and figures of the on/off operation 9 cases (+3 controls)"""
import json, glob, os
import numpy as np, pandas as pd
os.makedirs('fig', exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 8.5, 'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False,
                     'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#898781', 'ytick.color': '#898781'})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']; INK, INK2 = '#0b0b0b', '#52514e'
rows = []; dfs = {}
for f in sorted(glob.glob('out/onoff_case*_summary.json')):
    s = json.load(open(f)); tag = os.path.basename(f).replace('_summary.json', '')
    df = pd.read_csv(f'out/{tag}_1min.csv.gz', parse_dates=['datetime']); ev = pd.read_csv(f'out/{tag}_events.csv', parse_dates=['datetime'])
    opens = ev[ev.code.isin([1, 3])]; gaps = opens['t_s'].diff().fillna(1e9); n_epi = int((gaps > 600).sum())
    rows.append(dict(tag=tag, case=s['case'], f_on=s['f_on'], pfd=s['pfd'], fill_mean=s['fill_mean'], fill_p5=s['fill_p5'], fill_p95=s['fill_p95'],
                     share_full=s['share_full'], share_empty=s['share_empty'], use_stops=s['use_stops'], use_on_share=s['use_on_share'],
                     flare_starts=s['flare_starts'], flare_fails=s['flare_fails'], flare_on_share=s['flare_on_share'],
                     pv_episodes=n_epi, pv_cycles=s['PV_A_opens'] + s['PV_B_opens'], pv_open_h=(s['PV_A_open_s'] + s['PV_B_open_s']) / 3600,
                     vent_m3=s['PV_vented_m3'], vent_share=s['PV_vented_m3'] / s['prod_m3'], ch4_kg=s['PV_CH4_kg'], h2s_kg=s['PV_H2S_kg'],
                     PT_mean=s['PT_mean'], PT_max=s['PT_max'], PT_min=s['PT_min'], WV_opens=s['WV_opens'], blower_starts=s['blower_starts'],
                     WV_air=s['WV_vented_air_m3'], blower_air=s['blower_air_m3']))
    dfs[tag] = df
r = pd.DataFrame(rows).sort_values(['pfd', 'case', 'f_on'], ascending=[False, True, False]); r.to_csv('out/onoff_summary.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250); print(r.round(3).to_string(index=False))

# figure 1: 9-case fill time series (daily mean and min/max) 3x3
fig, ax = plt.subplots(3, 3, figsize=(15, 10.5), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.5, wspace=0.12))
for i, case in enumerate((1, 2, 3)):
    for j, fo in enumerate((1.0, 0.8, 0.6)):
        tag = f'onoff_case{case}_f{int(round(fo*100))}_pfd5'
        if tag not in dfs: continue
        d = dfs[tag].set_index('datetime'); fl = d['V_holder_gas_m3'] / 5
        w = fl.resample('D')
        a = ax[i, j]; a.fill_between(w.min().index, w.min(), w.max(), color=C[2], alpha=0.3, lw=0); a.plot(w.mean().index, w.mean(), color=C[2], lw=0.9)
        pv = d['PV_open_frac'].resample('D').sum() / 60
        a2 = a.twinx(); a2.bar(pv.index, pv, width=1, color=C[7], alpha=0.8); a2.set_ylim(0, 6); a2.grid(False); a2.spines['right'].set_visible(True)
        if j == 2: a2.set_ylabel('PVRV open (h/day)', color=C[7])
        else: a2.set_yticklabels([])
        rr = r[r.tag == tag].iloc[0]
        a.set_title(f'Case {case} ({4-i}.0 kPag), start {int(fo*100)} %\n{rr.pv_episodes:.0f} openings, {rr.vent_m3:,.0f} m³/yr', loc='left', fontsize=8, color=INK, fontweight='bold')
        a.set_ylim(0, 105)
        if j == 0: a.set_ylabel('holder fill (%)')
        a.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
fig.suptitle('on/off operation, 9 cases: holder fill (daily range and mean) and PVRV open time (ignition failure 0.05, 30-min retry)', x=0.01, ha='left', color=INK, fontweight='bold')
fig.savefig('fig/onoff_9cases_fill_pv.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# figure 2: one-day detail (Case 1, three flare-start fill levels, 2021-03-10, 1 s)
fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw=dict(hspace=0.4))
for j, fo in enumerate((1.0, 0.8, 0.6)):
    tag = f'onoff_case1_f{int(round(fo*100))}_pfd5'
    try: sec = pd.read_csv(f'out/{tag}_1s_days.csv.gz', parse_dates=['datetime'])
    except FileNotFoundError: continue
    s = sec[sec['datetime'].dt.strftime('%Y-%m-%d') == '2021-03-10']
    a = ax[j]; a.plot(s['datetime'], s['V_holder_gas_m3'] / 5, color=C[2], label='holder fill (%)')
    a.plot(s['datetime'], s['PT101A_kPag'] * 20, color=C[0], lw=0.8, label='PT-101A ×20 (kPag)')
    a.plot(s['datetime'], s['Q_PV_m3h'] / 2, color=C[7], lw=0.8, label='PVRV flow ÷2 (m³/h)')
    a.set_ylim(0, 110); a.set_ylabel('%  |  kPag×20  |  m³/h÷2'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7.5)
    a.set_title(f'Case 1, flare start {int(fo*100)} % — 2021-03-10 (1 s)', loc='left', color=INK, fontweight='bold', pad=22)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.savefig('fig/onoff_case1_day.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# figure 3: 9-case bars (opening episodes, vented volume)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw=dict(wspace=0.3))
sub = r[r.pfd > 0]; x = np.arange(3); wdt = 0.26
for j, fo in enumerate((1.0, 0.8, 0.6)):
    v = [sub[(sub.case == c) & (sub.f_on == fo)].pv_episodes.values[0] if len(sub[(sub.case == c) & (sub.f_on == fo)]) else 0 for c in (1, 2, 3)]
    ax[0].bar(x + (j - 1) * wdt, v, wdt, color=C[j], label=f'start {int(fo*100)} %')
    for xi, vi in zip(x + (j - 1) * wdt, v): ax[0].text(xi, vi + 5, f'{vi:.0f}', ha='center', fontsize=7.5)
    v = [sub[(sub.case == c) & (sub.f_on == fo)].vent_m3.values[0] if len(sub[(sub.case == c) & (sub.f_on == fo)]) else 0 for c in (1, 2, 3)]
    ax[1].bar(x + (j - 1) * wdt, v, wdt, color=C[j], label=f'start {int(fo*100)} %')
    for xi, vi in zip(x + (j - 1) * wdt, v): ax[1].text(xi, vi * 1.02 + 20, f'{vi:,.0f}', ha='center', fontsize=7.5)
for a, t in zip(ax, ('PVRV opening episodes (per year)', 'PVRV vented volume (m³/year)')):
    a.set_xticks(x); a.set_xticklabels([f'Case {c}\np_op {5-c}.0 kPag' for c in (1, 2, 3)]); a.set_title(t, loc='left', color=INK, fontweight='bold', pad=24)
    a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7.5)
fig.savefig('fig/onoff_9cases_bars.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('figs ok')
