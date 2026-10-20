# -*- coding: utf-8 -*-
"""Summary table and figure of the 100 % operation year runs"""
import json, glob, os
import numpy as np, pandas as pd
os.makedirs('fig', exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 8.5, 'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False,
                     'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#898781', 'ytick.color': '#898781'})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; INK, INK2 = '#0b0b0b', '#52514e'
rows = []
for f in sorted(glob.glob('out/full100_case*_summary.json')):
    s = json.load(open(f)); tag = os.path.basename(f).replace('_summary.json', '')
    df = pd.read_csv(f'out/{tag}_1min.csv.gz', parse_dates=['datetime'])
    ev = pd.read_csv(f'out/{tag}_events.csv', parse_dates=['datetime'])
    prod = df['Q_gen_m3h'].sum() / 60
    opens = ev[ev.code.isin([1, 3])]
    # episode: a gap of more than 10 min between openings starts a new episode
    gaps = opens['t_s'].diff().fillna(1e9); n_epi = int((gaps > 600).sum())
    monthly = df.set_index('datetime')['PV_open_frac'].resample('MS').sum() / 60   # monthly open hours
    rows.append(dict(tag=tag, case=s['case'], pfd=s['pfd'], flare_starts=s['flare_starts'], flare_fails=s['flare_fails'], pv_episodes=n_epi,
                     pv_cycles=s['PV_A_opens'] + s['PV_B_opens'], pv_open_h=(s['PV_A_open_s'] + s['PV_B_open_s']) / 3600, vent_m3=s['PV_vented_m3'],
                     vent_share=s['PV_vented_m3'] / prod, ch4_kg=s['PV_CH4_kg'], h2s_kg=s['PV_H2S_kg'], fill_mean=s['fill_mean'], share_full=s['share_full'],
                     PT_mean=s['PT_mean'], PT_max=s['PT_max'], WV_opens=s['WV_opens'], blower_starts=s['blower_starts'],
                     open_h_month_max=float(monthly.max()), open_h_month_min=float(monthly.min())))
    if s['case'] == 1 and s['pfd'] > 0:
        fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw=dict(hspace=0.45))
        d = df.set_index('datetime')
        wk = d.resample('W').agg({'PV_open_frac': 'sum', 'Q_PV_m3h': 'sum', 'Q_gen_m3h': 'mean', 'V_holder_gas_m3': 'mean'})
        ax[0].bar(wk.index, wk['PV_open_frac'] / 60, width=6, color=C[1]); ax[0].set_ylabel('PVRV open time (h/week)')
        ax[0].set_title('Case 1, 100 % operation (pressure-controlled flare, ignition failure probability 0.05): relief-valve behaviour over one year', loc='left', color=INK, fontweight='bold')
        ax[1].bar(wk.index, wk['Q_PV_m3h'] / 60, width=6, color=C[0]); ax[1].set_ylabel('vented volume (m³/week)')
        ax[2].plot(wk.index, wk['Q_gen_m3h'], color=C[0], label='gas production (m³/h)'); ax[2].plot(wk.index, wk['V_holder_gas_m3'] / 5, color=C[2], label='holder fill (%)')
        ax[2].axhline(60, color=INK2, lw=0.6, ls=':'); ax[2].text(wk.index[0], 60, ' consumer 60', fontsize=7.5, va='bottom', color=INK2)
        ax[2].set_ylabel('m³/h  |  %'); ax[2].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.savefig('fig/full100_case1_year.png', dpi=150, bbox_inches='tight'); plt.close(fig)
r = pd.DataFrame(rows); r.to_csv('out/full100_summary.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250); print(r.round(3).T.to_string())
