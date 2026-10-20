# -*- coding: utf-8 -*-
"""Figures and summary tables of the base operation and the downstream-stop scan"""
import json, glob, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pressure_model as pm

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False, 'font.size': 8.5,
                     'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e', 'xtick.color': '#898781', 'ytick.color': '#898781',
                     'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'legend.frameon': False, 'lines.linewidth': 1.2})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']   # category colours (fixed order)
INK, INK2 = '#0b0b0b', '#52514e'
OUT = 'out'; FIG = 'fig'; os.makedirs(FIG, exist_ok=True)
CASES = pm.CASES

def load_case(c):
    df = pd.read_csv(f'{OUT}/base_case{c}_1min.csv.gz', parse_dates=['datetime'])
    ev = pd.read_csv(f'{OUT}/base_case{c}_events.csv', parse_dates=['datetime'])
    sec = pd.read_csv(f'{OUT}/base_case{c}_1s_selected_days.csv.gz', parse_dates=['datetime'])
    sm = json.load(open(f'{OUT}/base_case{c}_summary.json'))
    return df, ev, sec, sm

def fig_year(c, df, ev):
    d10 = df.set_index('datetime').resample('10min').mean(numeric_only=True)
    dmax = df.set_index('datetime')['PT101A_max_kPag'].resample('10min').max()
    dmin = df.set_index('datetime')['PT101A_min_kPag'].resample('10min').min()
    daily = ev.set_index('datetime').groupby([pd.Grouper(freq='D'), 'event']).size().unstack(fill_value=0)
    p_op = CASES[c]
    fig, ax = plt.subplots(6, 1, figsize=(13, 17), sharex=True, gridspec_kw=dict(hspace=0.45))
    a = ax[0]
    a.plot(d10.index, d10['p_atm_kPa'], color=C[0], label='atmospheric pressure (KMA station pressure)')
    a.plot(d10.index, d10['p_N1A_abs_kPa'], color=C[1], label='digester A absolute pressure (nozzle N1)')
    a.plot(d10.index, d10['p_PVset_abs_kPa'], color=C[2], label='PVRV set pressure seen as absolute (atmosphere + 5.0)')
    a.set_ylabel('absolute pressure (kPa)'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
    a.set_title(f'Case {c} (p_op {p_op} kPag) — one-year pressure behaviour, downstream consumes all gas', loc='left', color=INK, fontweight='bold', pad=22)
    a = ax[1]
    a.fill_between(d10.index, dmin, dmax, color=C[0], alpha=0.25, lw=0, label='1-min min~max')
    a.plot(d10.index, d10['PT101A_kPag'], color=C[0], lw=0.8, label='PT-101A (10-min mean)')
    a.axhline(5.0, color=C[7], lw=0.8, ls='--'); a.text(d10.index[0], 5.0, ' PVRV setting 5.0', va='bottom', fontsize=7.5, color=C[7])
    a.axhline(p_op, color=INK2, lw=0.6, ls=':'); a.axhline(p_op - 0.3, color=INK2, lw=0.6, ls=':')
    a.text(d10.index[0], p_op - 0.3, f' blower start {p_op - 0.3:.1f} ~ vent valve {p_op:.1f}', va='top', fontsize=7.5, color=INK2)
    a.set_ylabel('PT-101A gauge (kPag)'); a.set_ylim(p_op - 0.5, 5.3); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    a = ax[2]
    hg = df.set_index('datetime')['holder_gas_kPag']
    a.fill_between(hg.resample('D').min().index, hg.resample('D').min(), hg.resample('D').max(), color=C[0], alpha=0.25, lw=0, label='daily min~max')
    a.plot(hg.resample('D').mean().index, hg.resample('D').mean(), color=C[0], lw=1.0, label='daily mean')
    a.set_ylabel('holder gas-side gauge (kPag)'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    a.text(0.995, 0.04, 'holder gas side = air side (membrane differential 0). Within a day it moves between the blower start (−0.3) and the vent valve (+1 %)', transform=a.transAxes, ha='right', va='bottom', fontsize=7.5, color=INK2)
    a = ax[3]
    a.plot(d10.index, d10['V_holder_gas_m3'] / 5.0, color=C[2])
    a.set_ylabel('holder fill (%)')
    a = ax[4]
    bs = daily.get('blower start', pd.Series(dtype=float)); wv = daily.get('WV open', pd.Series(dtype=float))
    a.bar(bs.index, bs.values, width=1.0, color=C[0], label='blower starts (per day)')
    a.bar(wv.index, -wv.values, width=1.0, color=C[1], label='vent valve openings (per day, lower)')
    a.axhline(0, color='#c3c2b7', lw=0.6); a.set_ylabel('per day'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    a = ax[5]
    a.plot(d10.index, d10['Q_gen_m3h'], color=C[0], lw=0.8, label='gas production (both digesters)')
    a.plot(d10.index, d10['rho_gas_kgm3'] * 100, color=C[1], lw=0.8, label='pipe gas density ×100 (kg/m³, composition and pressure)')
    a.set_ylabel('m³/h  |  kg/m³ ×100'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.savefig(f'{FIG}/base_case{c}_year.png', dpi=150, bbox_inches='tight'); plt.close(fig)

def fig_day(c, sec, day, hours=None, tag=''):
    s = sec[sec['datetime'].dt.strftime('%Y-%m-%d') == day].copy()
    if hours is not None:
        s = s[(s['datetime'].dt.hour >= hours[0]) & (s['datetime'].dt.hour < hours[1])]
    if s.empty:
        return
    p_op = CASES[c]
    fig, ax = plt.subplots(5, 1, figsize=(13, 14), sharex=True, gridspec_kw=dict(hspace=0.42))
    a = ax[0]; a.plot(s['datetime'], s['p_atm_kPa'], color=C[0]); a.set_ylabel('atmospheric pressure (kPa)')
    a.set_title(f'Case {c} — {day} 1-s behaviour' + (f' ({hours[0]}~{hours[1]} h)' if hours else ''), loc='left', color=INK, fontweight='bold', pad=4)
    a = ax[1]
    a.plot(s['datetime'], s['PT101A_kPag'], color=C[0], lw=0.9, label='PT-101A')
    a.plot(s['datetime'], s['holder_gas_kPag'], color=C[1], lw=0.9, label='holder gas side')
    a.axhline(p_op, color=INK2, lw=0.6, ls=':'); a.axhline(p_op - 0.3, color=INK2, lw=0.6, ls=':')
    a.set_ylabel('gauge (kPag)'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    a = ax[2]
    a.plot(s['datetime'], s['p_N1A_abs_kPa'], color=C[1], label='digester A absolute')
    a.plot(s['datetime'], s['p_PVset_abs_kPa'], color=C[2], label='PVRV setting (absolute)')
    a.set_ylabel('absolute (kPa)'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    a = ax[3]
    a.plot(s['datetime'], s['Q_pipeA_m3h'], color=C[0], lw=0.8, label='branch A pipe flow (+ towards holder)')
    a.plot(s['datetime'], s['Q_gen_m3h'] / 2, color=INK2, lw=0.6, ls='--', label='digester A production')
    a.axhline(0, color='#c3c2b7', lw=0.6); a.set_ylabel('m³/h'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    a = ax[4]
    a.fill_between(s['datetime'], 0, s['blower_on'] * 400, color=C[0], alpha=0.5, lw=0, label='blower (400 m³/h)')
    a.plot(s['datetime'], s['Q_WV_m3h'], color=C[1], lw=0.9, label='vent valve discharge (m³/h)')
    a.set_ylabel('air m³/h'); a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.savefig(f'{FIG}/base_case{c}_{day}{tag}.png', dpi=150, bbox_inches='tight'); plt.close(fig)

def fig_cases_pt(dfs):
    fig, ax = plt.subplots(1, 1, figsize=(13, 4))
    for i, (c, df) in enumerate(dfs.items()):
        d = df.set_index('datetime')['PT101A_kPag'].resample('1h').mean()
        ax.plot(d.index, 5.0 - d, color=C[i], lw=0.9, label=f'Case {c} (p_op {CASES[c]} kPag)')
    ax.set_ylabel('margin to the PVRV = 5.0 − PT-101A (kPa)'); ax.legend(loc='upper right', ncol=3)
    ax.set_title('margin to the PVRV set pressure in base operation (1-h mean)', loc='left', color=INK, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.savefig(f'{FIG}/base_margin_cases.png', dpi=150, bbox_inches='tight'); plt.close(fig)

def fig_scan():
    rows = []
    fig, ax = plt.subplots(3, 1, figsize=(13, 12), sharex=True, gridspec_kw=dict(hspace=0.5))
    for i, c in enumerate((1, 2, 3)):
        for j, f in enumerate((50, 85)):
            fn = f'{OUT}/stop_case{c}_f{f}.csv.gz'
            if not os.path.exists(fn):
                continue
            s = pd.read_csv(fn, parse_dates=['start'])
            ls = '-' if f == 85 else ':'
            ax[0].plot(s['start'], s['t_first_open_min'], color=C[i], ls=ls, lw=0.8, label=f'Case {c}, fill {f} %')
            if f == 85:
                ax[1].plot(s['start'], s['opens_1h_after_open'], color=C[i], lw=0.8, label=f'Case {c}')
                ax[2].plot(s['start'], s['vent_1h_after_open_m3'], color=C[i], lw=0.8, label=f'Case {c}')
            d = s.describe()
            rows.append(dict(case=c, f0=f, n=len(s), t_full_min=d.loc['50%', 't_holder_full_min'], t_open_min=d.loc['50%', 't_first_open_min'],
                             t_open_min_min=d.loc['min', 't_first_open_min'], t_open_max=d.loc['max', 't_first_open_min'],
                             never_open_6h=int(s['t_first_open_min'].isna().sum()),
                             opens_1h=d.loc['50%', 'opens_1h_after_open'], open_s_1h=d.loc['50%', 'open_s_1h_after_open'],
                             vent_1h=d.loc['50%', 'vent_1h_after_open_m3'], vent_1h_max=d.loc['max', 'vent_1h_after_open_m3'],
                             opens_2h=d.loc['50%', 'opens_2h_after_open'], vent_2h=d.loc['50%', 'vent_2h_after_open_m3'],
                             opens_6h=d.loc['50%', 'opens_6h'], vent_6h=d.loc['50%', 'vent_6h_m3'], open_s_6h=d.loc['50%', 'open_s_6h'],
                             PT_max=d.loc['max', 'PT_max_6h']))
    ax[0].set_ylabel('time to first opening (min)'); ax[0].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
    ax[0].set_title('downstream-stop events (started at every full hour, 8,754 events): time to the first PVRV opening', loc='left', color=INK, fontweight='bold', pad=40)
    ax[1].set_ylabel('openings within 1 h after the first opening'); ax[1].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
    ax[1].set_title('fill 85 %: number of openings during the first hour after the first opening (chattering)', loc='left', color=INK, fontweight='bold', pad=26)
    ax[2].set_ylabel('m³ (20 °C basis)'); ax[2].legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3)
    ax[2].set_title('fill 85 %: gas vented through the PVRV during the first hour after the first opening', loc='left', color=INK, fontweight='bold', pad=26)
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.savefig(f'{FIG}/stop_scan.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    return pd.DataFrame(rows)

def fig_chatter():
    """Case 1, event starting 2020-11-01 00:00 reproduced with the scalar kernel; minimum stable opening 0 / 10 / 25 % compared (1 s)"""
    inp = pm.load_inputs()
    fig, ax = plt.subplots(6, 1, figsize=(13, 14), sharex=True, gridspec_kw=dict(hspace=0.45, height_ratios=[1, 0.7] * 3))
    for i, fmin in enumerate((0.0, 0.10, 0.25)):
        P = dict(pm.BASE); P['p_op'] = 4.0; P['out_mode'] = 'stop'; P['fill0'] = 0.85; P['f_min_PV'] = fmin
        df, ev, sec, sm = pm.run_year(inp, P, sec_days=[0], n_sec=4500, verbose=False)
        s = sec.iloc[3180:3900]                      # 00:53:00 ~ 01:05:00
        a = ax[2 * i]
        a.plot(s['datetime'], s['PT101A_kPag'], color=C[0], lw=0.9, label='PT-101A')
        a.axhline(5.0, color=C[7], lw=0.7, ls='--'); a.axhline(5.05, color=C[7], lw=0.5, ls=':')
        a.text(s['datetime'].iloc[-1], 5.0, 'setting 5.0 ', ha='right', va='top', fontsize=7.5, color=C[7])
        a.text(s['datetime'].iloc[-1], 5.05, 'opens 5.05 (stiction 1 %) ', ha='right', va='bottom', fontsize=7.5, color=C[7])
        a.set_ylim(4.93, 5.09); a.set_ylabel('kPag')
        n45 = sm["PV_A_opens"]; o45 = sm["PV_A_open_s"]; v45 = sm["PV_vented_m3"]
        a.set_title(f'minimum stable opening {int(fmin * 100)} %  —  openings in the 20 min after the first opening: {n45}, cumulative open {o45:.0f} s, vented {v45:.1f} m³',
                    loc='left', color=INK, fontsize=9)
        a = ax[2 * i + 1]
        a.plot(s['datetime'], s['Q_PV_m3h'], color=C[1], lw=0.8, label='PVRV flow')
        a.plot(s['datetime'], s['Q_gen_m3h'], color=INK2, lw=0.6, ls='--', label='gas production (both digesters)')
        a.set_ylabel('m³/h'); a.legend(loc='upper right', ncol=2, fontsize=7.5)
    ax[0].text(0.5, 1.25, 'Case 1, downstream stop at 85 % fill (start 2020-11-01 00:00): PVRV cycling after the holder is full (1 s, 00:53~01:05)',
               transform=ax[0].transAxes, ha='center', color=INK, fontweight='bold')
    ax[5].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.savefig(f'{FIG}/stop_chatter_fmin.png', dpi=150, bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    dfs = {}; summ = []
    for c in (1, 2, 3):
        if not os.path.exists(f'{OUT}/base_case{c}_summary.json'):
            continue
        df, ev, sec, sm = load_case(c); dfs[c] = df
        fig_year(c, df, ev)
        fig_day(c, sec, '2021-04-12'); fig_day(c, sec, '2021-08-24', hours=(0, 4), tag='_0-4h')
        if c == 1:
            fig_day(c, sec, '2021-08-24', hours=(4, 7), tag='_4-7h_blower')
        st = df['PT101A_kPag']
        summ.append(dict(case=c, p_op=CASES[c], PT_mean=st.mean(), PT_min=df['PT101A_min_kPag'].min(), PT_max=df['PT101A_max_kPag'].max(),
                         PT_p99=df['PT101A_max_kPag'].quantile(0.99), margin_min=5.0 - df['PT101A_max_kPag'].max(),
                         holder_gas_min=df['holder_gas_kPag'].min(), holder_gas_max=df['holder_gas_kPag'].max(),
                         fill_min=df['V_holder_gas_m3'].min() / 5, fill_max=df['V_holder_gas_m3'].max() / 5,
                         N1_abs_min=df['p_N1A_abs_kPa'].min(), N1_abs_max=df['p_N1A_abs_kPa'].max(),
                         set_abs_min=df['p_PVset_abs_kPa'].min(), set_abs_max=df['p_PVset_abs_kPa'].max(),
                         dp_fric_mean=df['dp_fric_A_Pa'].mean(), dp_fric_max=df['dp_fric_A_Pa'].max(), dp_head_mean=df['dp_head_PT_Pa'].mean(),
                         **{k: sm[k] for k in ('PV_A_opens', 'PV_B_opens', 'PV_A_open_s', 'PV_vented_m3', 'blower_starts', 'blower_run_s',
                                               'blower_air_m3', 'WV_opens', 'WV_open_s', 'WV_vented_air_m3')}))
    if dfs:
        fig_cases_pt(dfs)
        pd.DataFrame(summ).to_csv(f'{OUT}/summary_base.csv', index=False, float_format='%.4f')
        print(pd.DataFrame(summ).T.to_string())
    sc = fig_scan()
    sc.to_csv(f'{OUT}/summary_stop.csv', index=False, float_format='%.2f')
    print(sc.to_string())
    fig_chatter()
    print('figs ok')
