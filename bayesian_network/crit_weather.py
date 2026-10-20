# -*- coding: utf-8 -*-
"""Severe-weather scenarios: a synthetic typhoon (Gaussian pressure drop) is added to the KMA hourly pressure and the 1-s model, the ratchet, the event scan and the BN are recomputed"""
import sys, os, json, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pressure_model'))
import physics as ph, pv_bn as bn
import pressure_model as pm, scan_stop as ss

T0 = pd.Timestamp('2021-09-06 12:00')          # virtual typhoon centre passage time aligned with the Hinnamnor passage day
SCEN = {'base': (0.0, 8.0), 'typhoon': (5.0, 8.0), 'record': (6.0, 6.0)}   # (drop kPa, sigma h)

def make_p(grid, p, A, sig):
    th = (pd.DatetimeIndex(grid) - T0).total_seconds().to_numpy() / 3600
    return p - A * np.exp(-(th / sig) ** 2)

inp = bn.load_inputs()
grid = inp['t']
res = {}
for name, (A, sig) in SCEN.items():
    p = make_p(grid, inp['p'], A, sig)
    s = pd.Series(p, index=grid)
    d3 = -(s.diff(3)).min(); d24 = -(s.diff(24)).min(); rate = -(s.diff()).min()
    r = dict(p_min=float(p.min()), drop3h=float(d3), drop24h=float(d24), max_rate_kPa_h=float(rate))
    # --- ratchet (vent valve stuck, 7-day latent period)
    for c in (1, 2, 3):
        to, vent = ph.wv_stuck_ratchet(p, c, 168)
        r[f'wv_reach_case{c}'] = float((~np.isnan(to)).mean())
        gi = pd.DatetimeIndex(grid)[:len(to)]; w = (gi >= T0 - pd.Timedelta(hours=72)) & (gi <= T0)
        r[f'wv_reach_case{c}_if_stuck_3d_before'] = float((~np.isnan(to[np.asarray(w)])).mean())
    # --- BN annual
    inp2 = dict(inp); inp2['p'] = p
    for c in (1, 2, 3):
        df = bn.run_year(c, inp2); sm = bn.summarize(df)
        r[f'bn_open_case{c}'] = sm['total']['openings_per_yr']; r[f'bn_wv_case{c}'] = sm['wv']['openings_per_yr']
    res[name] = r
    print(name, json.dumps({k: round(v, 4) for k, v in r.items()}))

# --- 1-s model: 3 days around the typhoon, normal operation (all consumed) and vent valve stuck
pinp = pm.load_inputs()
h0 = int((T0 - pd.Timedelta(hours=36) - pinp['grid'][0]).total_seconds() // 3600)
out = {}
for name, (A, sig) in SCEN.items():
    pinp2 = dict(pinp); pinp2['patm_h'] = make_p(pd.DatetimeIndex(pinp['grid']), pinp['patm_h'], A, sig)
    for c in (1, 2, 3):
        P = dict(pm.BASE); P['p_op'] = pm.CASES[c]
        df, ev, sec, sm = pm.run_year(pinp2, P, n_sec=72 * 3600, t0_off=h0 * 3600.0, verbose=False)
        key = f'{name}_case{c}_normal'
        out[key] = dict(PV_opens=sm['PV_A_opens'] + sm['PV_B_opens'], PV_vent=sm['PV_vented_m3'], WV_opens=sm['WV_opens'], blower_starts=sm['blower_starts'],
                        PT_max=float(df['PT101A_max_kPag'].max()), PT_min=float(df['PT101A_min_kPag'].min()), WV_air=sm['WV_vented_air_m3'])
        P2 = dict(P); P2['Q_W'] = 0.0                         # vent valve stuck closed, no flare backup
        df, ev, sec, sm = pm.run_year(pinp2, P2, n_sec=72 * 3600, t0_off=h0 * 3600.0, verbose=False)
        o = ev[ev.code == 1]
        out[f'{name}_case{c}_wvstuck'] = dict(PV_opens=sm['PV_A_opens'] + sm['PV_B_opens'], PV_vent=sm['PV_vented_m3'], CH4_kg=sm['PV_CH4_kg'],
                                              open_s=sm['PV_A_open_s'] + sm['PV_B_open_s'], PT_max=float(df['PT101A_max_kPag'].max()),
                                              first_open=str(o['datetime'].iloc[0]) if len(o) else None)
for k, v in out.items():
    print(k, {kk: (round(vv, 2) if isinstance(vv, (int, float)) else vv) for kk, vv in v.items()})

# --- event scan: downstream stops starting within ±24 h of the typhoon (holder 85 %)
scan = {}
for name, (A, sig) in SCEN.items():
    pinp2 = dict(pinp); pinp2['patm_h'] = make_p(pd.DatetimeIndex(pinp['grid']), pinp['patm_h'], A, sig)
    starts = np.arange(int((T0 - pd.Timedelta(hours=24) - pinp['grid'][0]).total_seconds() // 3600), int((T0 + pd.Timedelta(hours=24) - pinp['grid'][0]).total_seconds() // 3600))
    for c in (1, 2, 3):
        P = dict(pm.BASE); P['p_op'] = pm.CASES[c]
        o = ss.scan(pinp2, P, 0.85, H_h=3.0, starts=starts)
        rig = (o['t_first_open_min'] - o['t_holder_full_min'])
        scan[f'{name}_case{c}'] = dict(t_open_med=float(o['t_first_open_min'].median()), t_open_min=float(o['t_first_open_min'].min()),
                                       rigid_med=float(rig.median()), rigid_min=float(rig.min()), vent1h_med=float(o['vent_1h_after_open_m3'].median()))
        print(name, c, {k: round(v, 2) for k, v in scan[f'{name}_case{c}'].items()})
json.dump(dict(scen=res, sec=out, scan=scan), open('out/crit_weather.json', 'w'), indent=1, default=float)
