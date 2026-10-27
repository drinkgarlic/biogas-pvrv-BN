# -*- coding: utf-8 -*-
"""v4 (on/off operation) sensitivity: Case 1, flare start 100 % and 80 %; base = automatic retry 30 min, PFD 0.05, no pressure backup"""
import os, sys, copy, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_bn as bn
HERE = os.path.dirname(os.path.abspath(__file__))

def base_prm(fo):
    prm = copy.deepcopy(bn.PARAMS)
    prm['f_on']['v'] = fo; prm['t_retry_h']['v'] = 0.5; prm['n_retry']['v'] = 4; prm['p_hard']['v'] = 0.0; prm['pressure_backup']['v'] = False
    tag = f'onoff_c1_f{int(round(fo*100))}'                       # Case 1 series from the 1-s model (v4.3)
    prm['fill_series']['v'] = f'fill_hourly_{tag}.csv'; prm['demand_series']['v'] = f'flare_demands_{tag}.csv'
    return prm

SENS = [
    ('ignition failure probability 0.01 / 0.10', {'pfd_flare': 0.01}, {'pfd_flare': 0.10}),
    ('share of hard faults (retry useless) 0.1 / 0.3', {'p_hard': 0.1}, {'p_hard': 0.3}),
    ('automatic retry interval 15 / 60 min', {'t_retry_h': 0.25}, {'t_retry_h': 1.0}),
    ('8-h response share 0 (alarm trusted) / doubled', {'R_day': {0.5: 0.65, 2.0: 0.35, 8.0: 0.0}, 'R_night': {0.5: 0.3, 2.0: 0.7, 8.0: 0.0}},
                                   {'R_day': {0.5: 0.5, 2.0: 0.3, 8.0: 0.2}, 'R_night': {0.5: 0.1, 2.0: 0.2, 8.0: 0.7}}),
    ('operator response always daytime / always night', {'R_night': {0.5: 0.6, 2.0: 0.3, 8.0: 0.1}}, {'R_day': {0.5: 0.2, 2.0: 0.4, 8.0: 0.4}}),
    ('level indicator error +5 % / -5 % (reads low)', {'level_bias': {-0.05: 1.0}}, {'level_bias': {0.05: 1.0}}),
    ('downstream stops 6 / 24 per year', {'lam_stop': 6.0}, {'lam_stop': 24.0}),
    ('stop duration shorter / longer', {'D_dist': {0.5: 0.6, 2.0: 0.3, 6.0: 0.08, 24.0: 0.02}}, {'D_dist': {0.5: 0.3, 2.0: 0.3, 6.0: 0.25, 24.0: 0.15}}),
    ('foaming 1 / 3 per year', {'lam_foam': 1.0}, {'lam_foam': 3.0}),
    ('probability that foam fouls the PVRV 0.3 / 0.03', {'p_pv_fouled': 0.3}, {'p_pv_fouled': 0.03}),
    ('condensate / XV blockage 0.3 / 1.2 per year', {'lam_cond': 0.25, 'lam_xv': 0.05}, {'lam_cond': 1.0, 'lam_xv': 0.2}),
    ('vent valve stuck 0.02 / 0.5 per year', {'lam_wv': 0.02}, {'lam_wv': 0.5}),
    ('stuck latent period 72 / 336 h', {'latent_wv_h': 72}, {'latent_wv_h': 336}),
]

res = {}
for fo in (1.0, 0.8):
    prm0 = base_prm(fo); inp = bn.load_inputs(prm=prm0)
    def total(mod):
        prm = copy.deepcopy(prm0)
        for k, v in mod.items(): prm[k]['v'] = v
        s = bn.summarize(bn.run_year(1, inp, prm), prm)['total']
        return s['openings_per_yr'], s['vent_m3_per_yr']
    b_n, b_v = total({})
    rows = []
    for name, lo, hi in SENS:
        n_lo, v_lo = total(lo); n_hi, v_hi = total(hi)
        rows.append(dict(parameter=name, n_low=n_lo, n_high=n_hi, v_low=v_lo, v_high=v_hi))
    df = pd.DataFrame(rows); tag = f'f{int(round(fo*100))}'
    df.to_csv(os.path.join(HERE, 'out', f'bn_sens_v4_case1_{tag}.csv'), index=False, float_format='%.4f')
    res[tag] = dict(base_openings=b_n, base_vent=b_v)
    print(f'--- f_on {fo}: base {b_n:.3f} per year, {b_v:.0f} m3'); print(df.round(3).to_string(index=False))
json.dump(res, open(os.path.join(HERE, 'out', 'bn_sens_v4_base.json'), 'w'), indent=1)
