# -*- coding: utf-8 -*-
"""
Dynamic QRA, stage 3: event tree after "PVRV opens" and the time-resolved risk integration.

Frequency side  : the hourly BN results (bayesian_network/pv_bn.py, v4) give, per hour h and initiating event k,
                    p_k(h)  probability that an event starting in hour h opens the valve
                    o_k(h)  ... and that the operator reaches the valve while it is still venting
                    s_k(h)  probability that the valve is open during hour h (state probability)
                    d_k(h)  expected open duration (h)
Consequence side: effect distances from consequence.py (plume on the digester roof), weather mix by day / night,
                  flash fire (person inside the LFL plume at ignition -> fatality) and H2S (RIVM probit).
Exposure        : two people are at risk - the operator who responds to the event and goes up to the valve (the dominant
                  term) and the routine inspection presence on the roof. Both are assumptions and are varied in the tornado.
Risk measures   : IR of the responding operator, IR of the routine round, location-specific IR (LSIR) 1 m from the vent for a
                  hypothetical permanent presence, annual CH4 emission, all per hour -> annual, monthly and hour-of-day profiles.
Cases           : Case 1 (4.0 kPag) and Case 3 (2.0 kPag) x flare start 100 % / 80 %  (+ the generic-frequency QRA for comparison)
"""
import os, sys, json, copy
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'bayesian_network')); sys.path.insert(0, HERE)
import pv_bn as bn
import consequence as cq
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- QRA parameters (sources / assumptions)
QRA = {
    'weather_day':   dict(v={'D5': 0.5, 'D3': 0.3, 'calm': 0.2}, src='Assumption (Pohang ASOS wind statistics can replace it): daytime mix of D / 5 m/s, D / 3 m/s and calm'),
    'weather_night': dict(v={'F2': 0.6, 'D3': 0.2, 'calm': 0.2}, src='Assumption: night-time mix of F / 2 m/s, D / 3 m/s and calm'),
    'p_ign':         dict(v=0.02, src='TNO Purple Book (CPR 18E) table 4.3: continuous release of flammable gas < 10 kg/s, immediate + delayed ignition'),
    'p_ign_worker':  dict(v=0.002, src='Assumption: ignition source brought by the worker at the valve (tools, torch, static) per visit; hot-work rules apply'),
    'p_fat_flash_small': dict(v=0.10, src='Assumption: fatality given ignition while inside the sub-metre plume at the vent (face/hand burns likely, death unlikely); TNO uses 1.0 for engulfment in large clouds'),
    'p_fat_flash_R5': dict(v=1.0, src='TNO Green Book: fatality 1.0 inside the flash-fire envelope of the 30-s depressurisation cloud (LFL 6-13 m)'),
    'p_zone':        dict(v=0.10, src='Assumption from the plume geometry (radius 0.1-0.4 m at the LFL distance): probability that a person at 1 m from the vent is inside the flammable plume'),
    'p_zone_R5':     dict(v=0.50, src='Assumption: the 30-s depressurisation cloud (LFL 6-13 m) covers most of the roof'),
    'p_zone_routine': dict(v=0.03, src='Assumption: random position on the roof (radius ~10 m) vs. a plume a few metres long'),
    't_at_valve_min': dict(v=5.0, src='Assumption: minutes a responding operator spends at a venting valve'),
    't_fix_min':     dict(v=15.0, src='Assumption: minutes from arrival until the venting is stopped (flare restart, bypass, clearing the blockage)'),
    'p_in_plume_h2s': dict(v=0.02, src='Assumption: probability that the visit involves a prolonged stay inside the H2S plume (downwind of the vent, leaning over it) despite the personal H2S alarm (10 ppm) that plant staff carry; 0.1-0.3 without a detector (tornado)'),
    'lsir_distance_m': dict(v=3.0, src='Location-specific IR evaluated for a permanent presence 3 m from the vent (roof walkway)'),
    'p_roof_day':    dict(v=0.5, src='Assumption: probability that the responding operator goes up to the digester roof while the valve vents (day)'),
    'p_roof_night':  dict(v=0.3, src='Assumption: same at night'),
    'routine_day':   dict(v=0.012, src='Assumption: daily 5-min inspection at each PVRV + 24 h/yr maintenance, within the daytime hours (08-20)'),
    'routine_night': dict(v=0.0, src='Assumption: no routine presence at night'),
    'p_fouled':      dict(v=0.10, src='Assumption: probability that foam reaching the digester outlet also fouls the PVRV. The BN uses 0.3 for the opening branch; 0.1 here keeps the seal-failure chain (0.6 x 0.1 x not prevented) near the EA 2019 incident base rate'),
    'p_injury_seal': dict(v=0.10, src='Assumption: fatality probability for a person on the roof at the moment a roof seal / manhole fails at 10 kPag'),
    'p_on_roof_at_failure': dict(v=0.1, src='Assumption: share of the responders who are on the roof at the moment the seal fails (27 min after the blockage)'),
    'yH2S_ppm':      dict(v=2527.0, src='PyADM1ODE annual mean of the raw gas (930-5,845 ppm). If the plant desulphurises in the digester headspace (micro-aeration) the PVRV sees 200-500 ppm instead - the single most important composition parameter of the QRA'),
    'duty':          dict(v={'R1 chatter (valve open)': 44.0 / 141.0, 'R2 downstream stop': 84.5 / 141.0, 'R3 isolated digester': 41.2 / 141.0, 'R5 blocked PVRV, continuous': 1.0},
                          src='1-s model: every release is realised as pulses at the minimum stable opening (141 m3/h); duty = time-mean rate / 141. The H2S dose (probit exponent n = 1.9) uses duty x stay at the open-valve concentration, i.e. delta^(1-n) = 2.9x more than a time-mean plume would give for R1'),
    'generic_f_pvrv': dict(v=0.01, src='Generic-QRA comparison: spurious / demand-driven PVRV opening 1e-2 per valve-year (CCPS-type initiating-event value), 2 valves'),
    'generic_dur_h': dict(v=1.0, src='Generic-QRA comparison: assumed release duration'),
}
CASES = [(1, 1.0), (1, 0.8), (3, 1.0), (3, 0.8)]
BR = ['demand', 'stop', 'foam', 'block', 'wv']
REL = {'demand': 'R1 chatter (valve open)', 'stop': 'R2 downstream stop', 'foam': 'R3 isolated digester', 'block': 'R3 isolated digester', 'wv': 'R1 chatter (valve open)'}
WX = {'D5': ('D', 5.0), 'D3': ('D', 3.0), 'F2': ('F', 2.0), 'calm': (None, 0.5)}

def effects(Q_m3h, wname, yH2S=None):
    yH2S = cq.yH2S if yH2S is None else yH2S
    """(C at 1 m vol fraction, H2S ppm at 1 m, LFL distance) for a continuous release Q at the vent in weather class wname"""
    cls, u = WX[wname]; Q = Q_m3h / 3600
    if cls is None:
        c1 = float(cq.mtt_plume(Q, 1.0)[0]); z = np.linspace(0.3, 30, 300); c, _ = cq.mtt_plume(Q, z); xl = float(z[np.argmax(c < cq.LFL)]) if (c < cq.LFL).any() else 30.0
    else:
        c1 = float(cq.plume_centreline(Q, u, cls, 1.0)); xl = cq.dist_to(cq.LFL, Q, u, cls)
    return c1, c1 * yH2S * 1e6, xl

def fatality_tables(P):
    """P(fatality | exposure mode, release type, weather) from the consequence models and the QRA parameters"""
    tab = {}
    rates = {'R1 chatter (valve open)': 141.0, 'R2 downstream stop': 141.0, 'R3 isolated digester': 141.0, 'R5 blocked PVRV, continuous': 41.2}
    for rel, Qh in rates.items():
        for w in WX:
            yH2S = P['yH2S_ppm']['v'] * 1e-6; delta = P['duty']['v'][rel]
            c1, h2s1, xl = effects(Qh, w, yH2S)
            t_h2s = P['t_at_valve_min']['v'] * delta                                         # minutes of the stay during which the valve is actually open
            f_overlap = min(1.0, P['t_at_valve_min']['v'] / P['t_fix_min']['v'])            # share of the remaining release spent at the valve
            flash_resp = P['p_zone']['v'] * (P['p_ign_worker']['v'] + P['p_ign']['v'] * f_overlap) * P['p_fat_flash_small']['v']
            h2s_resp = P['p_in_plume_h2s']['v'] * float(cq.h2s_probit_fatality(h2s1, max(t_h2s, 0.1)))   # inside the plume: dose at the open-valve concentration for duty x stay
            # location risk at lsir_distance_m: flash fire only if the LFL plume reaches that far, H2S for a 30-min stay in the plume
            xd = P['lsir_distance_m']['v']; cls, u = WX[w]; Q = Qh / 3600
            cd = float(cq.mtt_plume(Q, xd)[0]) if cls is None else float(cq.plume_centreline(Q, u, cls, xd))
            lsir_ev = (P['p_zone']['v'] * P['p_ign']['v'] * P['p_fat_flash_small']['v'] if xl >= xd else 0.0) + P['p_in_plume_h2s']['v'] * float(cq.h2s_probit_fatality(cd * yH2S * 1e6, 30.0 * delta))
            tab[(rel, w)] = dict(C_1m=c1, H2S_1m_ppm=h2s1, LFL_m=xl, duty=delta, p_fat_responder=flash_resp + h2s_resp, p_fat_flash=flash_resp, p_fat_h2s=h2s_resp, H2S_lsir_ppm=cd * yH2S * 1e6, p_fat_lsir_per_event=lsir_ev)
    return tab

def weather_mix(P, day):
    return P['weather_day']['v'] if day else P['weather_night']['v']

def run_case(case, f_on, P=QRA, prm_over=None):
    prm = copy.deepcopy(bn.PARAMS); prm['f_on']['v'] = f_on; prm['t_retry_h']['v'] = 0.5; prm['n_retry']['v'] = 4; prm['pressure_backup']['v'] = False
    tag = f'onoff_f{int(round(f_on * 100))}'; prm['fill_series']['v'] = f'fill_hourly_{tag}.csv'; prm['demand_series']['v'] = f'flare_demands_{tag}.csv'
    for k, v in (prm_over or {}).items(): prm[k]['v'] = v
    inp = bn.load_inputs(prm=prm)
    df = bn.run_year(case, inp, prm, occupancy=True)
    ft = fatality_tables(P)
    day = inp['day'][:len(df)]; n = len(df)
    out = df[['h', 'q', 'fill', 'p_open', 'p_state_open', 'v_open', 'ch4_kg']].copy(); out['day'] = day
    out['datetime'] = inp['t'].iloc[:n].to_numpy()
    tot_resp = np.zeros(n); tot_rout = np.zeros(n); tot_lsir = np.zeros(n)
    for k in BR:
        rel = REL[k]
        p_roof = np.where(day, P['p_roof_day']['v'], P['p_roof_night']['v']); rout = np.where(day, P['routine_day']['v'], P['routine_night']['v'])
        pf_resp = np.zeros(n); pf_lsir = np.zeros(n)
        for w in WX:
            wd = np.where(day, weather_mix(P, True).get(w, 0.0), weather_mix(P, False).get(w, 0.0))
            pf_resp += wd * ft[(rel, w)]['p_fat_responder']; pf_lsir += wd * ft[(rel, w)]['p_fat_lsir_per_event']
        dur = np.where(df[f'p_{k}'] > 0, df[f'd_{k}'] / np.maximum(df[f'p_{k}'], 1e-12), 1.0)          # expected duration per event (h)
        ir_resp = df[f'o_{k}'].to_numpy() * p_roof * pf_resp
        ir_rout = df[f's_{k}'].to_numpy() * rout * P['p_zone_routine']['v'] * P['p_ign']['v'] * P['p_fat_flash_small']['v'] / np.maximum(dur, 0.1)
        lsir = df[f'p_{k}'].to_numpy() * pf_lsir
        out[f'IR_resp_{k}'] = ir_resp; out[f'IR_rout_{k}'] = ir_rout; out[f'LSIR_{k}'] = lsir
        tot_resp += ir_resp; tot_rout += ir_rout; tot_lsir += lsir
    # R5: foam reaches the outlet AND fouls the PVRV -> no relief -> seal failure unless the operator intervenes within 30 min
    p_foam_h = prm['lam_foam']['v'] * prm['p_block_foam']['v'] / 8760
    p_prevent = np.where(day, prm['R_day']['v'][0.5], prm['R_night']['v'][0.5])
    p_R5 = p_foam_h * P['p_fouled']['v'] * (1 - p_prevent)
    p_roof = np.where(day, P['p_roof_day']['v'], P['p_roof_night']['v'])
    pf5 = np.zeros(n)
    for w in WX:
        wd = np.where(day, weather_mix(P, True).get(w, 0.0), weather_mix(P, False).get(w, 0.0))
        c1, h2s1, xl = effects(41.2, w, P['yH2S_ppm']['v'] * 1e-6)
        pf5 += wd * (P['p_zone_R5']['v'] * (P['p_ign']['v'] + P['p_ign_worker']['v']) * P['p_fat_flash_R5']['v'] + P['p_in_plume_h2s']['v'] * float(cq.h2s_probit_fatality(h2s1, P['t_at_valve_min']['v'])))
    ir_R5 = p_R5 * p_roof * (P['p_on_roof_at_failure']['v'] * P['p_injury_seal']['v'] + pf5)
    out['p_R5'] = p_R5; out['IR_resp_R5'] = ir_R5; tot_resp += ir_R5
    out['LSIR_R5'] = p_R5 * (P['p_zone_R5']['v'] * P['p_ign']['v'] * P['p_fat_flash_R5']['v']); tot_lsir += out['LSIR_R5']
    out['IR_resp'] = tot_resp; out['IR_rout'] = tot_rout; out['IR_total'] = tot_resp + tot_rout; out['LSIR'] = tot_lsir
    return out, ft

def summarize(out, case, f_on):
    s = dict(case=case, f_on=f_on, openings=out['p_open'].sum(), R5_events=out['p_R5'].sum(), open_h=out['p_state_open'].sum(), vent_m3=out['v_open'].sum(), ch4_kg=out['ch4_kg'].sum(),
             IR_resp=out['IR_resp'].sum(), IR_rout=out['IR_rout'].sum(), IR_total=out['IR_total'].sum(), LSIR=out['LSIR'].sum())
    for k in BR + ['R5']:
        s[f'IR_resp_{k}'] = out[f'IR_resp_{k}'].sum()
    s['IR_day'] = out.loc[out.day, 'IR_total'].sum(); s['IR_night'] = out.loc[~out.day, 'IR_total'].sum()
    return s

def generic_qra(P=QRA):
    """conventional QRA: generic PVRV opening frequency, no time structure, same exposure and consequence assumptions"""
    f = P['generic_f_pvrv']['v'] * 2
    ft = fatality_tables(P)
    pf = sum(0.5 * weather_mix(P, True).get(w, 0) * ft[('R2 downstream stop', w)]['p_fat_responder'] + 0.5 * weather_mix(P, False).get(w, 0) * ft[('R2 downstream stop', w)]['p_fat_responder'] for w in WX)
    p_roof = 0.5 * P['p_roof_day']['v'] + 0.5 * P['p_roof_night']['v']
    lsir = sum(0.5 * weather_mix(P, True).get(w, 0) * ft[('R2 downstream stop', w)]['p_fat_lsir_per_event'] + 0.5 * weather_mix(P, False).get(w, 0) * ft[('R2 downstream stop', w)]['p_fat_lsir_per_event'] for w in WX)
    return dict(openings=f, IR_resp=f * p_roof * pf, LSIR=f * lsir, vent_m3=f * 84.5 * P['generic_dur_h']['v'])

if __name__ == '__main__':
    rows = []; profiles = {}
    for case, f_on in CASES:
        out, ft = run_case(case, f_on)
        out.to_csv(os.path.join(OUT, f'risk_hourly_case{case}_f{int(f_on * 100)}.csv.gz'), index=False, float_format='%.6g')
        rows.append(summarize(out, case, f_on)); profiles[(case, f_on)] = out
        print(f'Case {case}, flare start {int(f_on * 100)} %: openings {rows[-1]["openings"]:.2f}, IR responder {rows[-1]["IR_resp"]:.2e}, routine {rows[-1]["IR_rout"]:.2e}, LSIR(1 m) {rows[-1]["LSIR"]:.2e}', flush=True)
    r = pd.DataFrame(rows); g = generic_qra(); g.update(case='generic', f_on=np.nan); r = pd.concat([r, pd.DataFrame([g])], ignore_index=True)
    r.to_csv(os.path.join(OUT, 'risk_summary.csv'), index=False, float_format='%.4g')
    pd.set_option('display.width', 250); print(r.round(10).to_string(index=False))
    pd.DataFrame([dict(release=k[0], weather=k[1], **v) for k, v in fatality_tables(QRA).items()]).to_csv(os.path.join(OUT, 'fatality_table.csv'), index=False, float_format='%.4g')
    # monthly / hour-of-day profiles
    prof = []
    for (case, f_on), out in profiles.items():
        d = out.set_index('datetime')
        m = d[['IR_total', 'LSIR', 'p_open', 'ch4_kg']].resample('MS').sum(); m['case'] = case; m['f_on'] = f_on; m['kind'] = 'month'; prof.append(m.reset_index())
        hd = d[['IR_total', 'LSIR', 'p_open', 'p_state_open']].groupby(d.index.hour).sum(); hd['case'] = case; hd['f_on'] = f_on; hd['kind'] = 'hour'; prof.append(hd.reset_index().rename(columns={'datetime': 'hour'}))
    pd.concat(prof, ignore_index=True).to_csv(os.path.join(OUT, 'risk_profiles.csv'), index=False, float_format='%.6g')
    json.dump({k: v for k, v in QRA.items()}, open(os.path.join(OUT, 'qra_params.json'), 'w'), indent=1, default=str)
    # tornado on the QRA assumptions, Case 1 x (100 %, 80 %)
    SENS = [('ignition probability 0.005 / 0.05', {'p_ign': 0.005}, {'p_ign': 0.05}), ('worker ignition source 0 / 0.01', {'p_ign_worker': 0.0}, {'p_ign_worker': 0.01}),
            ('fatality given flash in the small plume 0.03 / 1.0', {'p_fat_flash_small': 0.03}, {'p_fat_flash_small': 1.0}),
            ('P(inside flammable plume at the valve) 0.03 / 0.3', {'p_zone': 0.03}, {'p_zone': 0.3}), ('responder goes to the roof 0.2 / 0.8 (day)', {'p_roof_day': 0.2, 'p_roof_night': 0.1}, {'p_roof_day': 0.8, 'p_roof_night': 0.6}),
            ('time at the valve 2 / 15 min', {'t_at_valve_min': 2.0}, {'t_at_valve_min': 15.0}), ('prolonged stay in the H2S plume 0.005 / 0.3 (no detector)', {'p_in_plume_h2s': 0.005}, {'p_in_plume_h2s': 0.3}),
            ('PVRV fouled by foam 0.03 / 0.3', {'p_fouled': 0.03}, {'p_fouled': 0.3}),
            ('H2S in the gas 500 ppm (in-situ desulphurisation) / 5,845 ppm (max)', {'yH2S_ppm': 500.0}, {'yH2S_ppm': 5845.0}),
            ('duty cycle ignored (time-mean plume) / pulses at full stay', {'duty': {'R1 chatter (valve open)': 0.31 ** 0.9 * 0.31, 'R2 downstream stop': 0.6 ** 0.9 * 0.6, 'R3 isolated digester': 0.29 ** 0.9 * 0.29, 'R5 blocked PVRV, continuous': 1.0}}, {'duty': {'R1 chatter (valve open)': 1.0, 'R2 downstream stop': 1.0, 'R3 isolated digester': 1.0, 'R5 blocked PVRV, continuous': 1.0}}),
            ('weather: all D5 / all calm', {'weather_day': {'D5': 1.0}, 'weather_night': {'D5': 1.0}}, {'weather_day': {'calm': 1.0}, 'weather_night': {'calm': 1.0}}),
            ('routine presence x0 / x3', {'routine_day': 0.0}, {'routine_day': 0.036}), ('seal-failure injury 0.02 / 0.3', {'p_injury_seal': 0.02}, {'p_injury_seal': 0.3})]
    srows = []
    for case, f_on in ((1, 1.0), (1, 0.8)):
        base = summarize(run_case(case, f_on)[0], case, f_on)['IR_total']
        for name, lo, hi in SENS:
            P1 = copy.deepcopy(QRA); P2 = copy.deepcopy(QRA)
            for k, v in lo.items(): P1[k]['v'] = v
            for k, v in hi.items(): P2[k]['v'] = v
            srows.append(dict(case=case, f_on=f_on, parameter=name, base=base, low=summarize(run_case(case, f_on, P1)[0], case, f_on)['IR_total'], high=summarize(run_case(case, f_on, P2)[0], case, f_on)['IR_total']))
        # BN-side parameter as well
        for name, lo, hi in (('flare PFD 0.01 / 0.10', {'pfd_flare': 0.01}, {'pfd_flare': 0.10}), ('foaming 1 / 3 per year', {'lam_foam': 1.0}, {'lam_foam': 3.0}), ('hard-fault share 0 / 0.3', {'p_hard': 0.0}, {'p_hard': 0.3})):
            srows.append(dict(case=case, f_on=f_on, parameter=name, base=base, low=summarize(run_case(case, f_on, prm_over=lo)[0], case, f_on)['IR_total'], high=summarize(run_case(case, f_on, prm_over=hi)[0], case, f_on)['IR_total']))
    sd = pd.DataFrame(srows); sd.to_csv(os.path.join(OUT, 'risk_sensitivity.csv'), index=False, float_format='%.4g'); print(sd.round(10).to_string(index=False))
