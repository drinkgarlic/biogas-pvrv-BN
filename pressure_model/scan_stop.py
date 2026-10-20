# -*- coding: utf-8 -*-
"""
Downstream-stop event scan: a 'holder outlet flow = 0' event is assumed to start at every full hour of the year
(8,754 events) and each event is simulated for H hours at 1-s resolution. Same physics as pressure_model.kernel,
vectorised over events with numpy.

Event start state: holder gauge = operating pressure, fill fraction f0, gas production / atmospheric pressure /
composition from the inputs at that hour.
Recorded: time to the first relief-valve opening, number of openings, open time, vented volume (CH4, H2S),
          cumulative values at 2 / 4 / 6 h, vented volume within 1 h and 2 h after the first opening
          (for operator-response assumptions), maximum PT value.
Usage: python scan_stop.py <case> <f0>   (e.g. python scan_stop.py 1 0.5)
"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pressure_model as pm
from pressure_model import P_REF, T_REF, R_GAS, M_AIR, G, RHO_CH4_REF, RHO_H2S_REF, PIPE_ARR

def vdp(Q, D, L, eps, K, rho, mu):
    q = np.abs(Q) / 3600.0
    v = q / (np.pi * D * D / 4)
    Re = np.maximum(rho * v * D / mu, 1e-9)
    f_t = 0.25 / np.log10(eps / (3.7 * D) + 5.74 / Re ** 0.9) ** 2
    w = np.clip((Re - 2000.0) / 2000.0, 0, 1)
    f = (1 - w) * 64.0 / Re + w * f_t
    dp = (f * L / D + K) * rho * v * v / 2
    return np.where(Q > 0, dp, -dp)

def vvalve(p, pset, QR, was_open, h, fmin):
    is_open = np.where(was_open, p > pset, p >= pset * (1 + h))
    q = np.where(p <= 1.1 * pset, QR * (p - pset) / (0.1 * pset), QR * np.sqrt(np.maximum(p, 0) / (1.1 * pset)))
    q = np.where(is_open, np.maximum(q, fmin * QR), 0.0)
    return q, is_open

def scan(inp, P, f0, H_h=6.0, dt=1.0, starts=None, checkpoints=(7200, 14400, 21600)):
    q_h, patm_h, M_h, mu_h, yCH4_h, yH2S_h = (inp[k] for k in ('q_h', 'patm_h', 'M_h', 'mu_h', 'yCH4_h', 'yH2S_h'))
    n_h = len(q_h) - 1
    H = int(H_h * 3600)
    if starts is None:
        starts = np.arange(0, n_h - int(H_h) + 1)        # event start hours
    t0 = starts.astype(float) * 3600.0
    N = len(starts)
    c = P_REF / T_REF
    V_hs, V_pipe, V_shell, V_max = P['V_hs'], P['V_pipe'], P['V_shell'], P['V_max']
    T_hs, T_pipe, T_h, T_amb = P['T_hs'], P['T_pipe'], P['T_h'], P['T_amb']
    a_hs, a_hs1, a_pipe, a_sh = 2 * V_hs / T_hs, V_hs / T_hs, V_pipe / T_pipe, V_shell / T_h
    p_op, Q_W, f_min_W, dp_bl, Q_bl = P['p_op'], P['Q_W'], P['f_min_W'], P['dp_bl'], P['Q_bl']
    p_PV, Q_PV, tol_B, f_min_PV, h_open = P['p_PV'], P['Q_PV'], P['tol_B'], P['f_min_PV'], P['h_open']
    z_in, z_PT, z_PV, z_N1 = P['z_in'], P['z_PT'], P['z_PV'], P['z_N1']
    pv_abs = P['pv_ref'] == 'abs'
    # initial state
    i0 = np.minimum((t0 // 3600).astype(int), n_h - 1)
    patm0 = patm_h[i0]
    p0 = patm0 - 1.2 * G * z_in / 1000 + p_op
    V_g0 = f0 * V_max
    n_g = p0 * (a_hs + a_pipe + V_g0 / T_h) / c
    n_a = p0 * (V_shell - V_g0) / T_h / c
    blower = np.zeros(N, bool); w_open = np.zeros(N, bool)
    pvA = np.zeros(N, bool); pvB = np.zeros(N, bool)
    p_g_prev = None; qpvA_p = np.zeros(N); qpvB_p = np.zeros(N)
    dpfA_f = None; lam = dt / P['tau_pipe'] if P['tau_pipe'] > 0 else 1.0
    # records
    t_first = np.full(N, np.nan); opens_A = np.zeros(N); opens_B = np.zeros(N)
    open_s = np.zeros(N); vent = np.zeros(N); ch4 = np.zeros(N); h2s = np.zeros(N); pt_max = np.zeros(N)
    t_full = np.full(N, np.nan)
    chk = {}
    vent_1h = np.full(N, np.nan); vent_2h = np.full(N, np.nan); open_1h = np.full(N, np.nan); open_2h = np.full(N, np.nan)
    opens_1h = np.full(N, np.nan); opens_2h = np.full(N, np.nan)
    tw = time.time()
    for k in range(H):
        t = t0 + (k + 1) * dt
        i = np.minimum((t // 3600).astype(int), n_h - 1)
        fr = t / 3600.0 - i
        q_gen = q_h[i] + (q_h[i + 1] - q_h[i]) * fr
        p_atm = patm_h[i] + (patm_h[i + 1] - patm_h[i]) * fr
        M = M_h[i] + (M_h[i + 1] - M_h[i]) * fr
        mu = mu_h[i] + (mu_h[i + 1] - mu_h[i]) * fr
        X = a_hs + a_pipe + a_sh
        Y = n_a * c
        p_a = (n_g * c + Y) / X
        V_gh = V_shell - Y * T_h / p_a
        full = V_gh >= V_max
        empty = V_gh <= 0
        p_g = np.where(full, n_g * c / (a_hs + a_pipe + V_max / T_h), np.where(empty, n_g * c / (a_hs + a_pipe), p_a))
        p_a = np.where(full, Y * T_h / (V_shell - V_max), np.where(empty, Y * T_h / V_shell, p_a))
        V_gh = np.clip(V_gh, 0, V_max)
        newly_full = full & np.isnan(t_full)
        t_full[newly_full] = (k + 1) * dt
        rho_air = p_atm * 1000 * M_AIR / (R_GAS * T_amb)
        rho_g = p_g * 1000 * M / (R_GAS * T_pipe)
        p_atm_in = p_atm - rho_air * G * z_in / 1000
        pg_g = p_g - p_atm_in; pa_g = p_a - p_atm_in
        dpdt = np.zeros(N) if p_g_prev is None else (p_g - p_g_prev) / dt
        q_stor = a_hs1 / c * dpdt * 3600
        Q_A = q_gen * 0.5 - q_stor - qpvA_p
        Q_B = q_gen * 0.5 - q_stor - qpvB_p
        Q_T = Q_A + Q_B
        dp_c = vdp(Q_T, *PIPE_ARR[3], rho_g, mu) + vdp(Q_T, *PIPE_ARR[4], rho_g, mu)
        dpf_A = vdp(Q_A, *PIPE_ARR[0], rho_g, mu) + vdp(Q_A, *PIPE_ARR[2], rho_g, mu) + dp_c
        dpf_B = vdp(Q_B, *PIPE_ARR[1], rho_g, mu) + dp_c
        if dpfA_f is None:
            dpfA_f, dpfB_f = dpf_A.copy(), dpf_B.copy()
        dpfA_f = dpfA_f + (dpf_A - dpfA_f) * lam; dpfB_f = dpfB_f + (dpf_B - dpfB_f) * lam
        dpf_A, dpf_B = dpfA_f, dpfB_f
        head = (rho_air - rho_g) * G
        pPT_A = pg_g + (head * (z_PT - z_in) + dpf_A) / 1000
        pPV_A = pg_g + (head * (z_PV - z_in) + dpf_A) / 1000
        pPV_B = pg_g + (head * (z_PV - z_in) + dpf_B) / 1000
        set_A = p_PV + (P['p_atm_ref'] - p_atm) if pv_abs else np.full(N, p_PV)
        set_B = set_A * (1 + tol_B)
        qpvA, pvA_n = vvalve(pPV_A, set_A, Q_PV, pvA, h_open, f_min_PV)
        qpvB, pvB_n = vvalve(pPV_B, set_B, Q_PV, pvB, h_open, f_min_PV)
        oa = pvA_n & ~pvA; ob = pvB_n & ~pvB
        opens_A += oa; opens_B += ob
        first = (oa | ob) & np.isnan(t_first)
        t_first[first] = (k + 1) * dt
        pvA, pvB = pvA_n, pvB_n
        qW_act, w_open = vvalve(pa_g, p_op, Q_W, w_open, h_open, f_min_W)
        qW = qW_act * (p_a / P_REF) * (T_REF / T_h)
        blower = np.where(blower, pa_g < p_op, pa_g <= p_op - dp_bl)
        q_bl = np.where(blower, Q_bl, 0.0)
        h = dt / 3600
        n_g = n_g + (q_gen - qpvA - qpvB) * h              # downstream stop: q_out = 0
        n_a = np.maximum(n_a + (q_bl - qW) * h, 0.0)
        yCH4 = yCH4_h[i] + (yCH4_h[i + 1] - yCH4_h[i]) * fr
        yH2S = yH2S_h[i] + (yH2S_h[i + 1] - yH2S_h[i]) * fr
        v = (qpvA + qpvB) * h
        anyopen = pvA | pvB
        open_s += anyopen * dt; vent += v; ch4 += v * yCH4 * RHO_CH4_REF; h2s += v * yH2S * RHO_H2S_REF
        pt_max = np.maximum(pt_max, pPT_A)
        # cumulative values 1 h and 2 h after the first opening (operator response time)
        el = (k + 1) * dt - t_first
        m1 = el == 3600; m2 = el == 7200
        vent_1h[m1] = vent[m1]; open_1h[m1] = open_s[m1]; opens_1h[m1] = (opens_A + opens_B)[m1]
        vent_2h[m2] = vent[m2]; open_2h[m2] = open_s[m2]; opens_2h[m2] = (opens_A + opens_B)[m2]
        if (k + 1) in checkpoints:
            chk[k + 1] = dict(vent=vent.copy(), open_s=open_s.copy(), opens=(opens_A + opens_B).copy(), pt_max=pt_max.copy())
        p_g_prev = p_g; qpvA_p = qpvA; qpvB_p = qpvB
        if (k + 1) % 3600 == 0:
            print(f'  {(k + 1) // 3600} h  ({time.time() - tw:.0f} s)', flush=True)
    out = pd.DataFrame(dict(start=inp['grid'][0] + pd.to_timedelta(t0, unit='s'), start_hour=starts,
                            Q_gen_start_m3h=q_h[np.minimum(starts + 1, n_h)], p_atm_start_kPa=patm0,
                            t_holder_full_min=t_full / 60, t_first_open_min=t_first / 60,
                            opens_A=opens_A, opens_B=opens_B, open_s=open_s, vent_m3=vent, CH4_kg=ch4, H2S_kg=h2s, PT_max_kPag=pt_max,
                            vent_1h_after_open_m3=vent_1h, open_s_1h_after_open=open_1h, opens_1h_after_open=opens_1h,
                            vent_2h_after_open_m3=vent_2h, open_s_2h_after_open=open_2h, opens_2h_after_open=opens_2h))
    for s, d in chk.items():
        hh = s // 3600
        out[f'vent_{hh}h_m3'] = d['vent']; out[f'open_s_{hh}h'] = d['open_s']; out[f'opens_{hh}h'] = d['opens']; out[f'PT_max_{hh}h'] = d['pt_max']
    return out

if __name__ == '__main__':
    # python scan_stop.py <case> <f0> [f_min_PV] [pv_ref]   e.g.  scan_stop.py 1 0.85 0.0   /   scan_stop.py 1 0.85 0.10 abs
    case, f0 = int(sys.argv[1]), float(sys.argv[2])
    inp = pm.load_inputs()
    P = dict(pm.BASE); P['p_op'] = pm.CASES[case]
    suffix = ''
    if len(sys.argv) > 3 and float(sys.argv[3]) != P['f_min_PV']:
        P['f_min_PV'] = float(sys.argv[3]); suffix += f'_fmin{int(round(P["f_min_PV"] * 100))}'
    if len(sys.argv) > 4 and sys.argv[4] == 'abs':
        P['pv_ref'] = 'abs'; suffix += '_pvabs'
    qd = os.environ.get('PVRV_QUICK_DAYS')
    starts = np.arange(0, int(qd) * 24 - 6) if qd else None     # quick check mode: events starting in the first N days only
    print(f'Case {case} f0 {f0} f_min_PV {P["f_min_PV"]} pv_ref {P["pv_ref"]}', flush=True)
    out = scan(inp, P, f0, starts=starts)
    os.makedirs('out', exist_ok=True)
    out.to_csv(f'out/stop_case{case}_f{int(f0 * 100)}{suffix}.csv.gz', index=False, float_format='%.4f')
    d = out.describe().T
    print(d[['mean', 'min', '50%', 'max']].to_string())
