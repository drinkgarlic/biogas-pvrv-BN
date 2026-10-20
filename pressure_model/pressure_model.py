# -*- coding: utf-8 -*-
"""
One-second pressure simulation of the Pohang food-waste biogas plant (digesters -> double-membrane gas holder -> downstream).

Physical model
  * The gas side (two digester headspaces + piping + inside the holder's inner membrane) and the air side (closed space
    between the outer and inner membrane) are each treated as an ideal gas. While the inner membrane is free the two
    absolute pressures are equal (apart from the membrane differential dp_m):
      p_abs = (P_REF/T_REF) * (n_g + n_a) / (V_hs/T_hs + V_pipe/T_pipe + V_shell/T_h)
    n_g and n_a are the gas and air inventories counted as reference-state volumes (20 degC, 101.325 kPa).
  * When the inner membrane touches the outer membrane (holder full) the gas side becomes a rigid volume; when it
    touches the floor (holder empty) the gas side is only the headspaces plus piping.
  * Gauge pressure = absolute pressure - KMA station pressure (atmospheric pressure decreases with height by rho_air*g*z).
  * Digester pressures (nozzle N1, PT-101 and PV-101 elevations) are the holder pressure plus the elevation head
    (rho_air - rho_gas)*g*dz and the pipe friction / minor losses from drawing ISO-BG-001 REV.1. The pipe flow
    includes the headspace storage term (rate of pressure change).
  * Pressure/vacuum relief valves PV-101A/B: weight-loaded (gauge-referenced) proportional valves that start to open at
    the set pressure and reach rated flow at 110 % of it, with stiction (h_open) and a minimum stable opening (f_min).
    The set pressure seen in absolute terms = atmospheric pressure + gauge setting, so it moves with the weather.
    (pv_ref = 'abs' assumes a fixed absolute set pressure for sensitivity studies.)
  * Air side: weight-loaded vent valve WV-202 (setting = operating pressure, proportional), blower B-201
    (start at p_op - 0.3, stop at p_op), double check valves.
  * Downstream: several operating modes (follow / const / stop / const_level / onoff), see BASE['out_mode'].

Inputs
  * PyADM1ODE hourly results (interval-end values) -> linear interpolation to 1 s
  * KMA ASOS hourly station pressure -> linear interpolation to 1 s

Outputs (1-min aggregates + 1-s records for selected days + event log): see run_year().
"""
import math
import os
import time
import warnings
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
ADM1_CSV = os.path.join(DATA_DIR, 'ADM1_hourly_1to9_Q200_6000m3_ffOTS1.csv')       # PyADM1ODE hourly output (interval-end values)
KMA_CSV = os.path.join(DATA_DIR, 'kma_asos138_pohang_hourly_pressure_kPa.csv')      # KMA ASOS 138 (Pohang) hourly station pressure, kPa

try:
    from numba import njit
except ImportError:                       # pure Python if numba is not available
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

P_REF, T_REF = 101.325, 293.15            # kPa, K
R_GAS = 8.314462
M_AIR = 0.028964
G = 9.80665
M_CH4, M_CO2, M_H2O, M_H2S, M_NH3, M_H2 = 0.016043, 0.044010, 0.018015, 0.034081, 0.017031, 0.002016
RHO_CH4_REF = M_CH4 * 1000 / 24.055        # kg per ref m3 (20 ℃, 1 atm)
RHO_H2S_REF = M_H2S * 1000 / 24.055

# ----------------------------------------------------------------------------- equipment and operating data (common to all cases)
BASE = dict(
    # volumes and temperatures
    V_hs=400.0,        # digester headspace, m3 per digester
    V_pipe=1.6,        # digester -> holder piping volume, m3 (drawing REV.1)
    V_shell=672.0,     # volume inside the holder outer membrane, m3
    V_max=500.0,       # maximum gas volume of the holder, m3 (inner membrane touches the outer membrane)
    T_hs=308.15,       # headspace gas 35 degC
    T_pipe=298.15,     # gas in the piping 25 degC (assumed)
    T_h=293.15,        # holder gas and air 20 degC (replace with a temperature column if available)
    T_amb=288.15,      # ambient air 15 degC (air density for the elevation head; replace with a temperature column if available)
    dp_m=0.0,          # pressure difference across the inner membrane (gas - air), kPa
    fill0=0.50,        # initial holder fill fraction
    # air-side control
    p_op=4.0,          # operating pressure = setting of the weight-loaded vent valve, kPag (changed per case)
    Q_W=800.0,         # vent valve rated flow (at 110 % of the setting), actual m3/h
    f_min_W=0.10,      # vent valve minimum stable opening
    dp_bl=0.3,         # blower start = p_op - dp_bl, stop = p_op
    Q_bl=400.0,        # blower, m3/h
    # pressure/vacuum relief valves
    p_PV=5.0,          # pressure setting, kPag
    Q_PV=1410.0,       # rated flow (at 110 % of the setting), m3/h
    tol_B=0.01,        # PV-101B setting tolerance (+1 %) -> A opens first
    f_min_PV=0.10,     # minimum stable opening
    h_open=0.01,       # stiction: must exceed the setting by 1 % to start opening
    p_vac=-0.3,        # vacuum setting, kPag
    Q_vac=400.0,       # vacuum-side rated flow (at 110 % of the setting), m3/h
    pv_ref='gauge',    # 'gauge': weight-loaded (gauge-referenced) / 'abs': fixed absolute setting (sensitivity)
    p_atm_ref=101.325, # reference atmospheric pressure of the setting when pv_ref='abs'
    # elevations (EL, m)
    z_in=0.6, z_N1=17.8, z_PT=18.1, z_PV=18.25,
    # downstream
    out_mode='follow', # 'follow': Q_out = Q_gen / 'const': Q_out = Q_const / 'stop': Q_out = 0
    Q_const=60.0,
    tau_pipe=6.0,      # first-order lag of the pipe pressure loss, s (pipe gas inertia / compressibility; avoids exaggerated flow reversals)
    # flare in the constant-consumption modes (out_mode 'const' / 'const_level')
    Q_flare=150.0,     # flare capacity, m3/h
    dp_fl_on=0.5,      # 'const': pressure-based start = p_op + dp_fl_on (holder gas gauge)
    dp_fl_off=0.1,     # 'const': stop = p_op + dp_fl_off
    f_fl_on=0.85,      # 'const_level': fill-level start
    f_fl_off=0.70,     # 'const_level': stop (proportional control, minimum 25 %)
    L_stop=0.10,       # 'onoff': consumer stops at this fill fraction (holder low-low)
    L_resume=0.30,     # 'onoff': consumer restarts at this fill fraction
    pfd_flare=0.0,     # probability of ignition failure per flare start demand (0 = perfect)
    t_lockout=1800.0,  # lockout time before a retry after a failure, s
    seed=12345,        # random seed for the failure draws
)
CASES = {1: 4.0, 2: 3.0, 3: 2.0}            # operating pressure, kPag

# ----------------------------------------------------------------------------- piping (ISO-BG-001 REV.1)
# (inner diameter m, length m, roughness m, sum of K, fully turbulent fT)   K from Crane TP-410, K = n * fT
def _K(fT, **n):
    tab = dict(ell=14, teeb=60, teer=20, bfv=45)
    return sum(v * tab[k] * fT if k in tab else v for k, v in n.items())
PIPES = dict(
    brA=(0.1082, 27.6, 4.5e-5, _K(0.017, ent=0.5, ell=2, teeb=2, bfv=1, exp=(1 - (0.1082 / 0.1615) ** 2) ** 2)),
    brB=(0.1082, 27.9, 4.5e-5, _K(0.017, ent=0.5, ell=2, teeb=2, bfv=1, exp=(1 - (0.1082 / 0.1615) ** 2) ** 2)),
    hd1=(0.1615, 24.0, 4.5e-5, 0.0),                                     # header E0 -> E24 (flow of branch A only)
    hd2=(0.1615, 20.5, 4.5e-5, _K(0.015, teer=1, bfv=1, ell=1)),         # header E24 -> E44 + drop (A + B)
    pe=(0.1308, 14.0, 7.0e-6, _K(0.015, ell=2, teer=1, exit=1.0)),       # buried PE pipe -> holder inlet
)
PIPE_ARR = np.array([PIPES[k] for k in ('brA', 'brB', 'hd1', 'hd2', 'pe')])

@njit(cache=True)
def dp_pipe(Q_m3h, D, L, eps, Ksum, rho, mu):
    """Darcy-Weisbach + minor losses, Pa (sign = sign of the flow). Laminar 64/Re, turbulent Swamee-Jain, linear blend 2000-4000."""
    if Q_m3h == 0.0:
        return 0.0
    q = abs(Q_m3h) / 3600.0
    v = q / (math.pi * D * D / 4.0)
    Re = rho * v * D / mu
    f_t = 0.25 / math.log10(eps / (3.7 * D) + 5.74 / Re ** 0.9) ** 2
    if Re < 2000.0:
        f = 64.0 / Re
    elif Re < 4000.0:
        w = (Re - 2000.0) / 2000.0
        f = (1 - w) * 64.0 / Re + w * f_t
    else:
        f = f_t
    dp = (f * L / D + Ksum) * rho * v * v / 2.0
    return dp if Q_m3h > 0 else -dp

@njit(cache=True)
def valve_flow(p, p_set, Q_R, was_open, h_open, f_min):
    """Weight-loaded proportional valve. Returns (flow, open flag). p and p_set share the same sign convention (pressure: gauge, vacuum: -gauge)."""
    if not was_open:
        if p < p_set * (1.0 + h_open):
            return 0.0, False
    if p <= p_set:
        return 0.0, False
    if p <= 1.1 * p_set:
        q = Q_R * (p - p_set) / (0.1 * p_set)
    else:
        q = Q_R * math.sqrt(p / (1.1 * p_set))
    if q < f_min * Q_R:
        q = f_min * Q_R
    return q, True

@njit(cache=True)
def kernel(n_sec, dt, t0_off,
           q_h, patm_h, M_h, mu_h, yCH4_h, yH2S_h, Tamb_h, Th_h,
           pipes, prm, rec_every, rec, ev, ev_n, sec_days, sec_rec):
    """prm: parameter array (order below). rec: 1-min aggregates [n_rec, NR]. ev: event log [cap, 3] (t, code, value).
    sec_days: integer array of days (counted from 0) to record at 1 s, sec_rec: [len(sec_days)*86400, NS]"""
    (V_hs, V_pipe, V_shell, V_max, T_hs, T_pipe, dp_m, fill0,
     p_op, Q_W, f_min_W, dp_bl, Q_bl, p_PV, Q_PV, tol_B, f_min_PV, h_open, p_vac, Q_vac,
     pv_abs, p_atm_ref, z_in, z_N1, z_PT, z_PV, out_mode, Q_const, n_g0, n_a0, tau_pipe,
     Q_flare, dp_fl_on, dp_fl_off, f_fl_on, f_fl_off, pfd_fl, t_lock, seed, L_stop, L_resume) = (
        prm[0], prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7], prm[8], prm[9], prm[10], prm[11],
        prm[12], prm[13], prm[14], prm[15], prm[16], prm[17], prm[18], prm[19], prm[20], prm[21], prm[22],
        prm[23], prm[24], prm[25], prm[26], prm[27], prm[28], prm[29], prm[30], prm[31], prm[32], prm[33], prm[34], prm[35], prm[36], prm[37], prm[38], prm[39], prm[40])
    c = P_REF / T_REF
    a_hs = V_hs * 2.0 / T_hs
    a_hs1 = V_hs / T_hs
    a_pipe = V_pipe / T_pipe
    n_g = n_g0; n_a = n_a0
    blower = False; w_open = False; flare = False; n_flare = 0; n_fail = 0; lock_until = -1.0; rng = int(seed)
    use_on = True; n_use_stop = 0
    pvA_open = False; pvB_open = False; vacA_open = False; vacB_open = False
    p_g_prev = -1.0
    qpvA_prev = 0.0; qpvB_prev = 0.0; qvacA_prev = 0.0; qvacB_prev = 0.0
    dpfA_f = -1e9; dpfB_f = -1e9
    lam = dt / tau_pipe if tau_pipe > 0 else 1.0
    # accumulators
    cum = np.zeros(12)   # 0 PV-A open s, 1 PV-B open s, 2 vented ref m3, 3 CH4 kg, 4 H2S kg, 5 blower s, 6 WV open s, 7 WV vented air m3, 8 blower air m3, 9 vacuum air in m3, 10 PV-A openings, 11 PV-B openings
    nev = 0
    NR = rec.shape[1]
    acc = np.zeros(NR)
    mn = 1e9; mx = -1e9; cnt = 0; k_rec = 0
    k_sec = 0
    n_sd = sec_days.shape[0]
    sd_i = 0
    for k in range(n_sec):
        t = t0_off + (k + 1) * dt                     # interval-end time, s
        i = int(t // 3600.0)
        if i >= q_h.shape[0] - 1:
            i = q_h.shape[0] - 2
        fr = t / 3600.0 - i
        q_gen = q_h[i] + (q_h[i + 1] - q_h[i]) * fr
        p_atm = patm_h[i] + (patm_h[i + 1] - patm_h[i]) * fr
        M = M_h[i] + (M_h[i + 1] - M_h[i]) * fr
        mu = mu_h[i] + (mu_h[i + 1] - mu_h[i]) * fr
        T_amb = Tamb_h[i] + (Tamb_h[i + 1] - Tamb_h[i]) * fr
        T_h = Th_h[i] + (Th_h[i + 1] - Th_h[i]) * fr
        a_sh = V_shell / T_h
        # ---- pressures (pV = nRT, absolute), referenced to the holder inlet elevation
        X = a_hs + a_pipe + a_sh
        Y = n_a * c
        if dp_m == 0.0:
            p_a = (n_g * c + Y) / X
        else:
            B = dp_m * X - Y - n_g * c
            p_a = (-B + math.sqrt(B * B + 4.0 * X * dp_m * Y)) / (2.0 * X)
        V_gh = V_shell - Y * T_h / p_a
        state = 0
        if V_gh >= V_max:
            state = 1                                    # holder full: gas side rigid
            V_gh = V_max
            p_g = n_g * c / (a_hs + a_pipe + V_max / T_h)
            p_a = Y * T_h / (V_shell - V_max)
        elif V_gh <= 0.0:
            state = -1                                   # holder empty
            V_gh = 0.0
            p_g = n_g * c / (a_hs + a_pipe)
            p_a = Y * T_h / V_shell
        else:
            p_g = p_a + dp_m
        rho_air = p_atm * 1000.0 * M_AIR / (R_GAS * T_amb)
        rho_g = p_g * 1000.0 * M / (R_GAS * T_pipe)
        p_atm_in = p_atm - rho_air * G * z_in / 1000.0
        pg_g = p_g - p_atm_in                            # holder gas, gauge
        pa_g = p_a - p_atm_in                            # holder air, gauge
        # ---- pipe flows (including headspace storage) and pressure losses
        dpdt = 0.0 if p_g_prev < 0 else (p_g - p_g_prev) / dt
        q_stor = a_hs1 / c * dpdt * 3600.0               # storage flow of one headspace, ref m3/h
        # (PV / vacuum flows are decided below, so the previous-step values are used -> local variables)
        Q_A = q_gen * 0.5 - q_stor - qpvA_prev + qvacA_prev
        Q_B = q_gen * 0.5 - q_stor - qpvB_prev + qvacB_prev
        Q_T = Q_A + Q_B
        dp_common = (dp_pipe(Q_T, pipes[3, 0], pipes[3, 1], pipes[3, 2], pipes[3, 3], rho_g, mu)
                     + dp_pipe(Q_T, pipes[4, 0], pipes[4, 1], pipes[4, 2], pipes[4, 3], rho_g, mu))
        dpf_A = (dp_pipe(Q_A, pipes[0, 0], pipes[0, 1], pipes[0, 2], pipes[0, 3], rho_g, mu)
                 + dp_pipe(Q_A, pipes[2, 0], pipes[2, 1], pipes[2, 2], pipes[2, 3], rho_g, mu) + dp_common)
        dpf_B = dp_pipe(Q_B, pipes[1, 0], pipes[1, 1], pipes[1, 2], pipes[1, 3], rho_g, mu) + dp_common
        if dpfA_f < -1e8:
            dpfA_f = dpf_A; dpfB_f = dpf_B
        dpfA_f += (dpf_A - dpfA_f) * lam; dpfB_f += (dpf_B - dpfB_f) * lam
        dpf_A = dpfA_f; dpf_B = dpfB_f
        head = (rho_air - rho_g) * G                     # Pa/m, elevation gradient of the gauge pressure
        pPT_A = pg_g + (head * (z_PT - z_in) + dpf_A) / 1000.0
        pPT_B = pg_g + (head * (z_PT - z_in) + dpf_B) / 1000.0
        pPV_A = pg_g + (head * (z_PV - z_in) + dpf_A) / 1000.0
        pPV_B = pg_g + (head * (z_PV - z_in) + dpf_B) / 1000.0
        pN1_A = p_g - (rho_g * G * (z_N1 - z_in) - dpf_A) / 1000.0   # nozzle absolute pressure
        pN1_B = p_g - (rho_g * G * (z_N1 - z_in) - dpf_B) / 1000.0
        # ---- relief valve set pressure (gauge-referenced / fixed absolute)
        set_A = p_PV if pv_abs == 0.0 else p_PV + (p_atm_ref - p_atm)
        set_B = set_A * (1.0 + tol_B)
        set_abs_A = set_A + p_atm - rho_air * G * z_PV / 1000.0           # the setting seen as an absolute pressure at the PV elevation
        qpvA, pvA_open_new = valve_flow(pPV_A, set_A, Q_PV, pvA_open, h_open, f_min_PV)
        qpvB, pvB_open_new = valve_flow(pPV_B, set_B, Q_PV, pvB_open, h_open, f_min_PV)
        qvacA, vacA_open_new = valve_flow(-pPV_A, -p_vac, Q_vac, vacA_open, h_open, f_min_PV)
        qvacB, vacB_open_new = valve_flow(-pPV_B, -p_vac * (1.0 + tol_B), Q_vac, vacB_open, h_open, f_min_PV)
        if pvA_open_new and not pvA_open:
            cum[10] += 1
            if nev < ev.shape[0]: ev[nev, 0] = t; ev[nev, 1] = 1; ev[nev, 2] = pPT_A; nev += 1
        if (not pvA_open_new) and pvA_open:
            if nev < ev.shape[0]: ev[nev, 0] = t; ev[nev, 1] = 2; ev[nev, 2] = pPT_A; nev += 1
        if pvB_open_new and not pvB_open:
            cum[11] += 1
            if nev < ev.shape[0]: ev[nev, 0] = t; ev[nev, 1] = 3; ev[nev, 2] = pPT_B; nev += 1
        if (not pvB_open_new) and pvB_open:
            if nev < ev.shape[0]: ev[nev, 0] = t; ev[nev, 1] = 4; ev[nev, 2] = pPT_B; nev += 1
        pvA_open = pvA_open_new; pvB_open = pvB_open_new; vacA_open = vacA_open_new; vacB_open = vacB_open_new
        # ---- air side: weight-loaded vent valve, blower
        qW_act, w_open_new = valve_flow(pa_g, p_op, Q_W, w_open, h_open, f_min_W)
        qW = qW_act * (p_a / P_REF) * (T_REF / T_h)      # actual -> reference m3/h
        if w_open_new and not w_open:
            if nev < ev.shape[0]: ev[nev, 0] = t; ev[nev, 1] = 5; ev[nev, 2] = pa_g; nev += 1
        if (not w_open_new) and w_open:
            if nev < ev.shape[0]: ev[nev, 0] = t; ev[nev, 1] = 6; ev[nev, 2] = pa_g; nev += 1
        w_open = w_open_new
        if blower:
            if pa_g >= p_op:
                blower = False
                if nev < ev.shape[0]: ev[nev, 0] = t; ev[nev, 1] = 8; ev[nev, 2] = pa_g; nev += 1
        else:
            if pa_g <= p_op - dp_bl:
                blower = True
                if nev < ev.shape[0]: ev[nev, 0] = t; ev[nev, 1] = 7; ev[nev, 2] = pa_g; nev += 1
        q_bl = Q_bl if blower else 0.0
        # ---- downstream
        if out_mode == 0.0:
            q_out = q_gen
        elif out_mode == 1.0:                                  # constant consumption + pressure-controlled flare (no fill-level signal)
            if flare:
                if pg_g <= p_op + dp_fl_off: flare = False
            elif pg_g >= p_op + dp_fl_on and t >= lock_until:
                rng = (1103515245 * rng + 12345) % 2147483648
                if rng / 2147483648.0 < pfd_fl:
                    n_fail += 1; lock_until = t + t_lock          # ignition failure: lockout, then retry
                else:
                    flare = True; n_flare += 1
            q_out = (Q_const if (state != -1) else 0.0) + (Q_flare if flare else 0.0)
        elif out_mode == 3.0:                                  # constant consumption + fill-level flare (proportional)
            fillr = V_gh / V_max
            if flare:
                if fillr <= f_fl_off: flare = False
            elif fillr >= f_fl_on and t >= lock_until:
                rng = (1103515245 * rng + 12345) % 2147483648
                if rng / 2147483648.0 < pfd_fl:
                    n_fail += 1; lock_until = t + t_lock
                else:
                    flare = True; n_flare += 1
            q_fl = 0.0
            if flare:
                q_fl = Q_flare * (fillr - f_fl_off) / (0.95 - f_fl_off)
                if q_fl < 0.25 * Q_flare: q_fl = 0.25 * Q_flare
                if q_fl > Q_flare: q_fl = Q_flare
            q_out = (Q_const if (state != -1) else 0.0) + q_fl
        elif out_mode == 4.0:                                  # consumer on/off (fill low-low stop) + flare on/off (fill-level start)
            fillr = V_gh / V_max
            if use_on:
                if fillr <= L_stop: use_on = False; n_use_stop += 1
            elif fillr >= L_resume:
                use_on = True
            if flare:
                if fillr <= f_fl_off: flare = False
            elif fillr >= f_fl_on and t >= lock_until:
                rng = (1103515245 * rng + 12345) % 2147483648
                if rng / 2147483648.0 < pfd_fl:
                    n_fail += 1; lock_until = t + t_lock
                else:
                    flare = True; n_flare += 1
            q_out = (Q_const if use_on else 0.0) + (Q_flare if flare else 0.0)
        else:
            q_out = 0.0
        # ---- state update (reference m3)
        h = dt / 3600.0
        n_g += (q_gen - q_out - qpvA - qpvB + qvacA + qvacB) * h
        n_a += (q_bl - qW) * h
        if n_a < 0.0:
            n_a = 0.0
        # ---- accumulate
        yCH4 = yCH4_h[i] + (yCH4_h[i + 1] - yCH4_h[i]) * fr
        yH2S = yH2S_h[i] + (yH2S_h[i + 1] - yH2S_h[i]) * fr
        vent = (qpvA + qpvB) * h
        cum[0] += dt if pvA_open else 0.0
        cum[1] += dt if pvB_open else 0.0
        cum[2] += vent
        cum[3] += vent * yCH4 * RHO_CH4_REF
        cum[4] += vent * yH2S * RHO_H2S_REF
        cum[5] += dt if blower else 0.0
        cum[6] += dt if w_open else 0.0
        cum[7] += qW * h
        cum[8] += q_bl * h
        cum[9] += (qvacA + qvacB) * h
        # ---- 1-min aggregation
        acc[0] += p_atm; acc[1] += pN1_A; acc[2] += set_abs_A; acc[3] += pPT_A; acc[4] += pPT_B
        acc[5] += pg_g; acc[6] += pa_g; acc[7] += V_gh; acc[8] += q_gen; acc[9] += dpf_A; acc[10] += head * (z_PT - z_in)
        acc[11] += rho_g; acc[12] += (1.0 if blower else 0.0); acc[13] += (1.0 if w_open else 0.0)
        acc[14] += (1.0 if (pvA_open or pvB_open) else 0.0); acc[15] += qpvA + qpvB; acc[16] += p_g; acc[17] += p_a
        acc[18] += Q_A; acc[19] += q_out
        acc[20] += (1.0 if flare else 0.0); acc[21] += (1.0 if use_on else 0.0)
        if pPT_A < mn: mn = pPT_A
        if pPT_A > mx: mx = pPT_A
        cnt += 1
        if cnt == rec_every:
            for j in range(22):
                rec[k_rec, j] = acc[j] / cnt
                acc[j] = 0.0
            rec[k_rec, 22] = mn; rec[k_rec, 23] = mx
            rec[k_rec, 24] = cum[2]; rec[k_rec, 25] = cum[0] + cum[1]
            k_rec += 1; cnt = 0; mn = 1e9; mx = -1e9
        # ---- 1-s record for the selected days
        if sd_i < n_sd:
            d = int((t - 1) // 86400)
            if d == sec_days[sd_i]:
                sec_rec[k_sec, 0] = p_atm; sec_rec[k_sec, 1] = pN1_A; sec_rec[k_sec, 2] = set_abs_A
                sec_rec[k_sec, 3] = pPT_A; sec_rec[k_sec, 4] = pPT_B; sec_rec[k_sec, 5] = pg_g; sec_rec[k_sec, 6] = pa_g
                sec_rec[k_sec, 7] = V_gh; sec_rec[k_sec, 8] = q_gen; sec_rec[k_sec, 9] = Q_A; sec_rec[k_sec, 10] = dpf_A
                sec_rec[k_sec, 11] = 1.0 if blower else 0.0; sec_rec[k_sec, 12] = qW_act; sec_rec[k_sec, 13] = qpvA + qpvB
                sec_rec[k_sec, 14] = p_g; sec_rec[k_sec, 15] = rho_g
                k_sec += 1
            elif d > sec_days[sd_i]:
                sd_i += 1
        p_g_prev = p_g
        qpvA_prev = qpvA; qpvB_prev = qpvB; qvacA_prev = qvacA; qvacB_prev = qvacB
    ev_n[0] = nev
    # final state
    cum_out = np.zeros(19)
    for j in range(12):
        cum_out[j] = cum[j]
    cum_out[16] = n_flare; cum_out[17] = n_fail; cum_out[18] = n_use_stop
    cum_out[12] = n_g; cum_out[13] = n_a; cum_out[14] = V_gh; cum_out[15] = k_rec
    return cum_out

REC_COLS = ['p_atm_kPa', 'p_N1A_abs_kPa', 'p_PVset_abs_kPa', 'PT101A_kPag', 'PT101B_kPag', 'holder_gas_kPag', 'holder_air_kPag',
            'V_holder_gas_m3', 'Q_gen_m3h', 'dp_fric_A_Pa', 'dp_head_PT_Pa', 'rho_gas_kgm3', 'blower_frac', 'WV_open_frac',
            'PV_open_frac', 'Q_PV_m3h', 'p_holder_gas_abs_kPa', 'p_holder_air_abs_kPa', 'Q_pipeA_m3h', 'Q_out_m3h',
            'flare_on_frac', 'use_on_frac', 'PT101A_min_kPag', 'PT101A_max_kPag', 'PV_vented_cum_m3', 'PV_open_cum_s']
SEC_COLS = ['p_atm_kPa', 'p_N1A_abs_kPa', 'p_PVset_abs_kPa', 'PT101A_kPag', 'PT101B_kPag', 'holder_gas_kPag', 'holder_air_kPag',
            'V_holder_gas_m3', 'Q_gen_m3h', 'Q_pipeA_m3h', 'dp_fric_A_Pa', 'blower_on', 'Q_WV_m3h', 'Q_PV_m3h', 'p_holder_gas_abs_kPa', 'rho_gas_kgm3']
EV_NAMES = {1: 'PV-A open', 2: 'PV-A close', 3: 'PV-B open', 4: 'PV-B close', 5: 'WV open', 6: 'WV close', 7: 'blower start', 8: 'blower stop'}

# ----------------------------------------------------------------------------- input preparation
def load_inputs(adm1_csv=ADM1_CSV, kma_csv=KMA_CSV, t_start='2020-11-01 00:00', t_end='2021-11-01 00:00', T_pipe=298.15):
    """Hourly knot arrays (length 8761: the t=0 knot + 8760 interval ends). The first knot repeats the first record."""
    a = pd.read_csv(adm1_csv, parse_dates=['datetime_interval_end'])
    grid = pd.date_range(t_start, t_end, freq='h')
    a = a.set_index('datetime_interval_end').reindex(grid)
    a.iloc[0] = a.iloc[1]
    a = a.interpolate(limit_direction='both')
    if str(kma_csv).endswith('.parquet'):
        k = pd.read_parquet(kma_csv)
    else:
        try:
            k = pd.read_csv(kma_csv)                      # UTF-8 file shipped in data/ (datetime, p_kPa)
        except UnicodeDecodeError:
            k = pd.read_csv(kma_csv, encoding='cp949')     # raw KMA download
    k = k.iloc[:, -2:]; k.columns = ['dt', 'p']
    k['dt'] = pd.to_datetime(k['dt'])
    p = k.set_index('dt')['p'].reindex(grid).interpolate(method='time', limit_area='inside').ffill().bfill()
    y = {s: a[f'{s}_wet_pct'].to_numpy() / 100 for s in ('CH4', 'CO2', 'H2O')}
    yH2S = a['H2S_gas_ppmv_eq'].to_numpy() * 1e-6
    yNH3 = a['NH3_gas_ppmv_eq'].to_numpy() * 1e-6
    yH2 = a['H2_dry_ppm'].to_numpy() * 1e-6 * (1 - y['H2O'])
    M = y['CH4'] * M_CH4 + y['CO2'] * M_CO2 + y['H2O'] * M_H2O + yH2S * M_H2S + yNH3 * M_NH3 + yH2 * M_H2
    # viscosity by Herning-Zipperer (component viscosities near T_pipe, Pa s)
    mu_i = {'CH4': 1.10e-5, 'CO2': 1.49e-5, 'H2O': 0.99e-5}
    Mi = {'CH4': M_CH4, 'CO2': M_CO2, 'H2O': M_H2O}
    num = sum(y[s] * mu_i[s] * math.sqrt(Mi[s]) for s in y); den = sum(y[s] * math.sqrt(Mi[s]) for s in y)
    mu = num / den
    return dict(grid=grid, q_h=a['Q_gas_wet_m3_d_20C'].to_numpy() / 24.0, patm_h=p.to_numpy(), M_h=M, mu_h=mu,
                yCH4_h=y['CH4'], yH2S_h=yH2S, comp=a[['CH4_wet_pct', 'CO2_wet_pct', 'H2O_wet_pct', 'H2S_gas_ppmv_eq']])

def _prm(P, inp, i0=0):
    c = P_REF / T_REF
    p0 = inp['patm_h'][i0] - 1.2 * G * P['z_in'] / 1000.0 + P['p_op']     # atmospheric pressure at the holder inlet elevation + operating pressure
    V_g0 = P['fill0'] * P['V_max']
    n_g0 = p0 * (2 * P['V_hs'] / P['T_hs'] + P['V_pipe'] / P['T_pipe'] + V_g0 / P['T_h']) / c
    n_a0 = p0 * (P['V_shell'] - V_g0) / P['T_h'] / c
    return np.array([P['V_hs'], P['V_pipe'], P['V_shell'], P['V_max'], P['T_hs'], P['T_pipe'], P['dp_m'], P['fill0'],
                     P['p_op'], P['Q_W'], P['f_min_W'], P['dp_bl'], P['Q_bl'], P['p_PV'], P['Q_PV'], P['tol_B'], P['f_min_PV'],
                     P['h_open'], P['p_vac'], P['Q_vac'], 1.0 if P['pv_ref'] == 'abs' else 0.0, P['p_atm_ref'],
                     P['z_in'], P['z_N1'], P['z_PT'], P['z_PV'], {'follow': 0.0, 'const': 1.0, 'stop': 2.0, 'const_level': 3.0, 'onoff': 4.0}[P['out_mode']],
                     P['Q_const'], n_g0, n_a0, P['tau_pipe'], P['Q_flare'], P['dp_fl_on'], P['dp_fl_off'], P['f_fl_on'], P['f_fl_off'], P['pfd_flare'], P['t_lockout'], P['seed'], P['L_stop'], P['L_resume']], dtype=float)

def run_year(inp, P, sec_days=(), rec_every=60, n_sec=None, t0_off=0.0, ev_cap=400000, verbose=True):
    n = len(inp['q_h']) - 1
    if n_sec is None:
        qd = os.environ.get('PVRV_QUICK_DAYS')                 # quick check mode: simulate only the first N days
        n_sec = int(min(n, int(qd) * 24) * 3600) if qd else int(n * 3600)
    else:
        n_sec = int(n_sec)
    rec = np.zeros((n_sec // rec_every + 1, len(REC_COLS)))
    ev = np.zeros((ev_cap, 3)); ev_n = np.zeros(1, dtype=np.int64)
    sd = np.array(sorted(sec_days), dtype=np.int64)
    sec_rec = np.zeros((len(sd) * 86400, len(SEC_COLS)))
    Tamb = np.full(n + 1, P['T_amb']) if 'Tamb_h' not in inp else inp['Tamb_h']
    Th = np.full(n + 1, P['T_h']) if 'Th_h' not in inp else inp['Th_h']
    t = time.time()
    args = (n_sec, 1.0, t0_off, inp['q_h'], inp['patm_h'], inp['M_h'], inp['mu_h'], inp['yCH4_h'], inp['yH2S_h'],
            Tamb, Th, PIPE_ARR, _prm(P, inp, int(t0_off // 3600)), rec_every, rec, ev, ev_n, sd, sec_rec)
    try:
        cum = kernel(*args)
    except Exception as e:                                   # numba compilation problem -> pure Python (slow but identical)
        if hasattr(kernel, 'py_func'):
            warnings.warn(f'numba kernel failed ({type(e).__name__}: {e}); falling back to pure Python')
            cum = kernel.py_func(*args)
        else:
            raise
    if verbose:
        print(f'  {n_sec} steps, {time.time() - t:.0f} s', flush=True)
    k_rec = int(cum[15])
    df = pd.DataFrame(rec[:k_rec], columns=REC_COLS)
    df.insert(0, 'datetime', inp['grid'][0] + pd.to_timedelta(t0_off + (np.arange(k_rec) + 1) * rec_every, unit='s'))
    evd = pd.DataFrame(ev[:ev_n[0]], columns=['t_s', 'code', 'p_kPag'])
    evd['datetime'] = inp['grid'][0] + pd.to_timedelta(evd['t_s'], unit='s')
    evd['event'] = evd['code'].map(EV_NAMES)
    sec = pd.DataFrame(sec_rec[:len(sd) * 86400], columns=SEC_COLS)
    if len(sd):
        tt = np.concatenate([d * 86400 + np.arange(1, 86401) for d in sd])
        sec.insert(0, 'datetime', inp['grid'][0] + pd.to_timedelta(tt, unit='s'))
    summary = dict(PV_A_open_s=cum[0], PV_B_open_s=cum[1], PV_vented_m3=cum[2], PV_CH4_kg=cum[3], PV_H2S_kg=cum[4],
                   blower_run_s=cum[5], WV_open_s=cum[6], WV_vented_air_m3=cum[7], blower_air_m3=cum[8], vac_air_in_m3=cum[9],
                   PV_A_opens=int(cum[10]), PV_B_opens=int(cum[11]), n_g_end=cum[12], n_a_end=cum[13], V_gh_end=cum[14],
                   blower_starts=int((evd['code'] == 7).sum()), WV_opens=int((evd['code'] == 5).sum()), flare_starts=int(cum[16]), flare_fails=int(cum[17]), use_stops=int(cum[18]))
    return df, evd, sec, summary
