# -*- coding: utf-8 -*-
"""
v4.3 input series for the on/off BN, taken from the 1-s pressure model itself (package 1) so that both packages share
the same holder, the same pressure conversion and the same flare/consumer logic. The runs used are the on/off years
WITHOUT ignition failures (PFD 0): the BN models the failures and their consequences itself, so the input series must
not contain them.

    python make_onoff_inputs.py [path to pressure_model/out]

For every Case (1-3) and flare start level (100/80/60 %) it reads onoff_case{c}_f{fo}_pfd0_1min.csv.gz and writes
    fill_hourly_onoff_c{c}_f{fo}.csv    hourly mean holder fill (actual gas volume / 500 m3), stamped with the ADM1 interval end of the hour
    flare_demands_onoff_c{c}_f{fo}.csv  flare start times in seconds from the start of the year (relight inside a minute
                                        is placed at the end of the minute minus the 'on' fraction)
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'pressure_model', 'out')
ADM1 = pd.read_csv(os.path.join(HERE, '..', 'data', 'ADM1_hourly_1to9_Q200_6000m3_ffOTS1.csv'), parse_dates=['datetime_interval_end'])

def series(case, fo):
    d = pd.read_csv(os.path.join(SRC, f'onoff_case{case}_f{fo}_pfd0_1min.csv.gz'), usecols=['datetime', 'V_holder_gas_m3', 'flare_on_frac'], parse_dates=['datetime'])
    n = (len(d) // 60) * 60
    fill = (d['V_holder_gas_m3'].to_numpy()[:n] / 500.0).reshape(-1, 60).mean(1)
    on = d['flare_on_frac'].to_numpy()
    k = np.where((on[1:] > 0) & (on[:-1] == 0))[0] + 1            # minute index of each start (0-based from the first record)
    if on[0] > 0: k = np.r_[0, k]
    t_s = (k + 1) * 60.0 - on[k] * 60.0                             # minute k covers [60k, 60(k+1)) s
    return fill, t_s

if __name__ == '__main__':
    for case in (1, 2, 3):
        for fo in (100, 80, 60):
            fill, t_s = series(case, fo)
            tag = f'onoff_c{case}_f{fo}'
            pd.DataFrame({'datetime': ADM1['datetime_interval_end'].iloc[:len(fill)].to_numpy(), 'fill': np.round(fill, 4)}).to_csv(os.path.join(HERE, f'fill_hourly_{tag}.csv'), index=False)
            pd.DataFrame({'t_s': np.round(t_s, 1)}).to_csv(os.path.join(HERE, f'flare_demands_{tag}.csv'), index=False)
            print(tag, 'hours', len(fill), 'flare starts', len(t_s), 'mean fill %.3f' % fill.mean())
