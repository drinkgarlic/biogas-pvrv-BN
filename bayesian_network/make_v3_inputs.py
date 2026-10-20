# -*- coding: utf-8 -*-
"""Regenerate the v3 input series (constant consumption 60 m3/h + proportional flare 150 m3/h between 70 and 85 % fill):
fill_hourly_use60.csv and flare_demands_use60.csv, read by pv_bn.py through PARAMS['fill_series'] / ['demand_series']."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics as ph, pv_bn as bn
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(bn.ADM1_CSV, parse_dates=['datetime_interval_end']); qh = d['Q_gas_wet_m3_d_20C'].to_numpy() / 24
t, fills, starts, st = ph.operating_year(qh)            # defaults = v3 operating model
fh = fills.reshape(-1, 60).mean(1)
pd.DataFrame({'datetime': d['datetime_interval_end'].iloc[1:len(fh) + 1].to_numpy(), 'fill': np.round(fh, 4)}).to_csv(os.path.join(HERE, 'fill_hourly_use60.csv'), index=False)
pd.DataFrame({'t_s': starts}).to_csv(os.path.join(HERE, 'flare_demands_use60.csv'), index=False)
print(f'{len(starts)} flare demands, mean fill {fh.mean():.3f}, flared share {st["flared_m3"] / st["production_m3"]:.3f}')
