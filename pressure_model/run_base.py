# -*- coding: utf-8 -*-
"""Base operation (downstream consumes all gas), one year at 1 s, Cases 1-3.  Usage: python run_base.py <case>"""
import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pressure_model as pm

case = int(sys.argv[1])
inp = pm.load_inputs()
grid = inp['grid']
def day(s): return int((pd.Timestamp(s) - grid[0]).days)
q_day = pd.Series(inp['q_h'][1:], index=grid[1:]).resample('D').mean()
d_max = q_day.idxmax()
sec_days = sorted({day('2021-04-12'), day('2021-04-13'), day('2021-08-23'), day('2021-08-24'), day(d_max), day('2021-01-15')})
n_days = int(os.environ.get('PVRV_QUICK_DAYS', 10 ** 6)); sec_days = [d for d in sec_days if d < n_days] or [3]
P = dict(pm.BASE); P['p_op'] = pm.CASES[case]
print(f'Case {case}: p_op {P["p_op"]} kPag, 1-s days {sec_days}', flush=True)
df, ev, sec, summ = pm.run_year(inp, P, sec_days=sec_days)
os.makedirs('out', exist_ok=True)
df.to_csv(f'out/base_case{case}_1min.csv.gz', index=False, float_format='%.5f')
ev.to_csv(f'out/base_case{case}_events.csv', index=False, float_format='%.4f')
sec.to_csv(f'out/base_case{case}_1s_selected_days.csv.gz', index=False, float_format='%.5f')
summ = {k: (float(v) if not isinstance(v, int) else v) for k, v in summ.items()}
summ.update(case=case, p_op=P['p_op'], sec_days=[str((grid[0] + pd.Timedelta(days=d)).date()) for d in sec_days])
json.dump(summ, open(f'out/base_case{case}_summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False), flush=True)
