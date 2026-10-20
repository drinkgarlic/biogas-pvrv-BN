# -*- coding: utf-8 -*-
"""100 % operation (pressure-controlled flare), one year: python run_full100.py <case> <pfd>"""
import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pressure_model as pm
os.makedirs('out', exist_ok=True)
case, pfd = int(sys.argv[1]), float(sys.argv[2])
inp = pm.load_inputs()
P = dict(pm.BASE); P['p_op'] = pm.CASES[case]; P['out_mode'] = 'const'; P['fill0'] = 0.5; P['pfd_flare'] = pfd
df, ev, sec, summ = pm.run_year(inp, P)
tag = f'full100_case{case}_pfd{int(pfd*100)}'
df.to_csv(f'out/{tag}_1min.csv.gz', index=False, float_format='%.5f'); ev.to_csv(f'out/{tag}_events.csv', index=False, float_format='%.4f')
summ = {k: (float(v) if not isinstance(v, int) else v) for k, v in summ.items()}
fill = df['V_holder_gas_m3'] / 5.0
summ.update(case=case, pfd=pfd, fill_mean=float(fill.mean()), share_full=float((fill >= 99).mean()), PT_mean=float(df['PT101A_kPag'].mean()),
            PT_max=float(df['PT101A_max_kPag'].max()), hg_max=float(df['holder_gas_kPag'].max()))
json.dump(summ, open(f'out/{tag}_summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summ, indent=1, ensure_ascii=False))
