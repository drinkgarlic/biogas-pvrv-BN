# -*- coding: utf-8 -*-
"""on/off operation, one year: python run_onoff.py <case 1-3> <f_on 1.0|0.8|0.6> <pfd>"""
import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pressure_model as pm
os.makedirs('out', exist_ok=True)
case, f_on, pfd = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
inp = pm.load_inputs()
P = dict(pm.BASE); P['p_op'] = pm.CASES[case]; P['out_mode'] = 'onoff'; P['fill0'] = 0.5; P['pfd_flare'] = pfd
P['f_fl_on'] = f_on; P['f_fl_off'] = f_on - 0.15
grid = inp['grid']
def day(s): return int((pd.Timestamp(s) - grid[0]).days)
sec_days = sorted({day('2021-03-10'), day('2021-04-12'), day('2021-08-24')})
n_days = int(os.environ.get('PVRV_QUICK_DAYS', 10 ** 6)); sec_days = [d for d in sec_days if d < n_days] or [3]
df, ev, sec, summ = pm.run_year(inp, P, sec_days=sec_days)
tag = f'onoff_case{case}_f{int(round(f_on*100))}_pfd{int(round(pfd*100))}'
df.to_csv(f'out/{tag}_1min.csv.gz', index=False, float_format='%.5f'); ev.to_csv(f'out/{tag}_events.csv', index=False, float_format='%.4f')
sec.to_csv(f'out/{tag}_1s_days.csv.gz', index=False, float_format='%.5f')
summ = {k: (float(v) if not isinstance(v, int) else v) for k, v in summ.items()}
fill = df['V_holder_gas_m3'] / 5.0
summ.update(case=case, f_on=f_on, f_off=P['f_fl_off'], pfd=pfd, fill_mean=float(fill.mean()), fill_p5=float(fill.quantile(.05)), fill_p50=float(fill.quantile(.5)),
            fill_p95=float(fill.quantile(.95)), share_full=float((fill >= 99).mean()), share_empty=float((fill <= 1).mean()),
            PT_mean=float(df['PT101A_kPag'].mean()), PT_max=float(df['PT101A_max_kPag'].max()), PT_min=float(df['PT101A_min_kPag'].min()),
            flare_on_share=float(df['flare_on_frac'].mean()), use_on_share=float(df['use_on_frac'].mean()), prod_m3=float(df['Q_gen_m3h'].sum() / 60))
json.dump(summ, open(f'out/{tag}_summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(summ, ensure_ascii=False))
