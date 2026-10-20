# -*- coding: utf-8 -*-
"""
on/off operation, 9-case BN (v4): operating-pressure Cases 1/2/3 x flare-start fill levels 100/80/60 %.
  1) input series: physics.operating_year with on/off logic (consumer fixed at 60 with low-low stop 10/30 %, flare 150 on/off, stop = start - 15 %-points)
  2) demand branch rise rate q - Q_use, automatic retry every 30 min (at most 4, each succeeding with 0.95) -> operator response R
  3) no pressure-based flare start -> no backup in the stuck-vent-valve branch
Variants computed as sensitivities: hard30 (30 % of failures are hard faults a retry cannot clear), manual (no automatic retry, v3 style)
"""
import sys, os, copy, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pv_bn as bn, physics as ph

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
d = pd.read_csv(bn.ADM1_CSV, parse_dates=['datetime_interval_end'])
qh = d['Q_gas_wet_m3_d_20C'].to_numpy() / 24

# ---------------------------------------------------------------- 1) input series
series_stats = {}
for fo in (1.0, 0.8, 0.6):
    tag = f'onoff_f{int(round(fo*100))}'
    t, fills, starts, st = ph.operating_year(qh, use=60.0, Qf=150.0, fmin=1.0, on=fo, off=fo - 0.15, L_derate=0.0, L_stop=0.10, L_resume=0.30)
    fh = fills.reshape(-1, 60).mean(1)
    pd.DataFrame({'datetime': d['datetime_interval_end'].iloc[1:len(fh) + 1].to_numpy(), 'fill': np.round(fh, 4)}).to_csv(os.path.join(HERE, f'fill_hourly_{tag}.csv'), index=False)
    pd.DataFrame({'t_s': starts}).to_csv(os.path.join(HERE, f'flare_demands_{tag}.csv'), index=False)
    series_stats[tag] = dict(f_on=fo, flare_starts=int(st['flare_starts']), flare_share=st['flared_m3'] / st['production_m3'], stop_h=st['stop_h'],
                             fill_mean=float(fh.mean()), fill_p5=float(np.percentile(fh, 5)), fill_p95=float(np.percentile(fh, 95)), share_full=float((fh >= 0.995).mean()))
    print(tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in series_stats[tag].items()})
json.dump(series_stats, open(os.path.join(HERE, 'out', 'bn_onoff_series.json'), 'w'), indent=1)

# ---------------------------------------------------------------- 2) 9 cases x {automatic retry 30 min, hard faults 30 %, no retry}
rows = []; allsum = {}
for fo in (1.0, 0.8, 0.6):
    tag = f'onoff_f{int(round(fo*100))}'
    for variant, t_retry, p_hard in (('retry30', 0.5, 0.0), ('hard30', 0.5, 0.3), ('manual', 0.0, 0.0)):
        prm = copy.deepcopy(bn.PARAMS)
        prm['f_on']['v'] = fo; prm['t_retry_h']['v'] = t_retry; prm['n_retry']['v'] = 4; prm['p_hard']['v'] = p_hard; prm['pressure_backup']['v'] = False
        prm['fill_series']['v'] = f'fill_hourly_{tag}.csv'; prm['demand_series']['v'] = f'flare_demands_{tag}.csv'
        inp = bn.load_inputs(prm=prm)
        for case in (1, 2, 3):
            df = bn.run_year(case, inp, prm, occupancy=(variant == 'retry30'))
            s = bn.summarize(df, prm)
            key = f'case{case}_f{int(round(fo*100))}_{variant}'
            allsum[key] = s
            if variant == 'retry30':
                df.to_csv(os.path.join(HERE, 'out', f'bn_hourly_{key}.csv.gz'), index=False, float_format='%.6g')
                bn.cpt_tables(case, inp, df, prm).to_csv(os.path.join(HERE, 'out', f'bn_cpt_{key}.csv'), index=False, float_format='%.5f')
            rows.append(dict(case=case, f_on=fo, variant=variant, demands=int(inp['demand'].sum()),
                             open_stop=s['stop']['openings_per_yr'], open_demand=s['demand']['openings_per_yr'], open_foam=s['foam']['openings_per_yr'],
                             open_block=s['block']['openings_per_yr'], open_wv=s['wv']['openings_per_yr'], openings=s['total']['openings_per_yr'],
                             P_at_least_one=s['total']['P_at_least_one_per_yr'], open_h=s['total']['open_h_per_yr'], vent_m3=s['total']['vent_m3_per_yr'],
                             vent_demand=s['demand']['vent_m3_per_yr'], vent_stop=s['stop']['vent_m3_per_yr'], ch4_kg=s['total']['ch4_kg_per_yr'],
                             chatter_cycles=s['total']['chatter_cycles_per_yr'], p_state_max=(float(df['p_state_open'].max()) if 'p_state_open' in df else np.nan)))
            print(key, 'openings %.3f  (stop %.3f demand %.3f foam %.3f block %.3f wv %.3f)  vent %.0f m3' % (
                s['total']['openings_per_yr'], s['stop']['openings_per_yr'], s['demand']['openings_per_yr'], s['foam']['openings_per_yr'], s['block']['openings_per_yr'], s['wv']['openings_per_yr'], s['total']['vent_m3_per_yr']))
r = pd.DataFrame(rows); r.to_csv(os.path.join(HERE, 'out', 'bn_onoff_9cases.csv'), index=False, float_format='%.5f')
json.dump(allsum, open(os.path.join(HERE, 'out', 'bn_onoff_summary.json'), 'w'), indent=1, ensure_ascii=False, default=float)
json.dump({k: {kk: (vv if kk != 'v' or not isinstance(vv, dict) else {str(a): b for a, b in vv.items()}) for kk, vv in v.items()} for k, v in bn.PARAMS.items()},
          open(os.path.join(HERE, 'out', 'bn_params_v4.json'), 'w'), indent=1, ensure_ascii=False, default=str)
pd.set_option('display.width', 250); print(r.round(3).to_string(index=False))
