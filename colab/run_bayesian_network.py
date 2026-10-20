# -*- coding: utf-8 -*-
"""
One-command driver for the Bayesian-network package.

    python colab/run_bayesian_network.py              # v4 on/off 9-case BN + sensitivity + figures (about one minute)
    python colab/run_bayesian_network.py --v3         # also the v3 proportional-flare BN, its tornado / DAG / profile figures and the buffer-band study
    python colab/run_bayesian_network.py --weather    # also the severe-weather scenarios (needs the pressure model; several minutes)

Stages
    v4       run_onoff9.py   nine cases x three flare-recovery variants  -> out/bn_onoff_9cases.csv, out/bn_hourly_*.csv.gz, out/bn_cpt_*.csv
             sens_v4.py      tornado inputs for Case 1 at flare start 100 % and 80 %
             fig_onoff9.py   breakdown / variant / 1-s comparison / weekly figures (uses pressure_model/out/onoff_summary.csv if present,
                             otherwise the shipped results/pressure_model/onoff_summary.csv)
    v3       make_v3_inputs.py, pv_bn.py   proportional flare 70-85 % (the first BN version), bn_sens_figs.py, buffer_band.py
    weather  crit_weather.py synthetic typhoon scenarios

Everything is written to bayesian_network/out/ and bayesian_network/fig/.
"""
import argparse, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BN = os.path.join(ROOT, 'bayesian_network')
PY = sys.executable

def run(script, *args):
    t = time.time()
    print(f'>> {script} {" ".join(args)}', flush=True)
    r = subprocess.run([PY, script, *args], cwd=BN, text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f'failed: {script}')
    print('\n'.join(r.stdout.strip().split('\n')[-12:]))
    print(f'   done in {time.time() - t:.0f} s', flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v3', action='store_true', help='also run the v3 (proportional flare) BN and its figures')
    ap.add_argument('--weather', action='store_true', help='also run the severe-weather scenarios')
    a = ap.parse_args()
    os.makedirs(os.path.join(BN, 'out'), exist_ok=True); os.makedirs(os.path.join(BN, 'fig'), exist_ok=True)
    run('run_onoff9.py')
    run('sens_v4.py')
    run('fig_onoff9.py')
    if a.v3:
        run('make_v3_inputs.py')
        run('pv_bn.py')
        run('bn_sens_figs.py')
        run('buffer_band.py')
    if a.weather:
        run('crit_weather.py')
    print('finished. Results: bayesian_network/out, figures: bayesian_network/fig')

if __name__ == '__main__':
    main()
