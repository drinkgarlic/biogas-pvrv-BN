# -*- coding: utf-8 -*-
"""
One-command driver for the 1-s pressure model, reproducing the runs used in the study.

    python colab/run_pressure_model.py                 # everything, full year (25 one-year runs + 9 event scans)
    python colab/run_pressure_model.py --quick         # same workflow on the first 30 days (a few minutes with numba)
    python colab/run_pressure_model.py --stages onoff  # only the on/off 9-case stage (+ its summary figures)
    python colab/run_pressure_model.py --jobs 4        # run the independent year simulations 4 at a time

Stages (in order)
    base     run_base.py for Cases 1-3        downstream consumes all gas, 1 year at 1 s
    scan     scan_stop.py                      downstream-stop event scan, Cases 1-3 x start fill 85 / 50 %,
                                               plus the Case 1 sensitivities (min. opening 0 / 25 %, absolute set pressure)
    figs     make_figs.py                      figures + summary_base.csv / summary_stop.csv
    full100  run_full100.py (4 runs) + sum_full100.py      holder kept full, pressure-controlled flare, PFD 0.05 (and 0 for Case 1)
    onoff    run_onoff.py (18 runs) + sum_onoff.py + fig_episode.py   consumer and flare on/off, flare start 100 / 80 / 60 %,
                                               PFD 0.05 and PFD 0 (the PFD 0 years are the BN input series, see make_onoff_inputs.py)

Outputs land in pressure_model/out/ and pressure_model/fig/. Run times: with numba a one-year run takes well under a
minute after the first compilation; in pure Python it takes 12-25 minutes, so use --quick first.
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PM = os.path.join(ROOT, 'pressure_model')
PY = sys.executable

def run(cmd, env=None):
    t = time.time()
    print(f'>> {" ".join(cmd)}', flush=True)
    r = subprocess.run([PY] + cmd, cwd=PM, env=env, text=True, capture_output=True)
    tail = '\n'.join(r.stdout.strip().split('\n')[-3:])
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f'failed: {cmd}')
    print(f'   done in {time.time() - t:.0f} s\n   {tail}', flush=True)

def parallel(cmds, jobs, env):
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        list(ex.map(lambda c: run(c, env), cmds))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='simulate only the first N days (default 30) to check the workflow')
    ap.add_argument('--days', type=int, default=30, help='number of days in --quick mode')
    ap.add_argument('--stages', default='base,scan,figs,full100,onoff', help='comma-separated subset of: base,scan,figs,full100,onoff')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 1), help='parallel year simulations (each needs ~1 GB RAM)')
    a = ap.parse_args()
    env = dict(os.environ)
    if a.quick:
        env['PVRV_QUICK_DAYS'] = str(a.days)
        print(f'quick mode: first {a.days} days only')
    stages = [s.strip() for s in a.stages.split(',')]
    os.makedirs(os.path.join(PM, 'out'), exist_ok=True); os.makedirs(os.path.join(PM, 'fig'), exist_ok=True)
    try:
        import numba  # noqa: F401
        print('numba available: compiled kernel')
    except ImportError:
        print('numba not installed: pure-Python kernel (slow). pip install numba')
    if 'base' in stages:
        parallel([['run_base.py', str(c)] for c in (1, 2, 3)], a.jobs, env)
    if 'scan' in stages:
        parallel([['scan_stop.py', str(c), f0] for c in (1, 2, 3) for f0 in ('0.85', '0.5')], a.jobs, env)
        parallel([['scan_stop.py', '1', '0.85', '0.0'], ['scan_stop.py', '1', '0.85', '0.25'], ['scan_stop.py', '1', '0.85', '0.10', 'abs']], a.jobs, env)
    if 'figs' in stages:
        run(['make_figs.py'], env)
    if 'full100' in stages:
        parallel([['run_full100.py', '1', '0.05'], ['run_full100.py', '2', '0.05'], ['run_full100.py', '3', '0.05'], ['run_full100.py', '1', '0.0']], a.jobs, env)
        run(['sum_full100.py'], env)
    if 'onoff' in stages:
        jobs = [['run_onoff.py', str(c), f, '0.05'] for c in (1, 2, 3) for f in ('1.0', '0.8', '0.6')] + [['run_onoff.py', str(c), f, '0.0'] for c in (1, 2, 3) for f in ('1.0', '0.8', '0.6')]
        parallel(jobs, a.jobs, env)
        run(['sum_onoff.py'], env)
        run(['fig_episode.py'], env)
    print('all stages finished. Results: pressure_model/out, figures: pressure_model/fig')

if __name__ == '__main__':
    main()
