# Reference results of the study

These files are the outputs of the exact runs described in the main README (§6), kept so that a fresh clone can be checked
against them and so that `bayesian_network/fig_onoff9.py` can draw the BN-vs-1-s comparison before the pressure model has been run.

## pressure_model/
| File | From | Content |
|---|---|---|
| `summary_base.csv` | `make_figs.py` | one row per operating-pressure case: PT statistics, margins, fill range, friction and head losses, valve and blower counts (year, downstream consumes all gas) |
| `summary_stop.csv` | `make_figs.py` | downstream-stop scan medians per case and start fill (time to full, time to first opening, openings and vented volume after opening, 6-h checkpoints) |
| `full100_summary.csv` | `sum_full100.py` | holder-kept-full regime with a pressure-controlled flare, PFD 0 and 0.05 |
| `onoff_summary.csv` | `sum_onoff.py` | the nine on/off cases (+ three PFD-0 controls): fill distribution, flare starts and failures, consumer stops, PVRV episodes / cycles / open hours / vented m³ / CH₄, PT range, air-side counts |
| `onoff_f100_failures.csv` | `fig_episode.py` | every ignition failure of the flare-start-100 % runs with the production at the time and whether the PVRV opened |
| `*_summary.json` | run scripts | per-run summaries (the same quantities as above, one file per run) |

## bayesian_network/
| File | From | Content |
|---|---|---|
| `bn_onoff_9cases.csv` | `run_onoff9.py` | 27 rows = 9 cases x 3 flare-recovery variants: expected openings per initiating event, probability of at least one opening, open hours, vented m³, CH₄, chattering cycles, peak state probability |
| `bn_onoff_series.json` | `run_onoff9.py` | statistics of the on/off operating-model inputs (flare demands, fill distribution) |
| `bn_sens_v4_case1_f100.csv`, `bn_sens_v4_case1_f80.csv`, `bn_sens_v4_base.json` | `sens_v4.py` | one-at-a-time sensitivities (tornado inputs) |
| `bn_cpt_case1_f100_retry30.csv` | `run_onoff9.py` | conditional probability table aggregated by production class, fill class and day/night (example) |
| `bn_params_v4.json` | `run_onoff9.py` | all parameters with units and sources |
| `bn_summary.json`, `bn_sensitivity_case1.csv`, `buffer_band.csv`, `bn_optional_nodes.csv`, `crit_weather.json` | v3 scripts | the earlier proportional-flare network and its studies |

## dynamic_qra/
| File | From | Content |
|---|---|---|
| `source_terms.csv`, `source_props.json`, `episodes_R1_f100.csv` | `source_terms.py` | release types R1-R5 (rate, duration, volume, composition), gas properties, the 104 chattering episodes of the flare-start-100 % runs |
| `effect_distances.csv`, `puff_R5.csv`, `h2s_probit_table.csv` | `consequence.py` | LFL and H₂S distances per release type and weather class, the seal-failure puff, the probit table |
| `fatality_table.csv`, `qra_params.json` | `risk.py` | fatality probabilities per exposure mode, all QRA parameters with sources |
| `risk_summary.csv`, `risk_profiles.csv`, `risk_sensitivity.csv` | `risk.py` | annual IR / LSIR / emission per case, monthly and hour-of-day profiles, tornado |
| `updating.csv` | `updating.py` | Bayesian updating of the flare PFD and the foaming rate, five synthetic years |

## figures/
Key figures regenerated with the English-labelled scripts: base-year pressure behaviour and one day at 1 s (Case 1), the
downstream-stop scan and the chattering comparison, the holder-kept-full year, the nine on/off cases (fill and openings,
bar summary, one opening episode), and the BN results (DAG, breakdown by initiating event with variants, BN-vs-1-s comparison,
weekly profile of Case 1), and the dynamic QRA (event tree, IR by case, profiles, tornado, updating, consequence curves).
