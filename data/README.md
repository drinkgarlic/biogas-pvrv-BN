# Input data

| File | Content | Source |
|---|---|---|
| `ADM1_hourly_1to9_Q200_6000m3_ffOTS1.csv` | Hourly PyADM1ODE output for the Pohang food-waste plant: feed 200 m³/d, digester volume 6,000 m³ (two digesters), interval-end timestamps 2020-11-01 01:00 to 2021-11-01 00:00. 34 columns: gas flows (wet and dry, 20 °C), CH₄ / CO₂ / H₂ / H₂O fractions, H₂S and NH₃ equivalents, pH, VFA, TAN. | PyADM1ODE (Gaida) run with the monthly feedstock composition of the plant; see Appendix III of the report |
| `kma_asos138_pohang_hourly_pressure_kPa.csv` | Hourly station pressure in kPa, columns `datetime`, `p_kPa`, 8,736 rows (25 missing hours are interpolated by the loaders). | Korea Meteorological Administration, ASOS site 138 (Pohang), converted from the public hourly download (`일시`, `현지 기압`) |

Both loaders (`pressure_model.load_inputs`, `pv_bn.load_inputs`) interpolate the hourly knots linearly; the 1-s model does this inside the kernel, so no second-resolution files are needed.

To run the models for another plant, replace these two files with the same column names (`datetime_interval_end`, `Q_gas_wet_m3_d_20C`, `CH4_wet_pct`, `CO2_wet_pct`, `H2O_wet_pct`, `H2S_gas_ppmv_eq`, `NH3_gas_ppmv_eq`, `H2_dry_ppm` for the ADM1 file; `datetime`, `p_kPa` for the pressure file) and adjust `t_start` / `t_end` in the loaders if the period differs.
