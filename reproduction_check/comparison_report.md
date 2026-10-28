# Comparison of the regenerated run with the manuscript

## A. Regenerated results vs archived results

- OK   stage7.json (deterministic): 0 entries outside tolerance; largest relative deviation 0.00e+00
- OK   val.json (deterministic): 0 entries outside tolerance; largest relative deviation 0.00e+00
- OK   caps.json (deterministic): 0 entries outside tolerance; largest relative deviation 0.00e+00
- OK   stage2.json (stochastic ML stage, rtol 0.1): 0 entries outside tolerance; largest relative deviation 1.02e-01
- OK   stage5.json (stochastic ML stage, rtol 0.1): 0 entries outside tolerance; largest relative deviation 9.91e-03
- OK   stage6.json (stochastic ML stage, rtol 0.1): 0 entries outside tolerance; largest relative deviation 1.05e-02
- OK   stage8.json (stochastic ML stage, rtol 0.1): 0 entries outside tolerance; largest relative deviation 3.76e-02
- INFO cheb.json: baseline rel. error 2.5e-08 vs 2.5e-08; optimum 1.8e-09 vs 1.8e-09; LHS cases compared 39 vs 40; max rel. LHS difference 2.2e-07 vs 2.2e-07
- OK   de_results.jsonl: 28 optimization runs, 0 differ in Nu or design vector
- INFO tables.json: 186 entries, 10 differ in the last printed digit (['cheb_n', 'T_cv', 'T_t10', 'T_abl', 'abl_nu_diff', 'abl_sh_diff', 'par_lowSh_Nu', 'par_cf_lo', 'par_cf_hi', 'abl_max']); these come from the stochastic ML stages. Section B shows whether any of them is quoted in the manuscript.
- OK   lhs_dataset (csv is rounded to 9 digits; regenerated src/lhs_raw.npy if the csv is absent): shape (2000, 13), max |diff| = 0.00e+00

## B. Regenerated tables.json vs the manuscript LaTeX source

- 236 of 262 table rows / numbers are found verbatim in the manuscript source.
- Entries that are archived but not used in the manuscript text (11 keys, 26 rows): T_Dtab, T_Ha, T_abl, T_cv, T_sens, T_t10, abl_max, abl_sh_diff, ein_rel, par_lowSh_Nu, wds_D
- OK   keys that are only partly found in the manuscript (a real mismatch): none
- INFO T_cv: 1 row(s) differ from the manuscript in the last printed digit after the clean re-run (random cross-validation folds); the manuscript keeps the archived-run values
- INFO T_t10: 1 row(s) differ from the manuscript in the last printed digit after the clean re-run (random cross-validation folds); the manuscript keeps the archived-run values

## C. Headline numbers in the abstract and text vs regenerated results

- OK   baseline Nu (abstract: raises from): manuscript 12.16, regenerated 12.16
- OK   optimum Nu (abstract: to): manuscript 42.2, regenerated 42.2
- OK   gain factor (abstract: n-fold): manuscript 3.5, regenerated 3.47
- OK   ten-seed minimum: manuscript 38.8, regenerated 38.75
- OK   ten-seed maximum: manuscript 45.7, regenerated 45.69
- OK   non-converged cases (abstract): manuscript 128, regenerated 128
- OK   converged above the Du Sr bound: manuscript 4, regenerated 4
- OK   cross-validated R2 of Nu_x (abstract): manuscript 0.95, regenerated 0.9454
- OK   share kept with f_w<=1.5 (abstract, %): manuscript 52, regenerated 52

## D. Regenerated figures vs manuscript figures

| regenerated figure | best manuscript figure | correlation | mean abs. diff | vs archived copy |
|---|---|---|---|---|
| Figure_1 | Figure_2 | 0.9976 | 0.0040 | 0.9976 |
| Figure_11 | Figure_11 | 0.9698 | 0.0066 | 0.9698 |
| Figure_2 | Figure_3 | 0.9984 | 0.0017 | 0.9984 |
| Figure_3 | Figure_4 | 0.9996 | 0.0013 | 0.9996 |
| Figure_4 | Figure_5 | 0.9988 | 0.0009 | 0.9988 |
| Figure_5a | Figure_6a | 0.9999 | 0.0007 | 0.9999 |
| Figure_5b | Figure_6b | 0.6638 | 0.0290 | 0.6638 |
| Figure_6a | Figure_7a | 0.9952 | 0.0017 | 0.9952 |
| Figure_6b | Figure_7b | 0.9939 | 0.0022 | 0.9939 |
| Figure_7 | Figure_8 | 0.9922 | 0.0021 | 0.9922 |
| Figure_8 | Figure_9 | 0.9875 | 0.0027 | 0.9875 |
| Figure_9 | Figure_10 | 0.9948 | 0.0024 | 0.9948 |

- OK   deterministic figures whose best manuscript match has correlation below 0.98: none
- INFO Figure_11 (conformal calibration with random splits) depends on random sampling: correlation 0.97; same axes and trend, individual points differ (check by eye)
- INFO Figure_5b (Pareto front of a sampled set) depends on random sampling: correlation 0.66; same axes and trend, individual points differ (check by eye)
- manuscript figures not matched by any regenerated figure: ['Figure_1'] (figures drawn without the pipeline, for example the schematic, or exported at another size)

**Result: all checks passed.**
