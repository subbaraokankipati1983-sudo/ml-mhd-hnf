# ml-mhd-hnf 

Solver-in-the-loop optimization of an MHD micropolar Cu-Al2O3/water hybrid nanofluid with Soret-Dufour effects, chemical reaction and thermal dispersion.

## Contents
- `src/model.py` collocation BVP solver. Energy equation and wall flux use the same effective conductivity: Nu_x = -A4 (1 + D) theta'(0), with the dispersion closure k_d = D k_hnf u/U_w.
- `src/cheb_check.py`, `src/cheb_verify.py` independent Chebyshev spectral collocation solver (cross-check, results in `results/cheb.json`).
- `src/stage1_lhs.py` ... `src/stage6_admissibility.py` Latin-hypercube dataset, surrogates, validation, sensitivity/Sobol/Pareto and the cross-diffusion admissibility analysis (Du*Sr < k_hnf/k_f).
- `src/stage3_de.py` solver-in-the-loop differential evolution (reference box, wide box, suction caps, admissibility-margin caps, closure variants).
- `src/figs.py`, `src/graphical_abstract.py`, `src/mktables.py` figures and table values.
- `quick_check.py` clean-clone smoke test (seconds); it does not regenerate the study
- `reproduce_all.py` regenerates everything (about 45 min on two CPU cores, clean-clone run of 9 Oct 2026; archived outputs reproduced identically); `verify_results.py` checks the archived results against the values quoted in the article.
- `results/` archived outputs (LHS dataset, DE runs, validation, admissibility, Chebyshev check, suction caps).
- `figures/` figure files.

## Quick check
```
pip install -r requirements.txt
python quick_check.py      # a few seconds: re-solves baseline, Crane limit and the reference optimum
python verify_results.py   # checks archived results against the paper values
```

## Notes
- The default model is the corrected one (kfactor, micro-rotation inertia A0, flux with dispersion). Setting `nu_disp=False` in `src/common.py` gives the bulk-conductivity flux used only as a comparison case.
- The comparison script with the author's V23 code was dropped in V27; the V23 baseline differs because of later model corrections.
- Replace the repository URL and Zenodo DOI placeholders in the article and cover letter after archiving.


`src/check_refs.py references.bib main.tex` is an offline bibliography checker (required fields, DOI syntax, year range, citation use).

`src/stage7_closure_family.py` (closure-exponent sweep), `src/stage8_conformal.py` (split-conformal intervals for the Nu surrogate) and `src/stage9_figs.py` (Figure 11) were added in V32; `model.py` gained the optional `closure='power'` (exponent `pexp`).

## Reproduction check against the manuscript (28 Oct 2026)
`compare_with_manuscript.py` compares a regenerated run with the manuscript (archived vs regenerated results, `tables.json` vs the LaTeX source, headline numbers of the abstract, regenerated vs manuscript figures). The manuscript itself is not in this repository, so pass your local copy:
```
python reproduce_all.py
python compare_with_manuscript.py --manuscript main.tex --manuscript-figs figures_png --archived <copy of this repository as shipped>
```
The report of the last full clean run is in `reproduction_check/` (`verify_results.py`: 21 of 21 checks; every deterministic result identical to the archived one; the random-forest/gradient-boosting/Sobol stages move in the 3rd to 4th digit between runs, which changes the last printed digit of two cross-validation table rows and the sampled Pareto and conformal figures; none of this changes a conclusion).

Figure numbering: the files in `figures/` are numbered one below the manuscript for Figures 2 to 4 and 6 to 10 because manuscript Figure 1 is a schematic drawn in LaTeX: `Figure_1` = Fig. 2, `Figure_2` = 3, `Figure_3` = 4, `Figure_4` = 5, `Figure_5a/5b` = 6a/6b, `Figure_6a/6b` = 7a/7b, `Figure_7` = 8, `Figure_8` = 9, `Figure_9` = 10, `Figure_11` = 11.

`reproduce_all.py` now starts from an empty `de_results.jsonl` (the DE stage appends) and writes `results/lhs_dataset.csv`.
