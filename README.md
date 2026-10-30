# Fibonacci Cosmology

Gravitotemporal Quasicrystal Cosmology project scaffold.

## Layout

Auto-generated from the current workspace (abridged for large data/cache folders).
Run `bash scripts/update_tree.sh` to refresh this section.

<!-- TREE:START -->
```text
fibonacci-cosmology/
├── .github/
│   └── copilot-instructions.md
├── data/
│   ├── bao/
│   │   └── README.md
│   ├── cmb/
│   │   ├── COM_PowerSpect_CMB-EE-full_R3.01.txt
│   │   ├── COM_PowerSpect_CMB-TE-full_R3.01.txt
│   │   ├── COM_PowerSpect_CMB-TT-binned_R3.01.txt
│   │   ├── COM_PowerSpect_CMB-TT-full_R3.01.txt
│   │   ├── planck_bestfit_theory_cl.dat
│   │   └── planck_tt_spectrum.txt
│   ├── ligo_cache/
│   │   └── ...
│   ├── pantheon/
│   │   └── ...
│   └── planck/
│       ├── COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt
│       ├── COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum_R3.01.txt
│       ├── COM_PowerSpect_CMB-EE-full_R3.01.txt
│       ├── COM_PowerSpect_CMB-TE-full_R3.01.txt
│       └── COM_PowerSpect_CMB-TT-full_R3.01.txt
├── notebooks/
│   ├── 01_pantheon_fit.ipynb
│   ├── 02_cmb_phi_search.ipynb
│   └── 03_bao_harmonics.ipynb
├── results/
│   ├── chains/
│   │   └── ...
│   ├── figures/
│   │   └── ...
│   ├── tables/
│   │   └── ...
│   ├── cmb_dsi_search_results.json
│   ├── cmb_dsi_search_results_tt_te_ee_unbinned_300.json
│   ├── cmb_dsi_search_results_tt_unbinned_10k.json
│   ├── cmb_phi_search_results.json
│   ├── ligo_dsi_search_results.json
│   ├── ligo_dsi_validation_results.json
│   ├── mcmc_full_run.log
│   └── pantheon_results.json
├── scripts/
│   └── update_tree.sh
├── src/
│   ├── bao_dsi_fit.py
│   ├── check_modelc_grid_convergence.py
│   ├── cmb_analysis.py
│   ├── cmb_dsi_lnell.py
│   ├── cmb_dsi_search.py
│   ├── cmb_phi_search.py
│   ├── constants.py
│   ├── data_loader.py
│   ├── fitting.py
│   ├── ligo_dsi_search.py
│   ├── ligo_dsi_validation.py
│   ├── models.py
│   ├── plotting.py
│   ├── process_pantheon.py
│   ├── quasicrystal_structure_factor.py
│   ├── run_lcdm_baseline_fit.py
│   ├── run_model_comparison.py
│   └── run_modelc_mcmc.py
├── tests/
│   ├── test_cmb.py
│   ├── test_constants.py
│   ├── test_data_loader.py
│   ├── test_fitting.py
│   └── test_models.py
├── execute.md
├── Gravitotemporal Quasicrystal Hypothesis.md
├── LICENSE
├── README.md
├── requirements.txt
└── SUMMARY.md
```
<!-- TREE:END -->

## Quick start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run tests:
   - `pytest`
