# Evidence sensitivity of Bayesian network credit risk assessment

Code for the article "Evidence Sensitivity of Bayesian Network Credit Risk Assessment under Two Information Sets".

The study compares a discrete Bayesian network (BN) learned by BIC hill-climbing with logistic regression (LR) and random forest (RF) on the public Give Me Some Credit data, under two information sets: an application-stage "profile" set (6 inputs, 9-node BN including two missingness indicators) and an "extended" set (10 inputs, 13-node BN). It covers exact inference, evidence sensitivity, value of information, missing-evidence (masking) analyses and decision-value (cost saving) analyses.

**This repository holds code and small result tables only. It contains no data and no trained models.** Everything that depends on the data or on fitted models has to be regenerated locally.

## Data
Kaggle, "Give Me Some Credit" (`cs-training.csv`). You must obtain it yourself from Kaggle under its terms; it is not redistributed here.

- 150,000 rows; 149,999 after dropping the single row with Age = 0.
- Stratified 80/20 split with seed 42: 119,999 training rows and 30,000 test rows (default rate 6.68%).
- Place the file at `<workdir>/Data/GiveMeSomeCredit/cs-training.csv` (see Setup).

Intermediate frames used by the code (`Data/processed/*.pkl`: `df_clean`, `split_indices`, `preprocessing_artifacts`, `df_train`, `df_test`, `X_train_df`, `X_test_df` and related) were saved from the original interactive session and are not part of this repository. No notebook or script in this repository saves `df_clean`, `split_indices`, `preprocessing_artifacts`, `df_train`, `df_test`, `X_train_df` or `X_test_df` (searched for save calls naming them); `notebooks/Expirements_FROZEN.ipynb` holds the in-session code that cleans, splits, bins and imputes the data, and `notebooks/01_data_prep.ipynb` writes the `sel_*` frames from the saved ones. Scripts and notebooks that read these frames, trained models or stored predictions cannot run until you have recreated them. In particular every notebook loads pickles, 30 of the 47 scripts in `scripts/analysis/` read pickles, and so do 4 of the 14 scripts in `scripts/figures/`.

## Setup
1. Python 3.14 and the pinned packages in `requirements.txt` (pgmpy 1.1.2 is pinned: do not upgrade, `HillClimbSearch` is removed in pgmpy 1.3.0). pygraphviz needs the Graphviz system library (for example `brew install graphviz`).
2. `PYTHONHASHSEED` must be `0` before Python starts, or structure learning is not reproducible. `env/run.sh` sets it for scripts. For notebooks, create a kernel and add `"env": {"PYTHONHASHSEED": "0"}` to its `kernel.json`; every notebook asserts this in its first executed cell.
3. The code expects one working folder with the original layout. Choose a folder and build the layout with symbolic links into this repository:
   ```
   ./env/setup_workdir.sh /path/to/workdir
   export CREDIT_BN_ROOT=/path/to/workdir
   ```
   Every former hard-coded project path is now read from `CREDIT_BN_ROOT`. All outputs (tables, figures, pickles) are written under that folder, never into the repository. `setup_workdir.sh` replaces existing links of the same name in the working folder; use an empty folder. The layout is `workdir/Data/`, `workdir/shared/src` (this repository's `src/`), `workdir/selection/notebooks`, `workdir/full_variable/notebooks`, `workdir/Submissions/4. Article/WRITEUP/analysis_additions/code` (the analysis scripts) and `.../figures_final/code` (the figure scripts).
4. Run a script with `CREDIT_BN_ROOT=... env/run.sh path/to/script.py [args]` (set `PYTHON` to choose the interpreter).

Hash guards: some notebooks and `scripts/analysis/j8_rerun.py` compare `sha12` hashes of the modules in `src/` with hashes recorded on the original machine. `src/rq3.py` and `src/safe_save.py` differ from the original by the root-path line only, so those checks will not match until the recorded strings are updated (see `docs/path_changes.md`).

## Repository layout
- `src/`: reusable modules (`loglik`, `em`, `structure`, `sensitivity`, `masking`, `rq3`, `checks`, `decision_value`, `fast_auc`, `safe_save`).
- `notebooks/`: `Expirements_FROZEN.ipynb` (original full-variable session; read as JSON by later code, never re-run) and `00` to `08`, `10`, `11` of the selection analysis, with outputs removed. `09_packaging` is not included.
- `scripts/analysis/`: scripts for the additional analyses (`j*.py`, `split_robust.py`, `j_lib.py`, `costlib.py`, `c1_data.py`, ...); `scripts/figures/`: figure scripts; `scripts/exploration/Data_Exploration.py`.
- `results/`: small result tables that the notebooks and scripts write (CSV/JSON, each under 1 MB).
- `env/`: `run.sh`, `setup_workdir.sh`, `REPRODUCIBILITY.md`, `strip_outputs.py` (strips notebook outputs; usable as a git clean filter).
- `docs/path_changes.md`: every edit made when the code was prepared for release.

## Run order
The notebooks are chronological work logs, not top-to-bottom scripts. Run each from a fresh kernel and only the result-producing cells listed here (cell numbers are code-cell positions, 1 = first code cell).

Notebooks (selection analysis):
- `00_precheck`: variable-set check; cell 2 defines `build_design` and is executed as source by `src/rq3.py` and later code, so keep it. Its LR outputs are superseded by `05`.
- `01_data_prep`: cell 3 saves the selection frames and node orders.
- `02_structure`: cell 3 saves the structure results and the Markov blanket report.
- `03_em_score`: cell 17 saves the EM fit, cell 18 the Default CPD cells, cell 19 the test predictions and BN metrics, cell 20 the blanket robustness table.
- `04_baselines_rq2`: cell 7 saves the LR/RF grids; cell 8 the baselines table and bootstrap; cell 9 EM versus deletion; cell 14 the RQ2, value-of-information and cost-ratio tables.
- `05_convergence_checks`: cell 1 and cell 3 save the converged-LR checks.
- `06_rq3_setup`, `07_rq3_grid`: masking setup and the full masking grid (selection and monitoring), summaries and clip sensitivity.
- `08_story_extras`: decision-value curves, information ladder, stage agreement, worked examples.
- `10_continuous_ref_and_rq3_ci`: continuous-feature LR/RF reference and test-row bootstrap intervals for the masking contrasts.
- `11_paper_figures`: not described here; see its source cells.

Then the scripts in `scripts/analysis/` (each script's docstring states its inputs and outputs; the outputs named below are the ones checked against the script text):
- `j12_run.py`, `j12_agg.py`: re-runs for further split seeds; table `j12_aggregate.csv`.
- `j17_fit.py`, `j17_mask.py`, `j17.py`: past-due variable variants V1 and V2; `j17_variant_summary.csv`, `j17_linear_lr_ladder.csv`.
- `j19.py`, `j19_agg.py`: fairer baselines and decision-level masking comparison; `j19_decision_level.csv`, `j19_masked_auc.csv`.
- `j20.py`: value-of-information-guided data collection; outputs: see its header.
- `j21.py`, `j21_agg.py`: all evidence subsets; `j21_subset_results.csv`, `j21_gap_vs_full.csv`.
- `j22.py`, `j22_agg.py`: V1/V2 across split seeds; `j22_split_seed_results.csv`.
- `j24.py`, `j24_mask.py`, `j24_subsets.py`: naive Bayes and tree-augmented structures; `j24_*.csv`.
- `j25_mask.py`, `j25_subsets.py`: post-hoc tree-augmented-network check after acceptance; `j25_*.csv`.
- `c1_data.py`: saving and saving differences with 95% intervals for r = 1 to 20; `cost_saving_by_r_with_intervals.csv`.
- Figures: `scripts/figures/` (see `run_tierA.py` and each script's header).

Any script or table not listed above: see the script header.

## EM iterations
The main-article results for the extended set use the stored BN at EM iteration 79. The re-fits made by the analysis scripts (variants, split seeds, subsets, alternative structures) use 100 fixed EM iterations after acceptance; at the application stage EM stops by the rule "gain < 0.01 for 3 consecutive iterations". For the main model the two iteration counts give the same AUC and savings at the printed precision.

## Known quirks
- The frozen notebook is never re-run; later code reads its source cells as JSON (for example cells 39 and 42 are asserted on).
- Monitoring thresholds (LR 0.69, RF 0.67, BN 0.17) were chosen on test predictions by a 0.01-step sweep, so those locked cost figures are slightly optimistic. Selection thresholds were fixed in advance (BN `1/(1+r)`) or taken from out-of-fold training predictions.
- Dependents binning keeps the original `int()` truncation, so negative imputed counts wrap (`wrap` mode in `src/rq3.py`).
- The first logistic-regression check of the selection variable set (`00_precheck`: `saga`, C = 100, `max_iter=1000`) did not converge; `05_convergence_checks` refits it with `lbfgs`, and `fit_converged` in `src/checks.py` asserts convergence for the LR fits that use it.
- Two models, two questions: the application-stage and monitoring-stage BNs need not agree. Fairness was not assessed.
- The figure scripts (scripts/figures) and several analysis scripts (addcommon, build, c1_data, j11_eval, j13, j15, j16_snumbers) read an internal numbers ledger that is not included in this release. They check printed values against it and will not run without it; the result tables in results/ are the released outputs.
- Notebooks 03 to 08 and 10 contain optional manifest and commit cells that run git in the working folder. Skip them; they are not needed to produce any result.

## Reproducibility status
Whether the notebooks reproduce the stored results from a clean checkout is not documented here, and nothing in this release has been run from a fresh checkout. Reproducibility is not claimed.

## Licence
MIT (see `LICENSE`).
