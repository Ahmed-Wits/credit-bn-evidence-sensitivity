# Path and identifier changes in the release copy

Every edit made to a code file relative to the working copy. Source cells were otherwise left byte-identical; notebook outputs, execution counts and widget metadata were removed.

| file | location | old | new |
|---|---|---|---|
| src/rq3.py | line 12 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| src/safe_save.py | line 12 | `ROOT = "<original project root>"` | `ROOT = os.environ["CREDIT_BN_ROOT"]` |
| scripts/analysis/addcommon.py | line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| scripts/analysis/j11_em.py | line 7 | `R = "<original project root>"; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"` | `R = os.environ["CREDIT_BN_ROOT"]; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"` |
| scripts/analysis/j8_boot.py | line 7 | `R = "<original project root>"; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"` | `R = os.environ["CREDIT_BN_ROOT"]; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"` |
| scripts/analysis/j8_rerun.py | line 9 | `R = "<original project root>"; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"` | `R = os.environ["CREDIT_BN_ROOT"]; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"` |
| scripts/analysis/split_robust.py | line 16 | `R = "<original project root>"; PROC = R + "/Data/processed/"; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"` | `R = os.environ["CREDIT_BN_ROOT"]; PROC = R + "/Data/processed/"; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"` |
| scripts/figures/common.py | line 12 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| scripts/figures/run_tierA.py | line 166 | `Python: Data/GiveMeSomeCredit/<venv dir> (matplotlib 3.11.1).` | `Python: project virtual environment (matplotlib 3.11.1).` |
| scripts/figures/s7_em_rerun.py | line 10 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| scripts/exploration/Data_Exploration.py | line 4 | `df = pd.read_csv('<original project root>/Data/GiveMeSomeCredit/cs-training.csv')` | `df = pd.read_csv(os.environ["CREDIT_BN_ROOT"] + '/Data/GiveMeSomeCredit/cs-training.csv')` |
| scripts/exploration/Data_Exploration.py | line 1 (inserted) | `(none)` | `import os` |
| notebooks/Expirements_FROZEN.ipynb | cell 0 line 15 | `RAW_PATH = "<original project root>/Data/GiveMeSomeCredit/cs-training.csv"` | `import os; RAW_PATH = os.environ["CREDIT_BN_ROOT"] + "/Data/GiveMeSomeCredit/cs-training.csv"` |
| notebooks/Expirements_FROZEN.ipynb | cell 82 line 5 | `ROOT = "<original project root>"` | `ROOT = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/Expirements_FROZEN.ipynb | cell 82 line 6 | `SKIP = {'<venv dir>', '.git', '__pycache__', '.ipynb_checkpoints'}` | `SKIP = {'.' + 'venv', '.git', '__pycache__', '.ipynb_checkpoints'}` |
| notebooks/Expirements_FROZEN.ipynb | cell 84 line 5 | `d = "<original project root>/Data/GiveMeSomeCredit"` | `d = os.environ["CREDIT_BN_ROOT"] + "/Data/GiveMeSomeCredit"` |
| notebooks/Expirements_FROZEN.ipynb | cell 85 line 6 | `d = "<original project root>/Data/GiveMeSomeCredit/"` | `import os; d = os.environ["CREDIT_BN_ROOT"] + "/Data/GiveMeSomeCredit/"` |
| notebooks/Expirements_FROZEN.ipynb | cell 86 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/Expirements_FROZEN.ipynb | cell 88 line 10 | `D = "<original project root>/Data/GiveMeSomeCredit/"` | `D = os.environ["CREDIT_BN_ROOT"] + "/Data/GiveMeSomeCredit/"` |
| notebooks/Expirements_FROZEN.ipynb | cell 89 line 7 | `D = "<original project root>/Data/GiveMeSomeCredit/"` | `D = os.environ["CREDIT_BN_ROOT"] + "/Data/GiveMeSomeCredit/"` |
| notebooks/Expirements_FROZEN.ipynb | cell 90 line 5 | `D = "<original project root>/Data/GiveMeSomeCredit/"` | `import os; D = os.environ["CREDIT_BN_ROOT"] + "/Data/GiveMeSomeCredit/"` |
| notebooks/Expirements_FROZEN.ipynb | cell 91 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/Expirements_FROZEN.ipynb | cell 92 line 7 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/Expirements_FROZEN.ipynb | cell 93 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/Expirements_FROZEN.ipynb | cell 93 line 40 | `print("\n3) Files under Research/ modified since 15:30 today (excl <venv dir>/.git)")` | `print("\n3) Files under Research/ modified since 15:30 today (excl venv/.git)")` |
| notebooks/Expirements_FROZEN.ipynb | cell 93 line 44 | `dns[:] = [d for d in dns if d not in {'<venv dir>', '.git', '__pycache__', '.ipynb_checkpoints'}]` | `dns[:] = [d for d in dns if d not in {'.' + 'venv', '.git', '__pycache__', '.ipynb_checkpoints'}]` |
| notebooks/Expirements_FROZEN.ipynb | cell 94 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/Expirements_FROZEN.ipynb | metadata.kernelspec.display_name | `<venv dir> (3.14.4)` | `Python 3` |
| notebooks/00_precheck.ipynb | cell 0 line 8 | `ROOT = "<original project root>"` | `ROOT = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/01_data_prep.ipynb | cell 0 line 8 | `ROOT = "<original project root>"` | `ROOT = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/02_structure.ipynb | cell 0 line 8 | `ROOT = "<original project root>"` | `ROOT = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 0 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 1 line 5 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 2 line 5 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 3 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 4 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 5 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 6 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 7 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 8 line 15 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 9 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 10 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 12 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 13 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 14 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 15 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 16 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 17 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 18 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 19 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 20 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/03_em_score.ipynb | cell 21 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 0 line 5 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 1 line 5 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 2 line 5 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 3 line 135 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 4 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 5 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 6 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 6 line 26 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 7 line 11 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 8 line 10 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 9 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 10 line 7 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 11 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 12 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 13 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 14 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 15 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 16 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/04_baselines_rq2.ipynb | cell 17 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/05_convergence_checks.ipynb | cell 0 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/05_convergence_checks.ipynb | cell 1 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/05_convergence_checks.ipynb | cell 2 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/05_convergence_checks.ipynb | cell 3 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/05_convergence_checks.ipynb | cell 4 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 0 line 31 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 1 line 7 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 2 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 3 line 13 | `ROOT = "<original project root>"` | `ROOT = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 3 line 100 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 4 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 5 line 69 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 6 line 13 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 6 line 237 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 7 line 5 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 8 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/06_rq3_setup.ipynb | cell 9 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/07_rq3_grid.ipynb | cell 0 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/07_rq3_grid.ipynb | cell 1 line 10 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/07_rq3_grid.ipynb | cell 2 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/07_rq3_grid.ipynb | cell 3 line 78 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/07_rq3_grid.ipynb | cell 4 line 12 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/07_rq3_grid.ipynb | cell 6 line 10 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/07_rq3_grid.ipynb | cell 7 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/07_rq3_grid.ipynb | cell 8 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/07_rq3_grid.ipynb | cell 9 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 0 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 1 line 53 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 2 line 11 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 3 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 4 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 5 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 6 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 7 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 8 line 8 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 9 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 10 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 11 line 9 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 12 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 13 line 4 | `R = "<original project root>"` | `import os; R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 14 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 15 line 11 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 16 line 11 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 17 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 18 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 19 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/08_story_extras.ipynb | cell 20 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/10_continuous_ref_and_rq3_ci.ipynb | cell 0 line 5 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/10_continuous_ref_and_rq3_ci.ipynb | cell 1 line 5 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/10_continuous_ref_and_rq3_ci.ipynb | cell 2 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/10_continuous_ref_and_rq3_ci.ipynb | cell 3 line 30 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/10_continuous_ref_and_rq3_ci.ipynb | cell 4 line 6 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/10_continuous_ref_and_rq3_ci.ipynb | cell 6 line 7 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/10_continuous_ref_and_rq3_ci.ipynb | cell 7 line 7 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/10_continuous_ref_and_rq3_ci.ipynb | cell 8 line 4 | `R = "<original project root>"` | `R = os.environ["CREDIT_BN_ROOT"]` |
| notebooks/11_paper_figures.ipynb | cell 0 line 7 | `SEED = 42; R = "<original project root>"` | `SEED = 42; R = os.environ["CREDIT_BN_ROOT"]` |
| env/run.sh | lines 7-8 / header | `PY="<project venv path>/bin/python"` | `PY="${PYTHON:-python}"; CREDIT_BN_ROOT required and exported` |

## Counts per file

| file | lines changed |
|---|---|
| env/run.sh | 1 |
| notebooks/00_precheck.ipynb | 1 |
| notebooks/01_data_prep.ipynb | 1 |
| notebooks/02_structure.ipynb | 1 |
| notebooks/03_em_score.ipynb | 21 |
| notebooks/04_baselines_rq2.ipynb | 19 |
| notebooks/05_convergence_checks.ipynb | 5 |
| notebooks/06_rq3_setup.ipynb | 12 |
| notebooks/07_rq3_grid.ipynb | 9 |
| notebooks/08_story_extras.ipynb | 21 |
| notebooks/10_continuous_ref_and_rq3_ci.ipynb | 8 |
| notebooks/11_paper_figures.ipynb | 1 |
| notebooks/Expirements_FROZEN.ipynb | 16 |
| scripts/analysis/addcommon.py | 1 |
| scripts/analysis/j11_em.py | 1 |
| scripts/analysis/j8_boot.py | 1 |
| scripts/analysis/j8_rerun.py | 1 |
| scripts/analysis/split_robust.py | 1 |
| scripts/exploration/Data_Exploration.py | 2 |
| scripts/figures/common.py | 1 |
| scripts/figures/run_tierA.py | 1 |
| scripts/figures/s7_em_rerun.py | 1 |
| src/rq3.py | 1 |
| src/safe_save.py | 1 |

## Further edits to the figure scripts

| file | change |
|---|---|
| scripts/figures/fig2.py, fig3.py, fig4.py, fig4x.py, fig5.py, fig6.py, figS8.py, figS9.py, figsB.py | The figure is saved to `$CREDIT_BN_ROOT/figures_out/<same file name>` (folder created if missing) instead of a hard-coded temporary path; `import os` added. No other change. |
| scripts/figures/run_tierA.py | Removed an optional read of an internal text file (two lines). |
| scripts/figures/figS9.py | Removed an internal revision label from one note string. |

## Not included

Internal build and checking scripts that are not needed to regenerate the reported results were not included. 09_packaging.ipynb is not included.
