"""J4: threshold rules. Reproduces the locked thresholds and the OOF thresholds from saved predictions; nothing is refitted."""
from addcommon import *
import pickle
sys.path.insert(0, R + "/shared/src")
import decision_value as dv
p = Part("j4")
ks = json.load(open(R + "/full_variable/tables/kernel_scalars.json"))
ab = pickle.load(open(R + "/full_variable/predictions/ablation_predictions.pkl", "rb")); bbp = pickle.load(open(R + "/full_variable/predictions/bn_test_predictions.pkl", "rb"))["bn_test_probs"]
sp = pickle.load(open(R + "/Data/processed/split_indices.pkl", "rb")); y_tr = np.asarray(sp["y_train"]).astype(int); y_te = np.asarray(sp["y_test"]).astype(int)
v = bbp.notna().to_numpy()
grid = np.arange(0.01, 1.00, 0.01)    # locked sweep, as in archive/kernel/In_history_at_cell78.txt line 2285
def sweep(y, pr, fn=5):
    c = np.array([int(fn * (((pr < t) & (y == 1)).sum()) + ((pr >= t) & (y == 0)).sum()) for t in grid]); return c
cl = sweep(y_te, np.asarray(ab["LR_all"], float)); cr = sweep(y_te, np.asarray(ab["RF_all"], float)); cb = sweep(y_te[v], bbp.to_numpy(float)[v])
tl, trf, tb = grid[np.argmin(cl)], grid[np.argmin(cr)], grid[np.argmin(cb)]
p.near("locked LR threshold reproduced (argmin on test, 0.01 grid, first minimum)", tl, ks["tau_lr"], 1e-9, "min cost %d" % cl.min())
p.near("locked RF threshold reproduced", trf, ks["tau_rf"], 1e-9, "min cost %d" % cr.min())
p.near("locked BN threshold reproduced (29,998 rows where the BN is defined)", tb, ks["tau_bn"], 1e-9, "min cost %d" % cb.min())
for nm, c in (("LR", cl), ("RF", cr), ("BN", cb)):
    best = grid[c == c.min()]
    p.check(f"locked {nm}: plateau of tied minima on the grid has {len(best)} point(s): {best.round(2).tolist()} (argmin takes the first, i.e. the lowest)", True)
# Phase 5 (selection) thresholds at r=5 and rq3 thresholds at r=10
bp = pickle.load(open(R + "/selection/predictions/sel_baseline_test_predictions.pkl", "rb"))
for m in ("LR", "RF"):
    t_low, t_high = dv.oof_tau(bp["oof_" + m.lower()], y_tr, 5, "lowest")[0], dv.oof_tau(bp["oof_" + m.lower()], y_tr, 5, "highest")[0]
    p.near(f"Application {m} r=5 threshold in sel_baseline_test_predictions.pkl equals the lowest-tie OOF value", bp["tau"][m], t_low, 1e-12, f"lowest {t_low:.2f}, highest {t_high:.2f}")
j10 = json.load(open(R + "/selection/tables/sel_rq3_thresholds_r10.json"))["taus"]
for m in ("LR", "RF"):
    p.near(f"Application {m} r=10 threshold in sel_rq3_thresholds_r10.json equals the OOF value (lowest tie)", j10[m], dv.oof_tau(bp["oof_" + m.lower()], y_tr, 10, "lowest")[0], 1e-12)
# ties: where does the rule matter?
oofm = pickle.load(open(R + "/monitoring_rq3/mon_oof_predictions.pkl", "rb"))["oof"]
tie_rows = []
for st, oo in (("Application", {"LR": bp["oof_lr"], "RF": bp["oof_rf"]}), ("After acceptance", oofm)):
    for m in ("LR", "RF"):
        for r in range(1, 21):
            t, c, (lo, hi) = dv.oof_tau(oo[m], y_tr, r, "highest")
            if abs(hi - lo) > 1e-12: tie_rows.append((st, m, r, lo, hi))
p.check("number of (stage, model, r) cells with tied minima on the OOF grid (rule matters)", True, "%d of 80: %s" % (len(tie_rows), "; ".join(f"{a[:3]} {b} r={r}: {lo:.2f}-{hi:.2f}" for a, b, r, lo, hi in tie_rows) if tie_rows else "none"))
same = []
for st, m, r, lo, hi in tie_rows:
    pt = {"LR": bp["p_lr"], "RF": bp["p_rf"]}[m] if st == "Application" else None
    if pt is not None: same.append(int((pt >= lo).sum()) == int((pt >= hi).sum()))
p.check("for every tied Application cell the lowest and the highest tied threshold flag the same number of test rows (so the tie rule changes no test result)", all(same) and len(same) == len(tie_rows), "test rows flagged at the lowest / highest tied threshold: " + ", ".join(f"{m} r={r}: {int(({'LR': bp['p_lr'], 'RF': bp['p_rf']}[m] >= lo).sum())}/{int(({'LR': bp['p_lr'], 'RF': bp['p_rf']}[m] >= hi).sum())}" for st, m, r, lo, hi in tie_rows[:3]) + " ...")
tau5 = {(st, m): None for st in ("Application", "After acceptance") for m in ("LR", "RF")}
comb = pd.read_csv(R + "/selection/tables/sel_decision_value_combined.csv")
rows = ["| Stage | Model | " + " | ".join(f"r={r}" for r in (5, 10, 15, 20)) + " |", "|---|---|" + "---|" * 4]
for st, key in (("Application", "selection"), ("After acceptance", "monitoring")):
    for m in ("LR", "RF"):
        rows.append(f"| {st} | {m} | " + " | ".join("%.2f" % comb[(comb.contribution == key) & (comb.model == m) & (comb.r == r)].tau.iloc[0] for r in (5, 10, 15, 20)) + " |")
md = ["## J4. Threshold rules", "",
      "### (a) After acceptance: the 'locked' thresholds (BN 0.17, LR 0.69, RF 0.67)", "",
      "- **Criterion.** The threshold that minimises the total cost `5 x FN + 1 x FP` (cost ratio r = 5, FN cost 5 and FP cost 1) of the rule 'flag if p >= threshold' (`cost_sensitive_loss`: `preds = probs >= threshold`).",
      "- **Grid.** `thresholds_sweep = np.arange(0.01, 1.00, 0.01)` (99 values, 0.01 to 0.99). **Tie rule:** `np.argmin` (first minimum, i.e. the lowest threshold among ties).",
      "- **Data.** The **test set**: LR and RF on all 30,000 test rows (`y_test`, `lr_test_probs`, `rf_restricted_test_probs`), the BN on the 29,998 test rows where the BN is defined (`y_test_valid`, `bn_probs_valid`). Hence the locked thresholds are test-set choices (caveat C1; the article says so).",
      "- **Which models.** All three, each with its own value: BN **0.17**, LR **0.69**, RF **0.67**. The BN value 0.17 is the cost-minimising grid value on the test set (close to, but not equal to, the Bayes threshold 1/6 = 0.1667); the same function was also run at other cost ratios (a cost-ratio sensitivity loop follows it, same sweep).",
      "- **Where.** `full_variable/notebooks/Expirements_FROZEN.ipynb`, cell index 54 (the notebook is not in the repository). The code text of that cell is preserved in `archive/kernel/In_history_at_cell78.txt` as `In[55]` ('Cell 50: Cost-sensitive threshold optimization', function `find_optimal_threshold` at line 2273, sweep at line 2285, calls at lines 2287 to 2289, ratio-sensitivity loop from line 2320); the same function is in `shared/src/_recovered_from_kernel.py`. The values are stored in `full_variable/tables/kernel_scalars.json`.",
      "- **Reproduction (saved predictions only).** The three values are reproduced by this sweep on `ablation_predictions.pkl` (LR_all, RF_all), `bn_test_predictions.pkl` and `split_indices.pkl` (see checks).", "",
      "### (b) LR and RF thresholds at each r (used for Fig 4 and the cost-saving results)", "",
      "- **Out-of-fold predictions on the training set.** Application: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` with `cross_val_predict(clone(best_estimator_), X_train, y_train, cv=cv, method='predict_proba', n_jobs=1)` for the grid-search winners (LR: C = 1, l2, balanced; RF: 500 trees, depth 10, min leaf 20), `selection/notebooks/04_baselines_rq2.ipynb` cell index 7 (Cell 137); saved as `oof_lr`, `oof_rf` in `selection/predictions/sel_baseline_test_predictions.pkl`. After acceptance: the same 5-fold stratified split (seed 42) on the 119,999 training rows, LR = lbfgs, C = 100, l2, balanced, max_iter 5000 (`fit_converged`), RF = the locked 500/15/10 forest with n_jobs = 1, `selection/notebooks/08_story_extras.ipynb` cell index 2 (Cell 178); saved in `monitoring_rq3/mon_oof_predictions.pkl`. The locked imputer was fitted on all training rows (features only, no labels), so the After-acceptance OOF predictions are not fully out-of-sample with respect to imputation (README caveat).",
      "- **Objective and data.** For each r, cost `r x FN + FP` of 'flag if p >= t' on the out-of-fold **training** predictions (119,999 rows, `y_train`), not on the test set.",
      "- **Grid.** `GRID = np.round(np.arange(1, 100)/100, 2)` (0.01 to 0.99), `shared/src/decision_value.py::oof_tau`.",
      "- **Tie rule.** For the r = 1..20 curves (Fig 4, `sel_decision_value_combined.csv`): the **highest** threshold among ties (`tie='highest'`, 'ties to the highest'). The earlier single-r thresholds used the **lowest** among ties (`best[0]` in 04 cell index 7 for the Application r = 5 thresholds LR 0.90 and RF 0.69; 'lowest tau at the minimum' for the Application r = 10 thresholds LR 0.57 and RF 0.57 in `sel_rq3_thresholds_r10.json`). Both rules give the same value whenever the minimum is unique: %s." % ("the minimum is tied in %d of the 80 (stage, model, r) cells" % len(tie_rows) if tie_rows else "no cell has tied minima, so the tie rule never matters"),
      "- **BN.** Fixed a priori at 1/(1+r), no tuning.", "",
      "Stored thresholds used in Fig 4 (`tau` column of `sel_decision_value_combined.csv`):", "", *rows, "", checks_md(p), "",
      "**Deviations / notes.** The Expirements_FROZEN notebook itself is not in the repository; the cell content was read from the kernel history text and the recovered-source module, and the thresholds were reproduced from saved predictions rather than quoted. The ledger has no new rows for J4 (rules, not numbers).", ""]
p.write("\n".join(md))
