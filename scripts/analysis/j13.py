"""J13: calibrated comparators. Platt scaling of LR and RF fitted on the 5-fold out-of-fold TRAINING predictions only; flag at 1/(1+r) like the BN."""
from addcommon import *
import pickle
from costlib import *
from sklearn.linear_model import LogisticRegression
p = Part("j13")
L = lambda f: pickle.load(open(R + "/" + f, "rb"))
comb = pd.read_csv(R + "/selection/tables/sel_decision_value_combined.csv")
sp = L("Data/processed/split_indices.pkl"); ytr = np.asarray(sp["y_train"]).astype(int); yte = np.asarray(sp["y_test"]).astype(int)
bp = L("selection/predictions/sel_baseline_test_predictions.pkl"); sb = L("selection/predictions/sel_bn_test_predictions.pkl")
bb = L("full_variable/predictions/bn_test_predictions.pkl")["bn_test_probs"]; ab = L("full_variable/predictions/ablation_predictions.pkl"); oofm = L("monitoring_rq3/mon_oof_predictions.pkl")["oof"]; v = bb.notna().to_numpy()
STAGES = {"Application": dict(key="selection", y=np.asarray(sb["y"]).astype(int), BN=np.asarray(sb["p"], float), LR=np.asarray(bp["p_lr"], float), RF=np.asarray(bp["p_rf"], float), oof={"LR": np.asarray(bp["oof_lr"], float), "RF": np.asarray(bp["oof_rf"], float)}, mask=np.ones(30000, bool)),
          "After acceptance": dict(key="monitoring", y=yte, BN=bb.to_numpy(float), LR=np.asarray(ab["LR_all"], float), RF=np.asarray(ab["RF_all"], float), oof={"LR": np.asarray(oofm["LR"], float), "RF": np.asarray(oofm["RF"], float)}, mask=v)}
logit = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
RS = [5, 10]; res = {}; v4 = pd.read_csv(ADD + "/ledger_additions_v4.csv", dtype=str, keep_default_na=False)
def nums(d): return [float(x) for x in d.replace(MINUS, "-").replace(" pts", "").replace("%", "").replace("[", "").replace("]", "").replace(",", "").split()]
for stage, S in STAGES.items():
    m = S["mask"]; y = S["y"][m]; P0 = {k: S[k][m] for k in ("BN", "LR", "RF")}
    T0 = {k: {r: float(comb[(comb.contribution == S["key"]) & (comb.model == k) & (comb.r == r)].tau.iloc[0]) for r in RS} for k in ("BN", "LR", "RF")}
    # gate: original savings with intervals equal the J2 ledger rows
    pt0 = savings_point(y, P0, T0, RS); W = boot_weights(len(y)); bt0 = saving_boot(y, P0, T0, RS, W=W)
    for k in ("BN", "LR", "RF"):
        for j, r in enumerate(RS):
            row = v4[v4.quantity.str.startswith("Cost saving vs the cheaper trivial policy with 95% interval") & v4.quantity.str.contains(f", {k}, {stage}, r={r}")].iloc[0]
            pv, lo, hi = nums(row.display); lo_, hi_ = pct_ci(100 * bt0[k][:, j])
            p.gate(f"{stage} {k} r={r}: original saving and interval reproduce the J2 ledger row {row.ledger_id} ({row.display})", abs(round(100 * pt0[(k, r)][0], 1) - pv) < 1e-9 and abs(round(lo_, 1) - lo) < 1e-9 and abs(round(hi_, 1) - hi) < 1e-9)
    # Platt scaling on OOF training predictions
    cal = {}; P1 = {"BN": P0["BN"]}; info = {}
    for k in ("LR", "RF"):
        z = logit(S["oof"][k]).reshape(-1, 1); pl = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, ytr); a, b = float(pl.coef_[0, 0]), float(pl.intercept_[0])
        P1[k] = 1 / (1 + np.exp(-(a * logit(P0[k]) + b)))
        info[k] = dict(a=a, b=b, mean_oof=float(S["oof"][k].mean()), mean_cal_test=float(P1[k].mean()), mean_raw_test=float(P0[k].mean()), obs_test=float(y.mean()), auc_raw=float(roc_auc_score(y, P0[k])), auc_cal=float(roc_auc_score(y, P1[k])))
        p.check(f"{stage} {k}: AUC unchanged by the monotone calibration (|diff| = {abs(info[k]['auc_raw']-info[k]['auc_cal']):.1e}, tolerance 1e-12)", abs(info[k]["auc_raw"] - info[k]["auc_cal"]) <= 1e-12, f"{info[k]['auc_raw']:.6f} -> {info[k]['auc_cal']:.6f}")
        p.check(f"{stage} {k}: Platt slope {a:.3f} (> 0) and the calibrated mean probability on the test rows ({info[k]['mean_cal_test']:.4f}) is close to the observed rate ({info[k]['obs_test']:.4f})", a > 0 and abs(info[k]["mean_cal_test"] - info[k]["obs_test"]) < 0.01)
    T1 = {k: {r: 1 / (1 + r) for r in RS} for k in ("BN", "LR", "RF")}
    pt1 = savings_point(y, P1, T1, RS); bt1 = saving_boot(y, P1, T1, RS, W=W); S1 = summarise(pt1, bt1, ["BN", "LR", "RF"], RS)
    res[stage] = dict(S1=S1, info=info, n=len(y), pt0=pt0, pt1=pt1)
# ---- rows
SRC = "Submissions/4. Article/WRITEUP/analysis_additions/code/j13.py"
CIM = "95% percentile interval, 2000 test-row bootstrap resamples (seed 42), paired across models; predictions and thresholds fixed (J2 procedure)"
for stage, Rr in res.items():
    st_led = "selection" if stage == "Application" else "monitoring"
    for _, x in Rr["S1"].iterrows():
        if x.kind == "saving" and x.model == "BN": continue
        if x.kind == "saving":
            p.row(st_led, f"Cost saving with Platt-calibrated {x.model} flagged at 1/(1+r), {stage}, r={int(x.r)}", x.point / 100, f"{x.point:.1f}% [{x.lo:.1f}%, {x.hi:.1f}%]", "%", SRC, f"stage={stage} | model={x.model} (Platt-calibrated on OOF training predictions) | r={int(x.r)} | tau=1/(1+r)", derived="Y", ci_lo=x.lo / 100, ci_hi=x.hi / 100, ci_method=CIM, n=Rr["n"], model=x.model, level=f"r={int(x.r)}",
                  notes=f"Platt scaling a={Rr['info'][x.model]['a']:.4f}, b={Rr['info'][x.model]['b']:.4f} fitted on the 5-fold out-of-fold training predictions only (logit of the OOF probability); same threshold 1/(1+r) as the BN; original-threshold saving {100*Rr['pt0'][(x.model, int(x.r))][0]:.1f}%; AUC unchanged", display_rule="pct1", caveat="C3;C5")
        else:
            lo_, hi_ = x.lo, x.hi
            p.row(st_led, f"Cost saving difference BN minus Platt-calibrated {x.comparator}, {stage}, r={int(x.r)} (percentage points)", x.point, f"{sgn(x.point,2,True)} [{sgn(x.lo,2)}, {sgn(x.hi,2)}]", "percentage points", SRC, f"stage={stage} | BN minus calibrated {x.comparator} | r={int(x.r)}", derived="Y", ci_lo=x.lo, ci_hi=x.hi, ci_method=CIM, n=Rr["n"], model="BN", comparator=x.comparator, level=f"r={int(x.r)}", tested=("yes-tie" if x.lo <= 0 <= x.hi else "yes-significant"),
                  notes="LR/RF calibrated by Platt scaling on out-of-fold training predictions and flagged at 1/(1+r) like the BN; J2 bootstrap", display_rule="txt", caveat="C3;C5")
    for k in ("LR", "RF"):
        i = Rr["info"][k]
        p.row(st_led, f"Platt calibration of {k}, {stage}: mean calibrated probability on the test rows", i["mean_cal_test"], f"{i['mean_cal_test']:.4f}", "probability", SRC, f"stage={stage} | {k} | mean of the calibrated test probabilities", derived="Y", notes=f"observed test default rate {i['obs_test']:.4f}; mean raw (balanced-weight) probability {i['mean_raw_test']:.4f}; AUC raw {i['auc_raw']:.6f}, calibrated {i['auc_cal']:.6f}", display_rule="p4", n=Rr["n"], model=k, caveat="C5")
def tab(stage):
    Rr = res[stage]; S1 = Rr["S1"]; L_ = ["| Model | r = 5 | r = 10 |", "|---|---|---|"]
    for k in ("LR", "RF"):
        cells = []
        for r in RS:
            x = S1[(S1.kind == "saving") & (S1.model == k) & (S1.r == r)].iloc[0]; o = Rr["pt0"][(k, r)][0] * 100
            cells.append(f"{x.point:.1f}% [{x.lo:.1f}, {x.hi:.1f}] (original threshold {o:.1f}%)")
        L_.append(f"| {k} calibrated | " + " | ".join(cells) + " |")
    for k in ("LR", "RF"):
        cells = []
        for r in RS:
            x = S1[(S1.kind == "difference") & (S1.comparator == k) & (S1.r == r)].iloc[0]; o = (Rr["pt0"][("BN", r)][0] - Rr["pt0"][(k, r)][0]) * 100
            cells.append(f"{sgn(x.point,2,True)} [{sgn(x.lo,2)}, {sgn(x.hi,2)}] (original {sgn(o,2,True)})")
        L_.append(f"| BN minus {k} calibrated | " + " | ".join(cells) + " |")
    bn = [S1[(S1.kind == "saving") & (S1.model == "BN") & (S1.r == r)].iloc[0] for r in RS]
    L_.append("| BN (unchanged) | " + " | ".join(f"{x.point:.1f}% [{x.lo:.1f}, {x.hi:.1f}]" for x in bn) + " |")
    return L_
md = ["## J13. Calibrated comparators (Platt scaling)", "",
      "**Method.** For LR and RF, a Platt map sigma(a x logit(p) + b) is fitted by (essentially unpenalised) logistic regression of the training labels on the logit of the 5-fold **out-of-fold training** predictions (Application: `oof_lr`, `oof_rf` in `sel_baseline_test_predictions.pkl`; After acceptance: `monitoring_rq3/mon_oof_predictions.pkl`; no test data). The map is applied to the test probabilities of the final models, which are then flagged at 1/(1+r) exactly as the BN (cost r x FN + FP, saving against the cheaper trivial policy; J2 pipeline and bootstrap, After acceptance on the 29,998 rows where the BN is defined). The calibration is monotone, so the AUC is unchanged (asserted to 1e-12).", "",
      "**Gate (PASSED).** With the original thresholds the savings and their intervals for the three models at r = 5 and 10 reproduce the J2 ledger rows (N-ADD-010 to 021) at both stages.", "",
      "### Application", "", *tab("Application"), "", "### After acceptance", "", *tab("After acceptance"), "", "Calibration summary (mean probability on the test rows):", "", "| Stage | Model | raw mean | calibrated mean | observed rate | slope a | intercept b |", "|---|---|---|---|---|---|---|"] + \
     [f"| {st} | {k} | {res[st]['info'][k]['mean_raw_test']:.4f} | {res[st]['info'][k]['mean_cal_test']:.4f} | {res[st]['info'][k]['obs_test']:.4f} | {res[st]['info'][k]['a']:.3f} | {res[st]['info'][k]['b']:.3f} |" for st in res for k in ("LR", "RF")] + ["", checks_md(p), "",
      "**Notes.** (a) The threshold rule for LR and RF changes from the out-of-fold cost-minimising grid value to the Bayes threshold 1/(1+r) applied to calibrated probabilities. (b) The calibration map is fitted on out-of-fold predictions but applied to the full-training fit, the usual practice; After acceptance the OOF imputer was fitted on all training rows (features only, README caveat). (c) Intervals cover test-row sampling with models, calibration maps and thresholds fixed. (d) At Application the balanced-weight probabilities are far above the observed rate (about 0.45); the calibrated means are near 0.067.", ""]
p.write("\n".join(md))
for st in res: print(st, {k: (round(100*res[st]['pt0'][(k,5)][0],2), round(100*res[st]['pt1'][(k,5)][0],2), round(100*res[st]['pt0'][(k,10)][0],2), round(100*res[st]['pt1'][(k,10)][0],2)) for k in ("LR","RF")})
