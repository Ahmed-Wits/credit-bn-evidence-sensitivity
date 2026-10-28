"""J17: dependence of the After-acceptance results on the past-due variables. Reads j17/*_fit.* (j17_fit.py), j17/mask_*.json (j17_mask.py); writes j17_*.csv, parts/j17.{json,md}, j17_snumbers_rows.csv.
Gates: the 'after_full' fit (iteration-100 BN, same code) reproduces the stored main After-acceptance results; the V2 linear LR equals Table 9 rung R2; the masking machinery reproduces the stored row-bootstrap rows."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from addcommon import Part, checks_md, sgn, LED
p = Part("j17"); J17 = ADD + "/j17"
VAR = ["after_full", "after_V1", "after_V2"]; LAB = {"after_full": "Main (all 10 variables)", "after_V1": "V1: drop 90+ Days Late (9 variables)", "after_V2": "V2: drop the three past-due counts (7 variables)"}
F = {v: load_fit(v, J17) for v in VAR}; short = lambda s: s.replace(" (bin)", "")
FR = pd.read_csv(ADD + "/flip_rate_by_r.csv"); FRa = FR[FR.stage.str.startswith("After")]
L = lambda f: pickle.load(open(R + "/" + f, "rb"))
# ------------------------------------------------------------------ gates for the main variant
fit0, info0, sres0 = F["after_full"]; y_te = fit0["yte"]; v0 = fit0["valid"]
d79 = L("full_variable/models/bn_final_model_iteration79.pkl"); stored_edges = {tuple(e) for e in d79["model_edges"]} - set(IND_EDGES)
p.gate("main variant: relearned structure equals the stored 22-edge After-acceptance structure (seed 53, BIC -1,026,115.7308)", {tuple(e) for e in info0["structure"]["edges"]} == stored_edges and info0["structure"]["best_seed"] == 53 and abs(info0["structure"]["best_bic"] + 1026115.7308) < 1e-2)
bb = L("full_variable/predictions/bn_test_predictions.pkl")["bn_test_probs"]; ab = L("full_variable/predictions/ablation_predictions.pkl")
vst = bb.notna().to_numpy(); p.gate("main variant: BN defined on the same 29,998 rows as the stored predictions", np.array_equal(v0, vst))
dd = np.abs(fit0["pbn"][v0] - bb.to_numpy(float)[v0]).max(); p.gate(f"main variant: iteration-100 BN posteriors equal the stored iteration-79 posteriors (max abs diff {dd:.2e})", dd <= 1e-9)
dl = np.abs(fit0["plr"] - np.asarray(ab["LR_all"], float)).max(); dr = np.abs(fit0["prf"] - np.asarray(ab["RF_all"], float)).max()
p.gate(f"main variant: LR and RF test probabilities equal the stored LR_all / RF_all (max abs diff {dl:.2e} / {dr:.2e})", dl <= 1e-6 and dr <= 1e-6)
p.gate("main variant: AUC BN/LR/RF = 0.8519 / 0.8618 / 0.8603", tuple(round(info0["auc"][m], 4) for m in ("BN", "LR", "RF")) == (0.8519, 0.8618, 0.8603))
led = {("BN", 5): 35.2, ("LR", 5): 35.0, ("RF", 5): 35.1, ("BN", 10): 47.7, ("LR", 10): 49.0, ("RF", 10): 49.1}
p.gate("main variant: Table 7 savings at r = 5, 10 (35.2/35.0/35.1, 47.7/49.0/49.1)", all(round(info0["saving"][f"{m}_r{r}"]["point"], 1) == x for (m, r), x in led.items()))
p.gate("main variant: blanket = {CUR, 30-59, 60-89, 90+}", set(map(short, info0["blanket"]["blanket"])) - {"M_Income", "M_Dependents"} == {"Credit Utilisation Rate", "30-59 Days Past Due", "60-89 Days Past Due", "90+ Days Late"})
j12 = json.load(open(f"{ADD}/j12/after_seed42_gate.json")); p.gate("main variant: LR/RF thresholds equal the J12 seed-42 run", info0["tau"] == j12["tau"])
edges_main = get_edges(get_split(42)[0]); p.check("bin edges of all variants equal the main-analysis edges (the five quantile-binned variables are retained in every variant; past-due counts use the fixed natural bins)", all(np.array_equal(F[v][1]["bin_edges"][c], [float(x) for x in edges_main[c]]) for v in VAR for c in EQ5))

# ------------------------------------------------------------------ flip rates, S_exp (with intervals)
def sens_tables(name):
    fit, info, sr = F[name]; rows = []
    mbv = [b for b in fit["blanket"] if b not in ("M_Income", "M_Dependents")]
    piv = {v: sr[sr.variable == v].set_index("row_index") for v in mbv}; idx = piv[mbv[0]].index
    for v in mbv: assert idx.equals(piv[v].index)
    sx = np.column_stack([piv[v].S_exp.to_numpy() for v in mbv]); n_ = len(idx)
    mats = [sx]; labels = [("S_exp", None)]
    for r in (5, 10, 15, 20):
        tau = 1 / (1 + r); mats.append(np.column_stack([((piv[v].baseline_prob.to_numpy() >= tau) != (piv[v].retracted_prob.to_numpy() >= tau)).astype(float) for v in mbv])); labels.append(("flip", r))
    bm = sens.boot_means(mats, B=2000, seed=42)
    for (kind, r), m_, b_ in zip(labels, mats, bm):
        lo, hi = np.percentile(b_, [2.5, 97.5], axis=0)
        for j, v in enumerate(mbv): rows.append(dict(variant=name, variable=short(v), quantity=kind, r=r if r else "", value=float(m_[:, j].mean()), lo=float(lo[j]), hi=float(hi[j]), n=n_))
    return pd.DataFrame(rows)
ST = {v: sens_tables(v) for v in VAR}
# gates against the stored numbers (main variant)
for _, rw in ST["after_full"].iterrows():
    if rw.quantity == "flip":
        s = FRa[(FRa.variable == rw.variable) & (FRa.r == rw.r)].iloc[0]
        p.gate(f"main variant: flip rate r={rw.r} {rw.variable} {100*rw.value:.2f}% [{100*rw.lo:.2f}, {100*rw.hi:.2f}] equals flip_rate_by_r.csv (1e-12)", abs(rw.value - s.flip_rate) < 1e-12 and abs(rw.lo - s.lower) < 1e-12 and abs(rw.hi - s.upper) < 1e-12)
for v, ref in {"Credit Utilisation Rate": 0.0280, "30-59 Days Past Due": 0.0245, "90+ Days Late": 0.0223, "60-89 Days Past Due": 0.0111}.items():
    x = ST["after_full"]; x = x[(x.variable == v) & (x.quantity == "S_exp")].value.iloc[0]; p.gate(f"main variant: mean S_exp {v} {x:.4f} = {ref}", round(x, 4) == ref)
pd.concat(ST.values()).to_csv(ADD + "/j17_sexp_flip_by_variant.csv", index=False)

# ------------------------------------------------------------------ AUC, savings table
sumrows = []
for name in VAR:
    fit, info, _ = F[name]
    row = dict(variant=name, label=LAB[name], n_variables=info["n_variables"], n_edges=info["structure"]["n_edges"], blanket=", ".join(sorted(short(b) for b in info["blanket"]["blanket"] if not b.startswith("M_"))),
               auc_BN=info["auc"]["BN"], auc_LR=info["auc"]["LR"], auc_RF=info["auc"]["RF"], lr_solver=info["lr_solver"], em_iterations=info["em"]["iterations"], n_bn_defined=info["n_bn_defined"])
    for m in ("LR", "RF"):
        d_ = info[f"auc_BN_minus_{m}"]; row.update({f"auc_BN_minus_{m}": d_["observed"], f"auc_BN_minus_{m}_lo": d_["lo"], f"auc_BN_minus_{m}_hi": d_["hi"]})
    for r in (5, 10):
        for m in ("BN", "LR", "RF"): s_ = info["saving"][f"{m}_r{r}"]; row.update({f"saving_{m}_r{r}": s_["point"], f"saving_{m}_r{r}_lo": s_["lo"], f"saving_{m}_r{r}_hi": s_["hi"]})
        for m in ("LR", "RF"): s_ = info["saving_diff"][f"BN_minus_{m}_r{r}"]; row.update({f"saving_diff_BN_minus_{m}_r{r}": s_["point"], f"saving_diff_BN_minus_{m}_r{r}_lo": s_["lo"], f"saving_diff_BN_minus_{m}_r{r}_hi": s_["hi"]})
    row.update({f"tau_{m}_r{r}": info["tau"][m][str(r)] for m in ("LR", "RF") for r in (5, 10)})
    sumrows.append(row)
SUM = pd.DataFrame(sumrows); SUM.to_csv(ADD + "/j17_variant_summary.csv", index=False)

# ------------------------------------------------------------------ linear LR information ladder (Table 9 settings: additive one-hot, balanced, l2, C = 1, lbfgs, checks.fit_converged, wrap binning)
g = get_g(); sp = pickle.load(open(R + "/Data/processed/split_indices.pkl", "rb")); ytr_ = np.asarray(sp["y_train"]).astype(int); yte_ = np.asarray(sp["y_test"]).astype(int)
APP = ["Age", "Debt Ratio", "Monthly Income", "Open Credit Lines", "Real Estate Loans", "Number of Dependents"]
LAD = {"R1 application six": APP, "V2 = R2: six + Credit Utilisation Rate": APP + ["Credit Utilisation Rate"], "V1: V2 + 30-59 + 60-89": APP + ["Credit Utilisation Rate", "30-59 Days Past Due", "60-89 Days Past Due"],
       "R3: six + CUR + three counts (no, = main minus nothing)": None, "R4/main: all ten": list(g["IMPUTE_ORDER"])}
ladrows = []; ladP = {}
for k, cols in LAD.items():
    if cols is None: continue
    d_ = g["build_design"](cols); m_ = ck.fit_converged(LogisticRegression(C=1, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=5000), d_["Xtr"].values, ytr_, name=k)
    pr = m_.predict_proba(d_["Xte"].values)[:, 1]; ladP[k] = pr; ladrows.append(dict(rung=k, n_variables=len(cols), auc=float(roc_auc_score(yte_, pr))))
LADdf = pd.DataFrame(ladrows)
stored = pd.read_csv(R + "/selection/tables/sel_info_ladder_table.csv").set_index("rung")
p.gate(f"linear LR with the six application variables + Credit Utilisation Rate reproduces Table 9 rung R2 ({LADdf.auc[1]:.6f} vs {stored.loc['R2','auc']:.6f})", abs(LADdf.auc[1] - stored.loc["R2", "auc"]) < 1e-9)
p.gate(f"linear LR on the six application variables reproduces Table 9 rung R1 ({LADdf.auc[0]:.6f}); on all ten reproduces R4 ({LADdf.auc[3]:.6f})", abs(LADdf.auc[0] - stored.loc["R1", "auc"]) < 1e-9 and abs(LADdf.auc[3] - stored.loc["R4", "auc"]) < 1e-9)
rng_ = np.random.RandomState(42); bt_ = {k: np.empty(2000) for k in ladP}
for i in range(2000):
    ix = rng_.randint(0, len(yte_), len(yte_)); yy = yte_[ix]
    for k in ladP: bt_[k][i] = roc_auc_score(yy, ladP[k][ix])
LADdf["lo"] = [np.percentile(bt_[k], 2.5) for k in LADdf.rung]; LADdf["hi"] = [np.percentile(bt_[k], 97.5) for k in LADdf.rung]
LADdf.to_csv(ADD + "/j17_linear_lr_ladder.csv", index=False)

# ------------------------------------------------------------------ descriptive leakage diagnostics (main split; no fitting)
dfc_ = g["dfc"]; tr_i, te_i = g["train_idx"], g["test_idx"]; leak = []
for c in ["Credit Utilisation Rate", "30-59 Days Past Due", "60-89 Days Past Due", "90+ Days Late"]:
    lo_ = None
    a_te = float(roc_auc_score(yte_, dfc_.loc[te_i, c].to_numpy(float))); a_tr = float(roc_auc_score(ytr_, dfc_.loc[tr_i, c].to_numpy(float)))
    rg = np.random.RandomState(42); xs = dfc_.loc[te_i, c].to_numpy(float); bs = np.empty(2000)
    for i in range(2000): ix = rg.randint(0, len(yte_), len(yte_)); bs[i] = roc_auc_score(yte_[ix], xs[ix])
    leak.append(dict(item="single-variable AUC (raw value as the score)", variable=c, test_auc=a_te, test_lo=float(np.percentile(bs, 2.5)), test_hi=float(np.percentile(bs, 97.5)), train_auc=a_tr))
x90 = dfc_["90+ Days Late"]; yall = dfc_["Default"].astype(int)
for lab, idx_, yy in (("training rows", tr_i, ytr_), ("test rows", te_i, yte_), ("all 149,999 rows", dfc_.index.to_numpy(), yall.to_numpy())):
    x = x90.loc[idx_].to_numpy(); z = x == 0; o = x >= 1
    leak.append(dict(item=f"default rate, 90+ Days Late = 0 vs >= 1 ({lab})", variable="90+ Days Late", rate_zero=float(yy[z].mean()), n_zero=int(z.sum()), rate_ge1=float(yy[o].mean()), n_ge1=int(o.sum()), share_defaulters_ge1=float((o & (yy == 1)).sum() / (yy == 1).sum()), n_defaulters=int((yy == 1).sum())))
for c in ["30-59 Days Past Due", "60-89 Days Past Due"]:
    for lab, idx_, yy in (("training rows", tr_i, ytr_),):
        x = dfc_[c].loc[idx_].to_numpy(); z = x == 0; o = x >= 1
        leak.append(dict(item=f"default rate, {c} = 0 vs >= 1 ({lab})", variable=c, rate_zero=float(yy[z].mean()), n_zero=int(z.sum()), rate_ge1=float(yy[o].mean()), n_ge1=int(o.sum()), share_defaulters_ge1=float((o & (yy == 1)).sum() / (yy == 1).sum()), n_defaulters=int((yy == 1).sum())))
LK = pd.DataFrame(leak); LK.to_csv(ADD + "/j17_leakage_diagnostics.csv", index=False)
past = ["30-59 Days Past Due", "60-89 Days Past Due", "90+ Days Late"]
STRONG = max(past, key=lambda c: float(roc_auc_score(ytr_, dfc_.loc[tr_i, c].to_numpy(float))))     # rule fixed in advance: largest TRAINING-set single-variable AUC; used by J19's decision-level labels
json.dump(dict(strongest_past_due_by_training_auc=STRONG), open(ADD + "/j19/strongest_past_due.json", "w"))

# ------------------------------------------------------------------ masking results
MK_ = {}
for name in VAR:
    for f in sorted(os.listdir(J17)):
        if f.startswith(f"mask_{name}_") and f.endswith(".json"):
            for k, v in json.load(open(f"{J17}/{f}")).items(): MK_[(name, v["variable"], v["level"], v["mechanism"])] = v
mrows = []
for (name, var, lev, mech), v in sorted(MK_.items()):
    for m in ("LR", "RF"):
        r = v["res"][m]; mrows.append(dict(variant=name, variable=var, level=lev, mechanism=mech, contrast=f"BN minus {m}", point=r["point"], boot_mean=r["boot_mean"], lo=r["lo"], hi=r["hi"], reading=reading(r["lo"], r["hi"]), deg_BN=v["deg"]["BN"], deg_comp=v["deg"][m]))
MKdf = pd.DataFrame(mrows); MKdf.to_csv(ADD + "/j17_masking_contrasts.csv", index=False)
# masking gate: the main-variant machinery reproduces the stored monitoring row-bootstrap rows
ref = pd.read_csv(ADD + "/mon_rq3_row_bootstrap_summary.csv"); ref["level"] = ref.level.round(2); nchk = 0; worst = 0.0
for _, rr in ref.iterrows():
    k = ("after_full", rr.variable, rr.level, rr.mechanism)
    if k in MK_:
        m = rr.contrast.split()[-1]; r = MK_[k]["res"][m]; worst = max(worst, abs(r["point"] - rr.point_estimate), abs(r["lo"] - rr.ci_lo), abs(r["hi"] - rr.ci_hi)); nchk += 1
if nchk: p.gate(f"masking machinery on the refit main model reproduces the stored monitoring row-bootstrap summary ({nchk} rows; point, lower, upper; max abs diff {worst:.2e}, tol 1e-6; LR and RF probabilities are identical to the stored ones, the residual comes from the BN: the stored masked BN is the iteration-79 model, this one iteration 100, and the CPDs that differ between the two iterations (Real Estate Loans, Monthly Income, Number of Dependents) can enter a posterior once a blanket variable is removed)", worst < 1e-6)

print(SUM.T.to_string()); print(LADdf); print(LK.to_string()); print(MKdf.round(4).to_string())
pickle.dump(dict(ST=ST, SUM=SUM, LAD=LADdf, LK=LK, MK=MKdf, STRONG=STRONG, checks=p.checks), open(J17 + "/j17_summary.pkl", "wb"))
p.write("(report text written by j17_report.py)")
