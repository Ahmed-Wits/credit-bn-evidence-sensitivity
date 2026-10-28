"""J16: blanket-only linear LR check (supports Section 5.1.2). Same code and settings as the Table 9 ladder (08_story_extras cell index 3 / Cell 179):
additive one-hot bins, balanced class weights, l2, C = 1, converged lbfgs refit (checks.fit_converged), 'wrap' binning of rq3 v2, main split (seed 42). No test-set choices."""
from addcommon import *
import importlib.util, pickle, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, R + "/shared/src"); sys.path.insert(0, ADD + "/code")
from costlib import paired_boot_full
def load(name, rel):
    s = importlib.util.spec_from_file_location(name, R + "/" + rel); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ck = load("checks", "shared/src/checks.py"); rq3 = load("rq3", "shared/src/rq3.py")
p = Part("j16"); L = lambda f: pickle.load(open(R + "/" + f, "rb")); PROC = R + "/Data/processed/"
sp = L("Data/processed/split_indices.pkl"); y_tr = np.asarray(sp["y_train"]).astype(int); y_te = np.asarray(sp["y_test"]).astype(int)
g = {"os": os, "np": np, "pd": pd, "L": L if False else (lambda q: pickle.load(open(q, "rb"))), "PROC": PROC, "ROOT": R}
g["train_idx"], g["test_idx"] = np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"]); g["y_train"], g["y_test"] = sp["y_train"], sp["y_test"]
import json as _j
code = "".join(_j.load(open(R + "/selection/notebooks/00_precheck.ipynb"))["cells"][2]["source"])
exec(compile(code[:code.index("# ---- validation")], "build_design_src", "exec"), g)
g["bin_natural_category"] = lambda s, max_category=3: rq3._bin_natural(s, "wrap", max_category)
bd = g["build_design"]
RUNG = {"R0": ["Age", "Debt Ratio"], "BL4": ["Credit Utilisation Rate", "30-59 Days Past Due", "60-89 Days Past Due", "90+ Days Late"], "R4": list(g["IMPUTE_ORDER"])}
P = {}; NIT = {}
for k, cols in RUNG.items():
    d = bd(cols); m = ck.fit_converged(LogisticRegression(C=1, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=5000), d["Xtr"].values, y_tr, name=k)
    P[k] = m.predict_proba(d["Xte"].values)[:, 1]; NIT[k] = int(np.max(m.n_iter_)); p.check(f"{k} fitted: {d['Xtr'].shape[1]} one-hot columns, {len(cols)} variables, converged in {NIT[k]} lbfgs iterations", True)
auc = {k: float(roc_auc_score(y_te, v)) for k, v in P.items()}
# ---------------------------------------------------------------- gates (consistency with Table 9)
lad = pd.read_csv(R + "/selection/tables/sel_info_ladder_table.csv").set_index("rung")
for k, lid in (("R0", "N-XST-023"), ("R4", "N-XST-027")):
    p.gate(f"{k} test AUC {auc[k]:.6f} equals the stored ladder value {lad.loc[k,'auc']:.6f} and ledger {lid} ({LED.loc[lid,'display'].split(' ')[0]})", abs(auc[k] - lad.loc[k, "auc"]) < 1e-9 and f"{auc[k]:.4f}" == LED.loc[lid, "display"].split(" ")[0])
n = len(y_te); rng = np.random.RandomState(42); boot = {k: np.empty(2000) for k in P}
for i in range(2000):
    ix = rng.randint(0, n, n); yy = y_te[ix]
    for k in P: boot[k][i] = roc_auc_score(yy, P[k][ix])
ci = {k: tuple(np.percentile(boot[k], [2.5, 97.5])) for k in P}
for k, lid in (("R0", "N-XST-023"), ("R4", "N-XST-027")):
    disp = LED.loc[lid, "display"]; lo, hi = [float(x) for x in disp.split("[")[1].rstrip("]").split(",")]
    p.gate(f"{k} AUC interval [{ci[k][0]:.4f}, {ci[k][1]:.4f}] equals ledger {lid} ({disp})", abs(round(ci[k][0], 4) - lo) < 1e-9 and abs(round(ci[k][1], 4) - hi) < 1e-9)
# stored application analogue and the earlier After-acceptance run
sb = L("selection/predictions/sel_bn_test_predictions.pkl"); yA = np.asarray(sb["y"]).astype(int); pbnA = np.asarray(sb["p"], float)
diag = pd.read_csv(R + "/selection/tables/sel_additive_lbfgs_diagnostic.csv").iloc[0]
obsA, difA = paired_boot_full(yA, pbnA, P["R0"]); loA, hiA = np.percentile(difA, [2.5, 97.5])
p.gate(f"Application: BN minus blanket-only LR reproduces the stored diagnostic sel_additive_lbfgs_diagnostic.csv ({diag['bn_minus_additive']:+.6f} [{diag['ci_lo']:+.6f}, {diag['ci_hi']:+.6f}])", abs(obsA - diag["bn_minus_additive"]) < 1e-9 and abs(loA - diag["ci_lo"]) < 1e-9 and abs(hiA - diag["ci_hi"]) < 1e-9, f"recomputed {obsA:+.6f} [{loA:+.6f}, {hiA:+.6f}]")
bb = L("full_variable/predictions/bn_test_predictions.pkl")["bn_test_probs"]; ab = L("full_variable/predictions/ablation_predictions.pkl"); v = bb.notna().to_numpy(); yv = y_te[v]; BN = bb.to_numpy(float)[v]
PRINTED_IS = {}
o1, d1 = paired_boot_full(yv, BN, np.asarray(ab["LR_4var"], float)[v]); o2, d2 = paired_boot_full(y_te, np.asarray(ab["LR_all"], float), np.asarray(ab["LR_4var"], float))
for lid, o, d_ in (("N-MON-044", o1, d1), ("N-MON-043", o2, d2)):
    disp = LED.loc[lid, "display"]; pv, lo, hi = [float(x) for x in disp.replace(MINUS, "-").replace("[", "").replace("]", "").replace(",", "").split()]
    is_obs = abs(round(o, 4) - pv) < 1e-9; is_mean = abs(round(d_.mean(), 4) - pv) < 1e-9
    PRINTED_IS[lid] = "observed difference" if is_obs and not is_mean else ("bootstrap mean" if is_mean and not is_obs else "both (identical at the printed precision)")
    p.gate(f"earlier After-acceptance run reproduces ledger {lid} ({disp}) from ablation_predictions.pkl: interval [{np.percentile(d_,2.5):+.4f}, {np.percentile(d_,97.5):+.4f}]; printed point = {PRINTED_IS[lid]} (observed {o:+.6f}, bootstrap mean {d_.mean():+.6f})", (is_obs or is_mean) and abs(round(np.percentile(d_, 2.5), 4) - lo) < 1e-9 and abs(round(np.percentile(d_, 97.5), 4) - hi) < 1e-9)
p.gate(f"earlier LR_4var AUC {roc_auc_score(y_te, np.asarray(ab['LR_4var'], float)):.4f} equals ledger N-MON-037 ({LED.loc['N-MON-037','display']})", f"{roc_auc_score(y_te, np.asarray(ab['LR_4var'], float)):.4f}" == LED.loc["N-MON-037", "display"])
# ---------------------------------------------------------------- new numbers
oB, dB = paired_boot_full(yv, BN, P["BL4"][v]); loB, hiB = np.percentile(dB, [2.5, 97.5])                  # (b) BN minus blanket-only LR, 29,998 rows
oC, dC = paired_boot_full(y_te, P["R4"], P["BL4"]); loC, hiC = np.percentile(dC, [2.5, 97.5])               # (c) all ten minus blanket-only, 30,000 rows
oD, dD = paired_boot_full(yv, BN, P["R4"][v]); loD, hiD = np.percentile(dD, [2.5, 97.5])                    # context: BN minus all-ten linear LR, 29,998 rows
aucBL4_v = float(roc_auc_score(yv, P["BL4"][v])); aucBN_v = float(roc_auc_score(yv, BN)); aucBN_A = float(roc_auc_score(yA, pbnA))
inc = lambda lo, hi: lo <= 0 <= hi
res = dict(A=(obsA, loA, hiA, float(difA.mean())), B=(oB, loB, hiB, float(dB.mean())), C=(oC, loC, hiC, float(dC.mean())), D=(oD, loD, hiD, float(dD.mean())))
print({k: tuple(round(x, 4) for x in v_) for k, v_ in res.items()}, auc, aucBL4_v, aucBN_v)
pd.DataFrame([dict(item="a_R0_auc", value=auc["R0"], lo=ci["R0"][0], hi=ci["R0"][1]), dict(item="a_BN_minus_R0", value=obsA, lo=loA, hi=hiA), dict(item="b_BL4_auc", value=auc["BL4"], lo=ci["BL4"][0], hi=ci["BL4"][1]),
              dict(item="b_BN_minus_BL4", value=oB, lo=loB, hi=hiB), dict(item="c_R4_minus_BL4", value=oC, lo=loC, hi=hiC), dict(item="ctx_BN_minus_R4", value=oD, lo=loD, hi=hiD)]).to_csv(ADD + "/j16_blanket_only_lr.csv", index=False)
# ---------------------------------------------------------------- ledger rows
SRC = "Submissions/4. Article/WRITEUP/analysis_additions/j16_blanket_only_lr.csv"
CIM = "95% percentile interval, 2000 paired test-row bootstrap resamples (RandomState(42), randint), as Table 4; observed difference printed"
CIA = "95% bootstrap CI as recorded (percentile of 2000 test-row resamples, seed 42), as the Table 9 ladder"
SET = "linear LR as the Table 9 ladder: additive one-hot bins (first bin dropped), balanced class weights, l2, C = 1, lbfgs refit with checks.fit_converged (converged in %d iterations), 'wrap' binning, main split seed 42, no test-set choice; code analysis_additions/code/j16.py"
p.row("selection", "Linear LR on Age and Debt Ratio only (the blanket variables), Application: paired AUC difference, BN minus this LR, with 95% interval", obsA, f"{sgn(obsA,4)} [{sgn(loA,4)}, {sgn(hiA,4)}]", "AUC difference", SRC, "BN minus R0 (Age + Debt Ratio) | observed | percentile interval", derived="Y", ci_lo=loA, ci_hi=hiA, ci_method=CIM, n=30000, model="BN", comparator="LR (Age, Debt Ratio only)", tested=("yes-tie" if inc(loA, hiA) else "yes-significant"), caveat="C4;C5",
      notes=SET % NIT["R0"] + f"; reproduces the stored diagnostic sel_additive_lbfgs_diagnostic.csv (+0.004498 [+0.001073, +0.007783], not previously in the ledger); the LR AUC is {auc['R0']:.4f} (N-XST-023), the BN AUC {aucBN_A:.4f}; bootstrap mean {res['A'][3]:+.6f}", display_rule="d4")
p.row("monitoring", "Linear LR on the four blanket variables only, After acceptance: test AUC with 95% interval", auc["BL4"], f"{auc['BL4']:.4f} [{ci['BL4'][0]:.4f}, {ci['BL4'][1]:.4f}]", "AUC", SRC, "BL4 | auc | percentile interval", derived="Y", ci_lo=float(ci["BL4"][0]), ci_hi=float(ci["BL4"][1]), ci_method=CIA, n=30000, model="LR", caveat="C4;C5",
      notes=SET % NIT["BL4"] + f"; variables Credit Utilisation Rate, 30-59 Days Past Due, 60-89 Days Past Due, 90+ Days Late; AUC on the 29,998 rows scored by the BN {aucBL4_v:.4f}; the earlier binned LR (saga, C = 100) on the same variables has AUC 0.8508 (N-MON-037)", display_rule="auc4")
p.row("monitoring", "Linear LR on the four blanket variables only, After acceptance: paired AUC difference, BN minus this LR (29,998 shared rows), with 95% interval", oB, f"{sgn(oB,4)} [{sgn(loB,4)}, {sgn(hiB,4)}]", "AUC difference", SRC, "BN minus BL4 | observed | percentile interval", derived="Y", ci_lo=loB, ci_hi=hiB, ci_method=CIM, n=29998, model="BN", comparator="LR (four blanket variables)", tested=("yes-tie" if inc(loB, hiB) else "yes-significant"), caveat="C4;C5",
      notes=SET % NIT["BL4"] + f"; BN AUC on these rows {aucBN_v:.4f}, LR {aucBL4_v:.4f}; bootstrap mean {res['B'][3]:+.6f}; the earlier binned-LR version of this comparison is N-MON-044 (+0.0013 [-0.0001, +0.0026], printed as a bootstrap mean)", display_rule="d4")
p.row("monitoring", "Linear LR on all ten variables minus linear LR on the four blanket variables, After acceptance: paired AUC difference with 95% interval (30,000 rows)", oC, f"{sgn(oC,4)} [{sgn(loC,4)}, {sgn(hiC,4)}]", "AUC difference", SRC, "R4 minus BL4 | observed | percentile interval", derived="Y", ci_lo=loC, ci_hi=hiC, ci_method=CIM, n=30000, model="LR", comparator="LR (four blanket variables)", tested=("yes-tie" if inc(loC, hiC) else "yes-significant"), caveat="C4;C5",
      notes=SET % NIT["R4"] + f"; all-ten LR AUC {auc['R4']:.4f} (N-XST-027), blanket-only {auc['BL4']:.4f}; positive = the ten-variable LR is higher; bootstrap mean {res['C'][3]:+.6f}; the earlier binned-LR version is N-MON-043 (+0.0110 [+0.0075, +0.0146])", display_rule="d4")
p.row("monitoring", "Linear LR on all ten variables, After acceptance: paired AUC difference, BN minus this LR (29,998 shared rows), with 95% interval", oD, f"{sgn(oD,4)} [{sgn(loD,4)}, {sgn(hiD,4)}]", "AUC difference", SRC, "BN minus R4 | observed | percentile interval", derived="Y", ci_lo=loD, ci_hi=hiD, ci_method=CIM, n=29998, model="BN", comparator="LR (all ten variables, linear, C = 1)", tested=("yes-tie" if inc(loD, hiD) else "yes-significant"), caveat="C4;C5",
      notes=SET % NIT["R4"] + f"; context for the comparison with the blanket-only LR; bootstrap mean {res['D'][3]:+.6f}", display_rule="d4")
# verdict
bl_ok = inc(loB, hiB); ten_higher = (loD > 0) is False and hiD < 0
md = ["## J16. Blanket-only linear LR check (Section 5.1.2)", "",
      "### Step 1: what already exists", "",
      "- **After acceptance: yes, an earlier run exists.** `ablation_predictions.pkl` holds `LR_4var`, the binned LR (saga, C = 100, the locked LR settings) on the one-hot columns of the four blanket variables, fitted in `Expirements_FROZEN` (cells 62/67/72; stored output). Ledger rows: test AUC **0.8508** (N-MON-037); BN minus this LR **+0.0013 [-0.0001, +0.0026]** on the 29,998 shared rows (N-MON-044; interval includes 0); LR on all ten variables minus this LR **+0.0110 [+0.0075, +0.0146]** (N-MON-043; excludes 0). The claim ledger lists them as CL-M11 ('discussion only'). **These rows are in the ledger but not in Table S-Numbers**, which lists only N-MON-038 to 042 of the ablation rows; that is why they could not be found there. Reproduced here from the saved predictions (gate below). **Note on the ledger text:** the ledger says the printed estimates of N-MON-043/044 are bootstrap means; for N-MON-044 the printed +0.0013 is the OBSERVED difference (observed 0.001259, bootstrap mean 0.001231 = +0.0012); N-MON-043 prints the same value either way (+0.0110).",
      f"- **Application: an analogue exists but is not in the ledger or the supplement.** Table 9 row R0 is a linear LR on Age and Debt Ratio only (the blanket) with AUC 0.6582 [0.6467, 0.6699] (N-XST-023). `selection/tables/sel_additive_lbfgs_diagnostic.csv` (the converged lbfgs refit that replaced the non-converged saga row of `sel_baselines_bootstrap.csv`) stores **BN minus this LR = +0.004498 [+0.001073, +0.007783]** (observed difference, bootstrap 2,000, seed 42), i.e. the interval excludes 0 and the BN is higher; it has no ledger row, and the claim ledger (gap G11) records that the application analogue 'has not been run'. The statement 'the same check was not run at application' is therefore not accurate: it was run as a diagnostic but never entered the ledger. (The superseded saga row `BN - LR_additive (diagnostic)` = +0.0399 came from a fit that hit max_iter.)", "",
      "### Step 2: recomputation with the Table 9 ladder settings (main split)", "",
      f"Settings: the ladder code (`08_story_extras` cell index 3) fits each rung with `LogisticRegression(C=1, penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=5000)` through `checks.fit_converged`, additive one-hot bins (first bin dropped), 'wrap' binning, no test-set choice. **Note:** the ladder uses the converged lbfgs refit, not saga as in the request; saga did not converge for R0 at max_iter 1000 (the reason the diagnostic file exists), so the ladder settings were reproduced exactly. All fits converged (lbfgs iterations: R0 {NIT['R0']}, blanket-only After acceptance {NIT['BL4']}, all ten {NIT['R4']}).", "",
      "| Quantity | Value | 95% interval | Interval includes 0? |", "|---|---|---|---|",
      f"| (a) Application: linear LR on Age and Debt Ratio only, test AUC | {auc['R0']:.4f} | [{ci['R0'][0]:.4f}, {ci['R0'][1]:.4f}] | n/a |",
      f"| (a) Application: BN minus this LR (30,000 rows; BN AUC {aucBN_A:.4f}) | {sgn(obsA,4)} | [{sgn(loA,4)}, {sgn(hiA,4)}] | {'yes' if inc(loA, hiA) else 'no (BN higher)'} |",
      f"| (b) After acceptance: linear LR on the four blanket variables, test AUC (30,000 rows; {aucBL4_v:.4f} on the 29,998 rows scored by the BN) | {auc['BL4']:.4f} | [{ci['BL4'][0]:.4f}, {ci['BL4'][1]:.4f}] | n/a |",
      f"| (b) After acceptance: BN minus this LR (29,998 rows; BN AUC {aucBN_v:.4f}) | {sgn(oB,4)} | [{sgn(loB,4)}, {sgn(hiB,4)}] | {'yes' if inc(loB, hiB) else 'no'} |",
      f"| (c) After acceptance: linear LR on all ten variables minus linear LR on the four blanket variables (30,000 rows; AUC {auc['R4']:.4f} against {auc['BL4']:.4f}) | {sgn(oC,4)} | [{sgn(loC,4)}, {sgn(hiC,4)}] | {'yes' if inc(loC, hiC) else 'no (ten variables higher)'} |",
      f"| context: After acceptance, BN minus linear LR on all ten variables (29,998 rows) | {sgn(oD,4)} | [{sgn(loD,4)}, {sgn(hiD,4)}] | {'yes' if inc(loD, hiD) else 'no (LR higher)'} |", "",
      checks_md(p), ""]
open(PARTS + "/j16_verdict.json", "w").write(json.dumps(dict(res={k: list(v_) for k, v_ in res.items()}, auc=auc, aucBL4_v=aucBL4_v, aucBN_v=aucBN_v, aucBN_A=aucBN_A, ci={k: list(v_) for k, v_ in ci.items()}, nit=NIT, bl_ok=bool(bl_ok), A_incl0=bool(inc(loA, hiA)), B_incl0=bool(inc(loB, hiB)), C_incl0=bool(inc(loC, hiC)), D_incl0=bool(inc(loD, hiD)))))
p.write("\n".join(md))
