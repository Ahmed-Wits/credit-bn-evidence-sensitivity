"""J19: fairer baselines and decision-level advantage. python j19.py STAGE VAR   (STAGE application|after; VAR a blanket variable of the article's masking grid)
Comparators for LR and RF, each scored on the same masked test rows as the BN (article design: blanket variable masked one at a time; MCAR and dependent; levels 20/40/50%; mask seeds 42-51;
shared 2,000 multinomial bootstrap resamples default_rng(42); rows invalid for the BN excluded from every model; mean over the 10 seeds):
  std : the article's models (stored lr/rf objects, trained on complete data) with the refit-IterativeImputer values            [existing]
  A   : LR and RF retrained on the training copy masked in the same way (same mechanism, level, variable; mask seed 42 = the imputer's training mask), imputed values from that imputer,
        PLUS indicator columns: masked variable, Monthly Income missing, Number of Dependents missing (the main LR/RF carry NO indicators)
  B   : LR and RF retrained on complete training data without the masked variable; masked test rows scored by that reduced model, other rows by the full (std) model
Degradation Deg(m) = AUC_std,complete - AUC_c(m) on the valid rows (common baseline = the article's complete-data model of the same family; A is also given with its own unmasked-test baseline).
Output: j19/res_STAGE_VAR.json"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
stage, var = sys.argv[1], sys.argv[2]; contribution = {"application": "selection", "after": "monitoring"}[stage]; t_all = time.time()
NJ = int(os.environ.get("RF_JOBS", "3")); RS = (5, 10)
ctx = rq3.load_context(contribution); y = ctx["y"]; n = len(y); g = ctx["g"]; g["bin_natural_category"] = lambda s, max_category=3: rq3._bin_natural(s, "wrap", max_category); ytr = np.asarray(g["y_train"]).astype(int); dfc_ = ctx["dfc"]
tr_idx, te_idx = ctx["train_idx"], ctx["test_idx"]
LRt, RFt = ctx["models"]["LR"], ctx["models"]["RF"]
J12 = json.load(open(f"{ADD}/j12/{stage}_seed42_gate.json"))
TAU_STD = {m: {r: J12["tau"][m][str(r)] for r in RS} for m in ("LR", "RF")}
W = get_W(n)
LOG = lambda s: print(s, flush=True)

def clone_fit_lr(X, yy):
    m = clone(LRt)
    try: ck.fit_converged(m, X, yy, name="LR"); return m, "saga"
    except AssertionError:
        m = clone(LRt).set_params(solver="lbfgs", max_iter=5000); ck.fit_converged(m, X, yy, name="LR lbfgs"); return m, "lbfgs-fallback"
def fit_rf(X, yy): return clone(RFt).set_params(n_jobs=NJ).fit(X, yy)
def oof_taus(X, yy):
    """5-fold out-of-fold training predictions, grid 0.01, ties to the highest; LR OOF model per the J12 rule (Application: the grid-search LR as is; After: converged lbfgs refit)."""
    lrm = clone(LRt) if stage == "application" else LogisticRegression(solver="lbfgs", max_iter=5000, C=LRt.C, penalty="l2", class_weight="balanced")
    o_lr = oof_pred(lrm, X, yy); o_rf = oof_pred(clone(RFt).set_params(n_jobs=NJ), X, yy)
    return {"LR": tau_pair(o_lr, yy), "RF": tau_pair(o_rf, yy)}

# ---------------------------------------------------------------- indicators
def ind_cols(idx, mask):
    return np.column_stack([np.asarray(mask, float), dfc_.loc[idx, "Monthly Income"].isna().to_numpy().astype(float), dfc_.loc[idx, "Number of Dependents"].isna().to_numpy().astype(float)])
ctx_tr = dict(ctx, test_idx=tr_idx)         # masked_matrix on the training rows

# ---------------------------------------------------------------- B: reduced models on complete training data (no masked variable)
red_raw = [c for c in ctx["raw_cols"] if c != var]
bred = g["build_design"](red_raw); Xr_tr, Xr_te = bred["Xtr"].values, bred["Xte"].values
t0 = time.time(); lrB, solB = clone_fit_lr(Xr_tr, ytr); rfB = fit_rf(Xr_tr, ytr)
PB = {"LR": lrB.predict_proba(Xr_te)[:, 1], "RF": rfB.predict_proba(Xr_te)[:, 1]}; TAU_B = oof_taus(Xr_tr, ytr)
LOG(f"[{stage} {var}] B reduced models fitted ({solB}), AUC reduced LR {roc_auc_score(y, PB['LR']):.4f} RF {roc_auc_score(y, PB['RF']):.4f} on all rows; OOF thresholds {TAU_B}  ({time.time()-t0:.0f}s)")

def wsave(flag, v, r):
    """weighted saving over the B resamples (rows with v = 0 get weight 0) and the observed saving on rows v"""
    vf = np.asarray(v, float); yy = y.astype(float); fn = ((y == 1) & ~flag).astype(np.float32) * vf.astype(np.float32); fp = ((y == 0) & flag).astype(np.float32) * vf.astype(np.float32)
    D = (yy * vf).astype(np.float32); out = np.empty(B)
    for a in range(0, B, 500):
        Wc = W[a:a + 500].astype(np.float32); cost = r * (Wc @ fn) + (Wc @ fp); Dw = Wc @ D; Nw = Wc @ vf.astype(np.float32)
        out[a:a + 500] = 1.0 - cost / np.minimum(r * Dw, Nw - Dw)
    FN = ((y == 1) & ~flag & (vf > 0)).sum(); FP = ((y == 0) & flag & (vf > 0)).sum(); D0 = ((y == 1) & (vf > 0)).sum(); N0 = (vf > 0).sum()
    return 1.0 - (r * FN + FP) / min(r * D0, N0 - D0), out

NAMES = ["BN", "LR_std", "RF_std", "LR_A", "RF_A", "LR_B", "RF_B", "LR_Aown", "RF_Aown"]
acc = MaskAcc(y); OUT = dict(stage=stage, variable=var, masking={}, decision={}, notes={}); base_p = {"LR": ctx["p0"]["LR"], "RF": ctx["p0"]["RF"], "BN": ctx["p0"]["BN"]}
OUT["notes"]["B_reduced"] = dict(solver=solB, tau=TAU_B, auc_all_rows={k: float(roc_auc_score(y, PB[k])) for k in PB}, n_columns=int(Xr_tr.shape[1]))
for mech in MECHS:
    for lev in LEVELS:
        key = (var, lev, mech); t1 = time.time()
        imp, _, _, _ = rq3.get_refit_imputer(ctx, var, lev, mech)
        cc = None if mech == "mcar" else rq3.cond_codes(ctx, var, "train"); tm, _real = MK.make_mask(len(tr_idx), lev, mech, 42, cc)
        Xm_tr = rq3.masked_matrix(ctx_tr, var, tm, "refit", imp=imp).values
        XA_tr = np.hstack([Xm_tr, ind_cols(tr_idx, tm)])
        lrA, solA = clone_fit_lr(XA_tr, ytr); rfA = fit_rf(XA_tr, ytr)
        XA_te0 = np.hstack([rq3.masked_matrix(ctx, var, np.zeros(n, bool), "refit", imp=imp).values, ind_cols(te_idx, np.zeros(n, bool))])      # own unmasked baseline of A
        own0 = {"LR": lrA.predict_proba(XA_te0)[:, 1], "RF": rfA.predict_proba(XA_te0)[:, 1]}
        if lev == 0.5: TAU_A = oof_taus(XA_tr, ytr)
        dec = {}; real = []; excl = []
        for seed in SEEDS:
            r = rq3.run_config(ctx, var, lev, mech, seed, return_preds=True, bp_check=False); P = r["preds"]; v = r["valid"]; mask = r["mask"]; real.append(r["realised"]); excl.append(r["excluded"]["total"])
            Xm = rq3.masked_matrix(ctx, var, mask, "refit", imp=imp).values; XA = np.hstack([Xm, ind_cols(te_idx, mask)])
            pA = {"LR": lrA.predict_proba(XA)[:, 1], "RF": rfA.predict_proba(XA)[:, 1]}
            pB = {k: np.where(mask, PB[k], P[k + "_primary"]) for k in ("LR", "RF")}
            S = {"BN": P["BN"], "LR_std": P["LR_primary"], "RF_std": P["RF_primary"], "LR_A": pA["LR"], "RF_A": pA["RF"], "LR_B": pB["LR"], "RF_B": pB["RF"], "LR_Aown": pA["LR"], "RF_Aown": pA["RF"]}
            BS = {"BN": base_p["BN"], "LR_std": base_p["LR"], "RF_std": base_p["RF"], "LR_A": base_p["LR"], "RF_A": base_p["RF"], "LR_B": base_p["LR"], "RF_B": base_p["RF"], "LR_Aown": own0["LR"], "RF_Aown": own0["RF"]}
            acc.add(key, seed, v, S, BS)
            if lev == 0.5:                                  # decision level: flags at the BN threshold 1/(1+r), LR/RF at OOF thresholds of the corresponding training data
                for rr in RS:
                    flags = {"BN": P["BN"] >= 1 / (1 + rr), "LR_std": P["LR_primary"] >= TAU_STD["LR"][rr], "RF_std": P["RF_primary"] >= TAU_STD["RF"][rr],
                             "LR_A": pA["LR"] >= TAU_A["LR"][rr], "RF_A": pA["RF"] >= TAU_A["RF"][rr],
                             "LR_B": np.where(mask, PB["LR"] >= TAU_B["LR"][rr], P["LR_primary"] >= TAU_STD["LR"][rr]), "RF_B": np.where(mask, PB["RF"] >= TAU_B["RF"][rr], P["RF_primary"] >= TAU_STD["RF"][rr])}
                    flags["BN"] = np.where(np.isfinite(P["BN"]), flags["BN"], False)
                    for nm, fl in flags.items():
                        pt, bt = wsave(fl, v, rr); d = dec.setdefault((nm, rr), [0.0, np.zeros(B), 0])
                        d[0] += pt; d[1] += bt; d[2] += 1
        res, deg = acc.finish(key, NAMES)
        mrow = dict(variable=var, level=lev, mechanism=mech, res=res, deg=deg, realised_mean=float(np.mean(real)), excluded_max=int(max(excl)), solver_A=solA,
                    auc=[(m, [(s, a, a0) for s, a, a0 in acc.auc[(key, m)]]) for m in NAMES])
        OUT["masking"]["|".join(map(str, key))] = mrow
        if lev == 0.5:
            dd = {}
            for rr in RS:
                m_bn = dec[("BN", rr)]; bn_pt, bn_bt = m_bn[0] / m_bn[2], m_bn[1] / m_bn[2]
                for nm in ("BN", "LR_std", "RF_std", "LR_A", "RF_A", "LR_B", "RF_B"):
                    pt, bt = dec[(nm, rr)][0] / dec[(nm, rr)][2], dec[(nm, rr)][1] / dec[(nm, rr)][2]; lo, hi = pct_ci(100 * bt)
                    e = dict(saving=100 * pt, lo=lo, hi=hi)
                    if nm != "BN":
                        dl, dh = pct_ci(100 * (bn_bt - bt)); e.update(diff_bn_minus=100 * (bn_pt - pt), diff_lo=dl, diff_hi=dh)
                    dd[f"{nm}_r{rr}"] = e
            OUT["decision"]["|".join(map(str, key))] = dict(variable=var, mechanism=mech, level=lev, saving=dd, tau_A=TAU_A, tau_B=TAU_B, tau_std=TAU_STD)
        LOG(f"[{stage} {var}] {mech} {lev:.1f}  D(BN-LR std) {res['LR_std']['point']:+.5f} [{res['LR_std']['lo']:+.5f},{res['LR_std']['hi']:+.5f}]  A {res['LR_A']['point']:+.5f}  B {res['LR_B']['point']:+.5f}  solver A {solA}  ({(time.time()-t1)/60:.1f} min; total {(time.time()-t_all)/60:.1f})")
        json.dump(OUT, open(f"{ADD}/j19/res_{stage}_{var.replace(' ', '_').replace('+','plus')}.json", "w"), indent=1, default=float)
LOG(f"saved {stage} {var} {(time.time()-t_all)/60:.1f} min")
