"""J24 Stage B (i), Application only (the only stage at which a structure passed the gate): the J19 masking comparison for the structures NB and TAN (and the BIC BN again, as a gate against J19) against the
existing, A (matched-training with indicators) and B (subset-retrained) LR/RF comparators. python j24_mask.py VAR   (Age | Debt Ratio). Same design, seeds, levels, shared resamples and common-baseline degradation as j19.py.
NB and TAN have no missingness-indicator nodes. Output j24/mask_application_VAR.json"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from j_tables import ev_dict
from pgmpy.inference import VariableElimination
var = sys.argv[1]; stage = "application"; contribution = "selection"; t_all = time.time(); NJ = int(os.environ.get("RF_JOBS", "3"))
ctx = rq3.load_context(contribution); y = ctx["y"]; n = len(y); g = ctx["g"]; g["bin_natural_category"] = lambda s, max_category=3: rq3._bin_natural(s, "wrap", max_category); ytr = np.asarray(g["y_train"]).astype(int); dfc_ = ctx["dfc"]
tr_idx, te_idx = ctx["train_idx"], ctx["test_idx"]; LRt, RFt = ctx["models"]["LR"], ctx["models"]["RF"]; LOG = lambda s: print(s, flush=True)
# ---------------------------------------------------------------- the two structures' BNs
fit, _i, _s = load_fit("application", ADD + "/j17"); STR = {}
for sname in ("NB", "TAN"):
    d = pickle.load(open(f"{ADD}/j24/application_{sname}.pkl", "rb")); bn_ = DiscreteBayesianNetwork([tuple(e) for e in d["model_edges"]]); bn_.add_cpds(*d["cpds"]); assert bn_.check_model()
    Fs = rq3._states_frame(fit["fte"], d["ev_cols"], d["S"]); cod = np.column_stack([pd.Index(d["S"][c]).get_indexer(Fs[c]) for c in d["ev_cols"]])
    STR[sname] = dict(bn=bn_, S=d["S"], ev_cols=d["ev_cols"], codes=cod, eng=VariableElimination(bn_), p_full=d["pbn"])
    LOG(f"{sname}: full-evidence AUC {roc_auc_score(y, d['pbn']):.4f}")
def struct_post(sname, mask):
    st = STR[sname]; c2 = st["codes"].copy(); c2[mask, st["ev_cols"].index(var + " (bin)")] = -1
    u, inv = np.unique(c2, axis=0, return_inverse=True); inv = np.asarray(inv).ravel(); pu = np.full(len(u), np.nan)
    for k, row in enumerate(u):
        try:
            with np.errstate(invalid="ignore", divide="ignore"): p_ = float(st["eng"].query(variables=["Default"], evidence=ev_dict(st["S"], st["ev_cols"], row), show_progress=False).get_value(Default=1))
            pu[k] = p_ if (np.isfinite(p_) and 0 <= p_ <= 1) else np.nan
        except Exception: pass
    return pu[inv]
# ---------------------------------------------------------------- comparators as in j19.py
def ind_cols(idx, mask): return np.column_stack([np.asarray(mask, float), dfc_.loc[idx, "Monthly Income"].isna().to_numpy().astype(float), dfc_.loc[idx, "Number of Dependents"].isna().to_numpy().astype(float)])
def clone_fit_lr(X, yy):
    m = clone(LRt)
    try: ck.fit_converged(m, X, yy, name="LR"); return m, "saga"
    except AssertionError:
        m = clone(LRt).set_params(solver="lbfgs", max_iter=5000); ck.fit_converged(m, X, yy, name="LR lbfgs"); return m, "lbfgs-fallback"
def fit_rf(X, yy): return clone(RFt).set_params(n_jobs=NJ).fit(X, yy)
ctx_tr = dict(ctx, test_idx=tr_idx)
red_raw = [c for c in ctx["raw_cols"] if c != var]; bred = g["build_design"](red_raw); Xr_tr, Xr_te = bred["Xtr"].values, bred["Xte"].values
lrB, solB = clone_fit_lr(Xr_tr, ytr); rfB = fit_rf(Xr_tr, ytr); PB = {"LR": lrB.predict_proba(Xr_te)[:, 1], "RF": rfB.predict_proba(Xr_te)[:, 1]}
NAMES = ["BN", "NB", "TAN", "LR_std", "RF_std", "LR_A", "RF_A", "LR_B", "RF_B"]
acc = MaskAcc(y); OUT = dict(stage=stage, variable=var, masking={}, notes=dict(B_solver=solB)); base_p = {"LR": ctx["p0"]["LR"], "RF": ctx["p0"]["RF"], "BN": ctx["p0"]["BN"], "NB": STR["NB"]["p_full"], "TAN": STR["TAN"]["p_full"]}
for mech in MECHS:
    for lev in LEVELS:
        key = (var, lev, mech); t1 = time.time()
        imp, _, _, _ = rq3.get_refit_imputer(ctx, var, lev, mech)
        cc = None if mech == "mcar" else rq3.cond_codes(ctx, var, "train"); tm, _r = MK.make_mask(len(tr_idx), lev, mech, 42, cc)
        Xm_tr = rq3.masked_matrix(ctx_tr, var, tm, "refit", imp=imp).values; XA_tr = np.hstack([Xm_tr, ind_cols(tr_idx, tm)])
        lrA, solA = clone_fit_lr(XA_tr, ytr); rfA = fit_rf(XA_tr, ytr); real = []; excl = []
        for seed in SEEDS:
            r = rq3.run_config(ctx, var, lev, mech, seed, return_preds=True, bp_check=False); P = r["preds"]; v = r["valid"]; mask = r["mask"]; real.append(r["realised"]); excl.append(r["excluded"]["total"])
            Xm = rq3.masked_matrix(ctx, var, mask, "refit", imp=imp).values; XA = np.hstack([Xm, ind_cols(te_idx, mask)])
            pA = {"LR": lrA.predict_proba(XA)[:, 1], "RF": rfA.predict_proba(XA)[:, 1]}; pB = {k: np.where(mask, PB[k], P[k + "_primary"]) for k in ("LR", "RF")}
            pN = struct_post("NB", mask); pT = struct_post("TAN", mask); assert np.isfinite(pN[v]).all() and np.isfinite(pT[v]).all()
            S_ = {"BN": P["BN"], "NB": pN, "TAN": pT, "LR_std": P["LR_primary"], "RF_std": P["RF_primary"], "LR_A": pA["LR"], "RF_A": pA["RF"], "LR_B": pB["LR"], "RF_B": pB["RF"]}
            BS = {"BN": base_p["BN"], "NB": base_p["NB"], "TAN": base_p["TAN"], "LR_std": base_p["LR"], "RF_std": base_p["RF"], "LR_A": base_p["LR"], "RF_A": base_p["RF"], "LR_B": base_p["LR"], "RF_B": base_p["RF"]}
            acc.add(key, seed, v, S_, BS)
        out_ = {}
        for ref in ("BN", "NB", "TAN"): res, deg = acc.finish(key, NAMES, ref=ref); out_[ref] = res
        OUT["masking"]["|".join(map(str, key))] = dict(variable=var, level=lev, mechanism=mech, res=out_, deg=deg, realised_mean=float(np.mean(real)), excluded_max=int(max(excl)), solver_A=solA, auc=[(m, [(s, a, a0) for s, a, a0 in acc.auc[(key, m)]]) for m in NAMES])
        LOG(f"[{var}] {mech} {lev:.1f}  D(BIC-LR std) {out_['BN']['LR_std']['point']:+.5f}  D(NB-LR std) {out_['NB']['LR_std']['point']:+.5f}  D(TAN-LR std) {out_['TAN']['LR_std']['point']:+.5f} ({(time.time()-t1)/60:.1f} min; total {(time.time()-t_all)/60:.1f})")
        json.dump(OUT, open(f"{ADD}/j24/mask_application_{var.replace(' ', '_')}.json", "w"), indent=1, default=float)
LOG(f"saved {var} {(time.time()-t_all)/60:.1f} min")
