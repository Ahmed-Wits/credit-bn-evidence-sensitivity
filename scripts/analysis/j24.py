"""J24 Stage A: structures BIC (the main model, reproduced), NB (naive Bayes) and TAN, same bins, same training rows, same custom EM (100 iterations After acceptance, stored rule at Application).
python j24.py STAGE STRUCTURE   (STAGE application|after, STRUCTURE BIC|NB|TAN). NB and TAN have NO missingness-indicator nodes: unrecorded Monthly Income / Dependents are simply unobserved.
TAN call: pgmpy.estimators.TreeSearch(data_complete_rows, root_node=<first non-class node>).estimate(estimator_type='tan', class_node='Default', edge_weights_fn='mutual_info'): conditional mutual information given Default.
Output: j24/STAGE_STRUCTURE.json (+ pickles of scores and, for masking, model edges/cpds)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from j_tables import *
from pgmpy.estimators import TreeSearch
stage, struct = sys.argv[1], sys.argv[2]; name = {"application": "application", "after": "after_full"}[stage]; t_all = time.time(); LOG = lambda s: print(s, flush=True); sh = lambda s: s.replace(" (bin)", "")
RS = (5, 10, 15, 20)
fit, info, _ = load_fit(name, ADD + "/j17"); g = get_g(); cfg = variant_cfg(stage, [])
tr, te = get_split(42); edges = get_edges(tr); ftr, cats = bn_frame(cfg, tr, edges); fte, _c = bn_frame(cfg, te, edges); nodes = cfg["nodes"]; y_all = g["y_test"].astype(int)
out = dict(stage=stage, structure=struct)
if struct == "BIC":
    ctx = variant_ctx(fit, None, None, name, base="skip"); bn, S, ev_cols, codes = ctx["bn"], ctx["S"], ctx["ev_cols"], ctx["codes"]; pbn = fit["pbn"].copy()
    out["edges"] = [list(e) for e in fit["model_edges"]]; out["em"] = dict(note="main model (j17 fit, reproduced and gated in J17)")
else:
    sn = {c: ([0, 1] if c == "Default" else cats[c]) for c in nodes}; data = ftr[nodes]
    if struct == "NB": el = [("Default", v) for v in nodes if v != "Default"]; out["tan_call"] = None
    else:
        cc = ftr[(ftr.M_Income == 1) & (ftr.M_Dependents == 1)][nodes]; root = [v for v in nodes if v != "Default"][0]
        D = TreeSearch(cc, root_node=root, n_jobs=1).estimate(estimator_type="tan", class_node="Default", edge_weights_fn="mutual_info", show_progress=False)
        el = [tuple(e) for e in D.edges()]; tree = [e for e in el if e[0] != "Default"]
        assert len(tree) == len(nodes) - 2 and all(("Default", v) in el for v in nodes if v != "Default")
        out["tan_call"] = f"TreeSearch(complete training rows, root_node='{root}', n_jobs=1).estimate(estimator_type='tan', class_node='Default', edge_weights_fn='mutual_info')"; out["tan_tree_edges"] = [[sh(a), sh(b)] for a, b in tree]; LOG(f"TAN tree ({root} as root): {out['tan_tree_edges']}")
    out["edges"] = [list(e) for e in el]
    t0 = time.time(); model, hist, why = em.run_em(el, data, sn, lambda m: ll.compute_log_likelihood(m, data), aggregate=True, nonmono_tol=1e18 if stage == "after" else 1e-6, verbose=False, **cfg["em"])
    out["em"] = dict(iterations=len(hist), stop=why, train_ll=float(hist[-1]["train_ll"]), seconds=time.time() - t0); LOG(f"EM {len(hist)} iterations ({why}) {time.time()-t0:.0f}s")
    bn = DiscreteBayesianNetwork([tuple(e) for e in model.edges()]); bn.add_cpds(*model.get_cpds()); assert bn.check_model()
    S = {n_: list(bn.get_cpds(n_).state_names[n_]) for n_ in bn.nodes()}; ev_cols = [c for c in nodes if c != "Default"]
    Fs = rq3._states_frame(fte, ev_cols, S); codes = np.column_stack([pd.Index(S[c]).get_indexer(Fs[c]) for c in ev_cols])
    for j, c in enumerate(ev_cols): assert ((codes[:, j] < 0) == Fs[c].isna().to_numpy()).all()
    eng0 = VariableElimination(bn); u, inv = np.unique(codes, axis=0, return_inverse=True); inv = np.asarray(inv).ravel(); pu = np.full(len(u), np.nan)
    for k, row in enumerate(u):
        try:
            with np.errstate(invalid="ignore", divide="ignore"): pu[k] = float(eng0.query(variables=["Default"], evidence=ev_dict(S, ev_cols, row), show_progress=False).get_value(Default=1))
        except Exception: pass
    pbn = pu[inv]
eng = VariableElimination(bn)
# ---------------------------------------------------------------- common rows and metrics
rows_all = fit["test_rows"]
if stage == "after": common = fit["valid"].copy()
else:
    per = pickle.load(open(R + "/selection/predictions/sel_rq2_per_row.pkl", "rb"))["per_row"]
    ga = per[per.variable == "Age (bin)"].set_index("row_index"); gd = per[per.variable == "Debt Ratio (bin)"].set_index("row_index")
    keep = ga.index[ga.retracted_prob.notna().values & gd.retracted_prob.reindex(ga.index).notna().values]; common = np.isin(rows_all, keep)
assert common.sum() == 29998; cm = common & np.isfinite(pbn); out["n_common"] = int(common.sum()); out["n_defined"] = int(cm.sum()); assert cm.sum() == common.sum(), "posterior undefined on common rows"
y = y_all[common]; P = {"S": pbn[common], "LR": fit["plr"][common], "RF": fit["prf"][common], "BIC": fit["pbn"][common]}
out["auc"] = {k: float(roc_auc_score(y, v)) for k, v in P.items()}; out["gap_to_LR"] = out["auc"]["LR"] - out["auc"]["S"]; out["gap_to_RF"] = out["auc"]["RF"] - out["auc"]["S"]
for nm in ("LR", "RF", "BIC"):
    obs, dif = paired_boot_full(y, P["S"], P[nm]); lo, hi = np.percentile(dif, [2.5, 97.5]); out[f"auc_S_minus_{nm}"] = dict(observed=float(obs), lo=float(lo), hi=float(hi))
RS2 = (5, 10); Tau = {"BN": {r: 1 / (1 + r) for r in RS2}, "LR": {r: fit["tau"]["LR"][r] for r in RS2}, "RF": {r: fit["tau"]["RF"][r] for r in RS2}}
Pm = {"BN": P["S"], "LR": P["LR"], "RF": P["RF"]}; pt = savings_point(y, Pm, Tau, RS2); Wt = boot_weights(len(y)); bt = saving_boot(y, Pm, Tau, list(RS2), W=Wt); out["saving"] = {}; out["saving_diff"] = {}
for j, r in enumerate(RS2):
    for m in ("BN", "LR", "RF"): lo, hi = pct_ci(100 * bt[m][:, j]); out["saving"][f"{m}_r{r}"] = dict(point=100 * pt[(m, r)][0], lo=lo, hi=hi, flagged=int(pt[(m, r)][1]))
    for m in ("LR", "RF"): lo, hi = pct_ci(100 * (bt["BN"][:, j] - bt[m][:, j])); out["saving_diff"][f"S_minus_{m}_r{r}"] = dict(point=100 * (pt[("BN", r)][0] - pt[(m, r)][0]), lo=lo, hi=hi)
out["n_distinct_scores"] = int(len(np.unique(np.round(P["S"], 12)))); out["mean_pred"] = float(P["S"].mean()); out["obs_rate"] = float(y.mean()); out["blanket"] = [sh(c) for c in fit["blanket"] if not c.startswith("M_")] if struct == "BIC" else [sh(c) for c in ev_cols]
# ---------------------------------------------------------------- S_exp and flip rates for every variable with non-zero S_exp (blanket for BIC; all variables for NB, TAN)
vars_ = sorted(c for c in fit["blanket"] if not c.startswith("M_")) if struct == "BIC" else ev_cols; SE = {}; FL = {r: {} for r in RS}; t0 = time.time(); okall = common.copy()
for v in vars_:
    j = ev_cols.index(v); inv_, P0, W, PX = var_tables(eng, S, ev_cols, codes, j); obs = codes[:, j]; pxo = PX[inv_, obs]
    miss = obs < 0; okv = common & np.isfinite(P0[inv_]) & np.isfinite(pxo) & ~miss
    SE[v] = (okv, np.where(W[inv_] > 0, W[inv_] * np.abs(PX[inv_] - pxo[:, None]), 0.0), pxo, P0[inv_]); okall &= okv | (miss & common)
    LOG(f"  {sh(v)}: {len(P0)} patterns, rows with the variable recorded {int((common & ~miss).sum())} ({time.time()-t0:.0f}s)")
# rows R_j: rows where the variable is recorded and everything defined (equal to the 29,998 rows for the always-recorded variables)
res_sens = {}
mats = []; labels = []
for v in vars_:
    okv, sx, pxo, p0 = SE[v]; j = ev_cols.index(v); obs = codes[:, j]; sx2 = sx.copy(); sx2[np.arange(len(obs)), np.maximum(obs, 0)] = 0.0; sexp = sx2.sum(axis=1)
    use = okv; res_sens[v] = dict(n=int(use.sum()), sexp=float(sexp[use].mean()))
    cols_v = [sexp[use]] + [((pxo[use] >= 1 / (1 + r)) != (p0[use] >= 1 / (1 + r))).astype(float) for r in RS]
    res_sens[v]["_cols"] = np.column_stack(cols_v)
for v in vars_:
    Mx = res_sens[v].pop("_cols"); bm = sens.boot_means([Mx], B=2000, seed=42)[0]; lo, hi = np.percentile(bm, [2.5, 97.5], axis=0); mm = Mx.mean(axis=0)
    res_sens[v]["sexp_ci"] = [float(lo[0]), float(hi[0])]; res_sens[v]["flip"] = {str(r): dict(rate=float(mm[1 + i]), lo=float(lo[1 + i]), hi=float(hi[1 + i])) for i, r in enumerate(RS)}
out["sens"] = {sh(v): d_ for v, d_ in res_sens.items()}; out["seconds"] = time.time() - t_all
os.makedirs(ADD + "/j24", exist_ok=True); json.dump(out, open(f"{ADD}/j24/{stage}_{struct}.json", "w"), indent=1, default=float)
pickle.dump(dict(pbn=pbn, edges=out["edges"], model_edges=[tuple(e) for e in bn.edges()], cpds=bn.get_cpds(), S=S, ev_cols=ev_cols, struct=struct, stage=stage), open(f"{ADD}/j24/{stage}_{struct}.pkl", "wb"))
LOG(f"done {stage} {struct}: AUC {out['auc']['S']:.4f} (LR {out['auc']['LR']:.4f}, RF {out['auc']['RF']:.4f}) gap_to_LR {out['gap_to_LR']:.4f}; distinct scores {out['n_distinct_scores']}; {out['seconds']/60:.1f} min")
