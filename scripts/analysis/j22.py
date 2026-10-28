"""J22: J17 variants V1 and V2 across split seeds. python j22.py VARIANT SEED  (VARIANT after_V1|after_V2; SEED 1..5, or 42 as a gate against the J17 results)
J12 split-seed orchestration (split, train-only edges, relearned structure, 100 fixed EM iterations, LR/RF, OOF thresholds, paired AUC and saving bootstraps) with the J17 variable lists.
LR: saga with the convergence check, lbfgs with the same C when saga does not converge (as J17). AUC, blanket and saving only. Output j22/NAME_seedN.json"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
name, seed = sys.argv[1], int(sys.argv[2]); t_all = time.time()
DROP = {"after_V1": ["90+ Days Late"], "after_V2": ["30-59 Days Past Due", "60-89 Days Past Due", "90+ Days Late"]}[name]
cfg = variant_cfg("after", DROP); stage = "after"; NJ = int(os.environ.get("RF_JOBS", "1"))
tr, te = get_split(seed); ytr = dfc.loc[tr, "Default"].values.astype(int); yte = dfc.loc[te, "Default"].values.astype(int); edges = get_edges(tr)
out = dict(name=name, seed=seed, drop=DROP, n_train=len(tr), n_test=len(te), default_rate_train=float(ytr.mean()), default_rate_test=float(yte.mean()))
ftr, cats = bn_frame(cfg, tr, edges); fte, _ = bn_frame(cfg, te, edges)
nodes = cfg["nodes"]; all_nodes = nodes + ["M_Income", "M_Dependents"]; sn = {c: ([0, 1] if (c == "Default" or c.startswith("M_")) else cats[c]) for c in all_nodes}
cc = ftr[(ftr.M_Income == 1) & (ftr.M_Dependents == 1)]; out["n_complete"] = len(cc)
res = stx.learn_structure(cc[nodes].copy(), nodes, seeds=range(42, 57), edge_density=0.15, verbose=False)
el = [tuple(e) for e in res["best_edges"]]; out["structure"] = dict(best_seed=int(res["best_seed"]), best_bic=float(res["best_score"]), edges=[list(e) for e in el], n_edges=len(el))
par, ch, cop, blanket = blanket_of(el + IND_EDGES); out["blanket"] = dict(blanket=blanket)
d = ftr[all_nodes]; model, hist, why = em.run_em(el + IND_EDGES, d, sn, lambda m: ll.compute_log_likelihood(m, d), aggregate=True, nonmono_tol=1e18, verbose=False, **cfg["em"])
out["em"] = dict(iterations=len(hist), stop=why)
post = sens.make_posterior_fn(model)
def ev(i):
    r_ = fte.loc[i]; e = {c: r_[c] for c in nodes if c != "Default" and pd.notnull(r_[c])}; e["M_Income"] = int(r_["M_Income"]); e["M_Dependents"] = int(r_["M_Dependents"]); return e
pbn = np.array([post(ev(i)) for i in fte.index]); valid = np.isfinite(pbn); out["n_bn_defined"] = int(valid.sum())
Xtr, Xte = SR.build_design(cfg["raw"], tr, te, edges, cfg["edge_vars"], cfg["nat_vars"]); Xtr, Xte = Xtr.values, Xte.values
lr, sol = lr_fit(cfg["lr"], Xtr, ytr); rf = RandomForestClassifier(n_jobs=NJ, **cfg["rf"]).fit(Xtr, ytr); out["lr_solver"] = sol
plr, prf = lr.predict_proba(Xte)[:, 1], rf.predict_proba(Xte)[:, 1]
oof_lr = oof_pred(oof_lr_model(stage, lr), Xtr, ytr); oof_rf = oof_pred(clone(rf), Xtr, ytr)
RS = (5, 10); Tau = {"BN": {r: 1 / (1 + r) for r in RS}, "LR": tau_pair(oof_lr, ytr), "RF": tau_pair(oof_rf, ytr)}; out["tau"] = {m: {str(r): float(Tau[m][r]) for r in RS} for m in Tau}
yv = yte[valid]; pv = {"BN": pbn[valid], "LR": plr[valid], "RF": prf[valid]}
out["auc"] = {"BN": float(roc_auc_score(yv, pv["BN"])), "LR": float(roc_auc_score(yte, plr)), "RF": float(roc_auc_score(yte, prf))}
for nm in ("LR", "RF"):
    obs, dif = paired_boot_full(yv, pv["BN"], pv[nm]); lo, hi = np.percentile(dif, [2.5, 97.5]); out[f"auc_BN_minus_{nm}"] = dict(observed=float(obs), lo=float(lo), hi=float(hi))
pt = savings_point(yv, pv, Tau, RS); W = boot_weights(len(yv)); bt = saving_boot(yv, pv, Tau, list(RS), W=W); out["saving"] = {}; out["saving_diff"] = {}
for j, r in enumerate(RS):
    for m in ("BN", "LR", "RF"):
        lo, hi = pct_ci(100 * bt[m][:, j]); out["saving"][f"{m}_r{r}"] = dict(point=100 * pt[(m, r)][0], lo=lo, hi=hi)
    for m in ("LR", "RF"):
        lo, hi = pct_ci(100 * (bt["BN"][:, j] - bt[m][:, j])); out["saving_diff"][f"BN_minus_{m}_r{r}"] = dict(point=100 * (pt[("BN", r)][0] - pt[(m, r)][0]), lo=lo, hi=hi)
out["seconds"] = time.time() - t_all
os.makedirs(ADD + "/j22", exist_ok=True); json.dump(out, open(f"{ADD}/j22/{name}_seed{seed}.json", "w"), indent=1, default=float)
print("done", name, seed, f"{out['seconds']/60:.1f} min", flush=True)
