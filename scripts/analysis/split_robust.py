"""J12: split-seed robustness. One function run_seed(stage, seed) redoes: 80/20 stratified split, train-only bin edges, BN structure search (15 restarts, complete training rows),
BN parameters (application: stored EM stopping rule; after acceptance: 100 fixed EM iterations, the iteration-100 model), LR and RF (Table S-Settings, no retuning),
thresholds (BN 1/(1+r); LR, RF 5-fold out-of-fold training thresholds, ties to the highest), and the test evaluation. Nothing uses the seed-42 structure or test rows for fitting."""
import os, sys, json, pickle, time, warnings, importlib.util
os.environ.setdefault("PYTHONHASHSEED", "0"); assert os.environ["PYTHONHASHSEED"] == "0"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
R = os.environ["CREDIT_BN_ROOT"]; PROC = R + "/Data/processed/"; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"
sys.path.insert(0, R + "/shared/src"); sys.path.insert(0, ADD + "/code")
import structure as stx, em, loglik as ll, decision_value as dv
from costlib import savings_point, saving_boot
from addcommon import boot_weights, pct_ci
spec = importlib.util.spec_from_file_location("sens", R + "/shared/src/sensitivity.py"); sens = importlib.util.module_from_spec(spec); spec.loader.exec_module(sens)
from pgmpy.models import DiscreteBayesianNetwork

dfc = pickle.load(open(PROC + "df_clean.pkl", "rb"))
NO = json.load(open(PROC + "node_orders.json")); SNO = json.load(open(PROC + "sel_node_order.json"))
EQ5 = ["Credit Utilisation Rate", "Age", "Debt Ratio", "Monthly Income", "Open Credit Lines"]
IMPUTE_ORDER = ['Credit Utilisation Rate', 'Age', 'Debt Ratio', 'Monthly Income', 'Open Credit Lines', '30-59 Days Past Due', '90+ Days Late', '60-89 Days Past Due', 'Number of Dependents', 'Real Estate Loans']
DESIGN_ORDER = ['Credit Utilisation Rate', 'Age', 'Debt Ratio', 'Open Credit Lines', '30-59 Days Past Due', '90+ Days Late', '60-89 Days Past Due', 'Real Estate Loans', 'Monthly Income', 'Number of Dependents']
SPEC_A = ['Age', 'Debt Ratio', 'Monthly Income', 'Open Credit Lines', 'Real Estate Loans', 'Number of Dependents']
IND_EDGES = [("M_Income", "Monthly Income (bin)"), ("M_Dependents", "Number of Dependents (bin)")]
CFG = {"application": dict(nodes=SNO["derived_7_nodes_selection"], edge_vars=["Age", "Debt Ratio", "Monthly Income", "Open Credit Lines"], nat_vars=["Real Estate Loans", "Number of Dependents"],
                           raw=SPEC_A, em=dict(max_iter=200, ll_tol=0.01, patience=3),
                           lr=dict(C=1, penalty="l2", solver="saga", class_weight="balanced", max_iter=1000, tol=1e-4, random_state=42),
                           rf=dict(n_estimators=500, max_depth=10, min_samples_leaf=20, max_features="sqrt", class_weight="balanced", random_state=42)),
       "after": dict(nodes=NO["original_11_nodes"], edge_vars=EQ5, nat_vars=["30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due", "Number of Dependents", "Real Estate Loans"],
                     raw=IMPUTE_ORDER, em=dict(max_iter=100, ll_tol=-1e18, patience=10 ** 9),
                     lr=dict(C=100, penalty="l2", solver="saga", class_weight="balanced", max_iter=1000, tol=1e-4, random_state=42),
                     rf=dict(n_estimators=500, max_depth=15, min_samples_leaf=10, max_features="sqrt", class_weight="balanced", random_state=42))}

def cut_edges(s, e):
    ea = np.array(e, float).copy(); ea[0] = -np.inf; ea[-1] = np.inf
    return pd.cut(s, bins=ea, include_lowest=True)
def bin_nat(series, max_category=3):
    labels = [str(i) for i in range(max_category)] + [f"{max_category}+"]
    def mv(x):
        if pd.isnull(x): return np.nan
        if x >= max_category: return labels[-1]
        i = int(x)
        return labels[i] if i >= -len(labels) else labels[i % len(labels)]   # rq3 v2 'wrap' rule: identical to the locked function where that is defined (int(x) >= -4)
    return pd.Categorical(series.map(mv), categories=labels, ordered=True)
def get_split(seed):
    tr, te = train_test_split(dfc.index, test_size=0.2, stratify=dfc["Default"], random_state=seed); return np.asarray(tr), np.asarray(te)
def get_edges(tr):
    return {c: pd.qcut(dfc.loc[tr, c], q=5, duplicates="drop", retbins=True)[1] for c in EQ5}
def tostr(s):
    """label strings (NaN kept) and the ordered category list"""
    if str(s.dtype) == "category":
        cats = [str(c) for c in s.cat.categories]; return s.astype(object).map(lambda x: str(x) if pd.notnull(x) else np.nan), cats
    return s, None
def bn_frame(cfg, idx, edges):
    raw = dfc.loc[idx]; out = pd.DataFrame(index=raw.index); out["Default"] = raw["Default"].values.astype(int); cats = {}
    for c in cfg["edge_vars"]: out[c + " (bin)"] = cut_edges(raw[c], edges[c])
    for c in cfg["nat_vars"]: out[c + " (bin)"] = bin_nat(raw[c], 3)
    out["M_Income"] = raw["Monthly Income"].notna().astype(int).values; out["M_Dependents"] = raw["Number of Dependents"].notna().astype(int).values
    res = pd.DataFrame(index=out.index)
    for c in out.columns:
        s, ct = tostr(out[c]); res[c] = s
        if ct is not None: cats[c] = ct
    return res, cats
def build_design(raw_cols, tr, te, edges, edge_vars, nat_vars):
    imp_cols = [c for c in IMPUTE_ORDER if c in raw_cols]; des_cols = [c for c in DESIGN_ORDER if c in raw_cols]
    a, b = dfc.loc[tr, imp_cols].copy(), dfc.loc[te, imp_cols].copy()
    if "Monthly Income" in imp_cols: a["Monthly Income"], b["Monthly Income"] = np.log1p(a["Monthly Income"]), np.log1p(b["Monthly Income"])
    imp = IterativeImputer(random_state=42).fit(a)
    a = pd.DataFrame(imp.transform(a), columns=imp_cols, index=a.index); b = pd.DataFrame(imp.transform(b), columns=imp_cols, index=b.index)
    if "Monthly Income" in imp_cols: a["Monthly Income"], b["Monthly Income"] = np.expm1(a["Monthly Income"]).clip(lower=0), np.expm1(b["Monthly Income"]).clip(lower=0)
    global LAST_DEP
    LAST_DEP = {k: int((d_["Number of Dependents"][dfc.loc[d_.index, "Number of Dependents"].isna()] <= thr).sum()) for k, thr, d_ in (("train_le_m1", -1, a), ("train_le_m5", -5, a), ("test_le_m1", -1, b), ("test_le_m5", -5, b))} if "Number of Dependents" in imp_cols else {}
    def binned(df):
        o = pd.DataFrame(index=df.index)
        for c in des_cols: o[c + " (bin)"] = cut_edges(df[c], edges[c]) if c in EQ5 else bin_nat(df[c], 3)
        return o.astype(str)
    s_tr, s_te = binned(a), binned(b)
    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore").fit(s_tr)
    names = enc.get_feature_names_out(s_tr.columns)
    return pd.DataFrame(enc.transform(s_tr), columns=names, index=a.index), pd.DataFrame(enc.transform(s_te), columns=names, index=b.index)
def paired_boot_full(y, pa, pb, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed); n = len(y); diffs = []
    for _ in range(n_boot):
        ix = rng.randint(0, n, n); yb = y[ix]
        if yb.min() == yb.max(): continue
        diffs.append(roc_auc_score(yb, pa[ix]) - roc_auc_score(yb, pb[ix]))
    return roc_auc_score(y, pa) - roc_auc_score(y, pb), np.array(diffs)

def blanket_of(edges, t="Default"):
    par = {a for a, b in edges if b == t}; ch = {b for a, b in edges if a == t}; cop = {a for a, b in edges if b in ch and a != t}
    return sorted(par), sorted(ch), sorted(cop), sorted(par | ch | cop)

def run_seed(stage, seed, log=print, n_jobs=2):
    cfg = CFG[stage]; t_all = time.time(); T = {}; out = dict(stage=stage, seed=seed)
    tr, te = get_split(seed); ytr = dfc.loc[tr, "Default"].values.astype(int); yte = dfc.loc[te, "Default"].values.astype(int)
    edges = get_edges(tr); out["bin_edges"] = {c: [float(x) for x in edges[c]] for c in EQ5}; out["n_train"], out["n_test"] = len(tr), len(te); out["default_rate_train"] = float(ytr.mean()); out["default_rate_test"] = float(yte.mean())
    ftr, cats = bn_frame(cfg, tr, edges); fte, _ = bn_frame(cfg, te, edges)
    nodes = cfg["nodes"]; all_nodes = nodes + ["M_Income", "M_Dependents"]
    sn = {}
    for c in all_nodes:
        if c == "Default": sn[c] = [0, 1]
        elif c.startswith("M_"): sn[c] = [0, 1]
        else: sn[c] = cats[c]
    for c in all_nodes:
        for fr in (ftr, fte): assert set(fr[c].dropna().unique()) <= set(sn[c]), c
    cc = ftr[(ftr.M_Income == 1) & (ftr.M_Dependents == 1)]; out["n_complete"] = len(cc)
    # ---- structure
    t0 = time.time(); data = cc[nodes].copy()
    res = stx.learn_structure(data, nodes, seeds=range(42, 57), edge_density=0.15, verbose=False)
    edges_learned = [tuple(e) for e in res["best_edges"]]; T["structure"] = time.time() - t0
    out["structure"] = dict(best_seed=int(res["best_seed"]), best_bic=float(res["best_score"]), edges=[list(e) for e in edges_learned], n_edges=len(edges_learned), seeds_reaching_best=int(sum(abs(s[0] - res["best_score"]) < 1e-6 for s in res["results"])))
    full_edges = edges_learned + IND_EDGES
    par, ch, cop, blanket = blanket_of(full_edges); out["blanket"] = dict(parents=par, children=ch, co_parents=cop, blanket=blanket)
    log(f"[{stage} seed {seed}] structure {len(edges_learned)} edges, BIC {res['best_score']:.2f}, best seed {res['best_seed']}, blanket {blanket} ({T['structure']:.0f}s)")
    # ---- EM
    t0 = time.time(); d = ftr[all_nodes]
    model, hist, why = em.run_em(full_edges, d, sn, lambda m: ll.compute_log_likelihood(m, d), aggregate=True, nonmono_tol=1e18 if stage == "after" else 1e-6, verbose=False, **cfg["em"])
    T["em"] = time.time() - t0; out["em"] = dict(iterations=len(hist), stop=why, train_ll=float(hist[-1]["train_ll"]), test_ll=float(ll.compute_log_likelihood(model, fte[all_nodes])))
    log(f"[{stage} seed {seed}] EM {len(hist)} iterations ({why}), {T['em']:.0f}s")
    # ---- BN test posteriors
    t0 = time.time(); post = sens.make_posterior_fn(model); test_rows = list(fte.index)
    def ev(i):
        r_ = fte.loc[i]; e = {c: r_[c] for c in nodes if c != "Default" and pd.notnull(r_[c])}; e["M_Income"] = int(r_["M_Income"]); e["M_Dependents"] = int(r_["M_Dependents"]); return e
    evid = {i: ev(i) for i in test_rows}; pbn = np.array([post(evid[i]) for i in test_rows]); valid = np.isfinite(pbn)
    out["n_bn_defined"] = int(valid.sum()); T["bn_posteriors"] = time.time() - t0
    # ---- LR, RF
    t0 = time.time(); Xtr, Xte = build_design(cfg["raw"], tr, te, edges, cfg["edge_vars"], cfg["nat_vars"])
    lr = LogisticRegression(**cfg["lr"]).fit(Xtr.values, ytr); rf = RandomForestClassifier(n_jobs=n_jobs, **cfg["rf"]).fit(Xtr.values, ytr)
    plr, prf = lr.predict_proba(Xte.values)[:, 1], rf.predict_proba(Xte.values)[:, 1]; out["lr_n_iter"] = int(np.max(lr.n_iter_)); out["imputed_dependents_wrap"] = dict(LAST_DEP); T["fit_lr_rf"] = time.time() - t0
    # ---- OOF thresholds (training rows only)
    t0 = time.time(); cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # out-of-fold LR as in the cost curves: application = the grid-search LR (saga, 04 cell 7); after acceptance = converged lbfgs refit (08 cell 2: lbfgs, C=100, l2, balanced, max_iter 5000)
    lr_oof_model = clone(lr) if stage == "application" else LogisticRegression(solver="lbfgs", max_iter=5000, C=cfg["lr"]["C"], penalty="l2", class_weight="balanced")
    oof_lr = cross_val_predict(lr_oof_model, Xtr.values, ytr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    oof_rf = cross_val_predict(clone(rf), Xtr.values, ytr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]; T["oof"] = time.time() - t0
    RS = (5, 10); Tau = {"BN": {r: 1 / (1 + r) for r in RS}, "LR": {r: dv.oof_tau(oof_lr, ytr, r, "highest")[0] for r in RS}, "RF": {r: dv.oof_tau(oof_rf, ytr, r, "highest")[0] for r in RS}}
    out["tau"] = {m: {str(r): float(Tau[m][r]) for r in RS} for m in Tau}
    log(f"[{stage} seed {seed}] LR/RF fitted, OOF done ({T['fit_lr_rf']:.0f}s + {T['oof']:.0f}s)")
    # ---- evaluation
    yv = yte[valid]; pv = {"BN": pbn[valid], "LR": plr[valid], "RF": prf[valid]}
    out["auc"] = {"BN": float(roc_auc_score(yv, pbn[valid])), "LR": float(roc_auc_score(yte, plr)), "RF": float(roc_auc_score(yte, prf)), "LR_common": float(roc_auc_score(yv, plr[valid])), "RF_common": float(roc_auc_score(yv, prf[valid]))}
    for nm, other in (("LR", pv["LR"]), ("RF", pv["RF"])):
        obs, dif = paired_boot_full(yv, pv["BN"], other); lo, hi = np.percentile(dif, [2.5, 97.5]); out[f"auc_BN_minus_{nm}"] = dict(observed=float(obs), boot_mean=float(dif.mean()), lo=float(lo), hi=float(hi))
    pt = savings_point(yv, pv, Tau, RS); W = boot_weights(len(yv)); bt = saving_boot(yv, pv, Tau, list(RS), W=W)
    out["saving"] = {}; out["saving_diff"] = {}
    for j, r in enumerate(RS):
        for m in ("BN", "LR", "RF"):
            lo, hi = pct_ci(100 * bt[m][:, j]); out["saving"][f"{m}_r{r}"] = dict(point=100 * pt[(m, r)][0], lo=lo, hi=hi, flagged=int(pt[(m, r)][1]))
        for m in ("LR", "RF"):
            lo, hi = pct_ci(100 * (bt["BN"][:, j] - bt[m][:, j])); out["saving_diff"][f"BN_minus_{m}_r{r}"] = dict(point=100 * (pt[("BN", r)][0] - pt[(m, r)][0]), lo=lo, hi=hi)
    # ---- blanket sensitivity
    t0 = time.time(); mbv = [b for b in blanket if b not in ("M_Income", "M_Dependents")]
    rows_ok = [i for i, ok in zip(test_rows, valid) if ok]
    if mbv:
        STATES = {c: sn[c] for c in nodes if c != "Default"}
        res_b = sens.analyze_blanket(model, rows_ok, evid, mbv, STATES, 1 / 11, skip_unobserved=True, verbose=False)
        out["sexp"] = {v: dict(mean=float(res_b[res_b.variable == v].S_exp.mean()), n=int(res_b[res_b.variable == v].S_exp.notna().sum())) for v in mbv}
    else: out["sexp"] = {}
    T["sexp"] = time.time() - t0; T["total"] = time.time() - t_all; out["seconds"] = T
    out["preds"] = dict(pbn=pbn, plr=plr, prf=prf, yte=yte, test_rows=np.asarray(test_rows), valid=valid)
    log(f"[{stage} seed {seed}] done in {T['total']/60:.1f} min")
    return out
