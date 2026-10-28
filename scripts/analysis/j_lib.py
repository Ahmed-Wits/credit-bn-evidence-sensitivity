"""Shared code for J17, J18 and J19. Reuses split_robust (J12 pipeline pieces), shared/src/rq3.py (masking), fast_auc and costlib.
fit_variant() is the J12 run_seed() orchestration with (a) a list of dropped variables and (b) the fitted objects returned/saved."""
import os, sys, json, pickle, time, copy, warnings, importlib.util
os.environ.setdefault("PYTHONHASHSEED", "0"); assert os.environ["PYTHONHASHSEED"] == "0"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from split_robust import *          # noqa: F401,F403  (dfc, CFG, get_split, get_edges, bn_frame, blanket_of, stx, em, ll, dv, sens, ...)
import split_robust as SR
from costlib import savings_point, saving_boot
from addcommon import boot_weights, pct_ci, ADD, R as _R
import rq3, fast_auc as fa, checks as ck
from pgmpy.models import DiscreteBayesianNetwork

MK = rq3.MK
SEEDS = list(range(42, 52)); LEVELS = [0.2, 0.4, 0.5]; MECHS = ["mcar", "dependent"]

# ------------------------------------------------------------------ the 00_precheck globals (build_design etc.), with the crash-free 'wrap' binning of rq3 v2 (as j16.py)
_G = {}
def get_g():
    if _G: return _G
    sp = pickle.load(open(R + "/Data/processed/split_indices.pkl", "rb"))
    g = {"os": os, "np": np, "pd": pd, "L": lambda q: pickle.load(open(q, "rb")), "PROC": PROC, "ROOT": R}
    g["train_idx"], g["test_idx"] = np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"]); g["y_train"], g["y_test"] = np.asarray(sp["y_train"]), np.asarray(sp["y_test"])
    code = "".join(json.load(open(R + "/selection/notebooks/00_precheck.ipynb"))["cells"][2]["source"])
    exec(compile(code[:code.index("# ---- validation")], "build_design_src", "exec"), g)
    g["bin_natural_category"] = lambda s, max_category=3: rq3._bin_natural(s, "wrap", max_category)
    _G.update(g); return _G

def variant_cfg(stage, drop):
    cfg = copy.deepcopy(CFG[stage]); dn = [d + " (bin)" for d in drop]
    cfg["nodes"] = [n for n in cfg["nodes"] if n not in dn]
    cfg["edge_vars"] = [v for v in cfg["edge_vars"] if v not in drop]; cfg["nat_vars"] = [v for v in cfg["nat_vars"] if v not in drop]
    cfg["raw"] = [v for v in cfg["raw"] if v not in drop]
    return cfg

def lr_fit(params, X, y, name="LR"):
    """LR with the locked parameters (saga); the convergence check of checks.fit_converged; if saga does not converge, the same penalty/C with lbfgs (max_iter 5000), recorded."""
    m = LogisticRegression(**params)
    try:
        ck.fit_converged(m, X, y, name=name); return m, "saga"
    except AssertionError:
        p2 = dict(params); p2.update(solver="lbfgs", max_iter=5000); m = LogisticRegression(**p2); ck.fit_converged(m, X, y, name=name + " (lbfgs fallback)"); return m, "lbfgs-fallback"

def oof_lr_model(stage, lr):
    """J12 rule: Application OOF with the grid-search LR (saga); After acceptance OOF with the converged lbfgs refit (C as locked)."""
    if stage == "application": return clone(lr)
    return LogisticRegression(solver="lbfgs", max_iter=5000, C=lr.C, penalty="l2", class_weight="balanced")

def oof_pred(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

def tau_pair(oof, y, rs=(5, 10)):
    return {r: dv.oof_tau(oof, y, r, "highest")[0] for r in rs}

# ------------------------------------------------------------------ fit one BN variant (J12 run_seed orchestration, seed 42)
def fit_variant(name, stage, drop, outdir, log=print, n_jobs=4):
    t_all = time.time(); T = {}; cfg = variant_cfg(stage, drop); g = get_g()
    tr, te = get_split(42); assert np.array_equal(tr, g["train_idx"]) and np.array_equal(te, g["test_idx"])
    ytr = dfc.loc[tr, "Default"].values.astype(int); yte = dfc.loc[te, "Default"].values.astype(int)
    edges = get_edges(tr); out = dict(name=name, stage=stage, drop=list(drop), n_variables=len([c for c in cfg["nodes"] if c != "Default"]), variables=[c.replace(" (bin)", "") for c in cfg["nodes"] if c != "Default"])
    out["bin_edges"] = {c: [float(x) for x in edges[c]] for c in EQ5}
    ftr, cats = bn_frame(cfg, tr, edges); fte, _ = bn_frame(cfg, te, edges)
    nodes = cfg["nodes"]; all_nodes = nodes + ["M_Income", "M_Dependents"]; sn = {}
    for c in all_nodes: sn[c] = [0, 1] if (c == "Default" or c.startswith("M_")) else cats[c]
    cc = ftr[(ftr.M_Income == 1) & (ftr.M_Dependents == 1)]; out["n_complete"] = len(cc)
    t0 = time.time(); res = stx.learn_structure(cc[nodes].copy(), nodes, seeds=range(42, 57), edge_density=0.15, verbose=False)
    edges_learned = [tuple(e) for e in res["best_edges"]]; T["structure"] = time.time() - t0
    out["structure"] = dict(best_seed=int(res["best_seed"]), best_bic=float(res["best_score"]), edges=[list(e) for e in edges_learned], n_edges=len(edges_learned), seeds_reaching_best=int(sum(abs(s[0] - res["best_score"]) < 1e-6 for s in res["results"])))
    full_edges = edges_learned + IND_EDGES; par, ch, cop, blanket = blanket_of(full_edges)
    out["blanket"] = dict(parents=par, children=ch, co_parents=cop, blanket=blanket)
    log(f"[{name}] structure {len(edges_learned)} edges, BIC {res['best_score']:.2f}, seed {res['best_seed']}, blanket {blanket} ({T['structure']:.0f}s)")
    t0 = time.time(); d = ftr[all_nodes]
    model, hist, why = em.run_em(full_edges, d, sn, lambda m: ll.compute_log_likelihood(m, d), aggregate=True, nonmono_tol=1e18 if stage == "after" else 1e-6, verbose=False, **cfg["em"])
    T["em"] = time.time() - t0; out["em"] = dict(iterations=len(hist), stop=why, train_ll=float(hist[-1]["train_ll"]))
    log(f"[{name}] EM {len(hist)} iterations ({why}), {T['em']:.0f}s")
    post = sens.make_posterior_fn(model); test_rows = list(fte.index)
    def ev(i):
        r_ = fte.loc[i]; e = {c: r_[c] for c in nodes if c != "Default" and pd.notnull(r_[c])}; e["M_Income"] = int(r_["M_Income"]); e["M_Dependents"] = int(r_["M_Dependents"]); return e
    evid = {i: ev(i) for i in test_rows}; pbn = np.array([post(evid[i]) for i in test_rows]); valid = np.isfinite(pbn); out["n_bn_defined"] = int(valid.sum())
    # LR, RF on the design of the global build_design (identical to J12's; checked against the stored predictions for the full variant)
    t0 = time.time(); base = g["build_design"](cfg["raw"]); Xtr, Xte = base["Xtr"].values, base["Xte"].values
    lr, lr_solver = lr_fit(cfg["lr"], Xtr, ytr); rf = RandomForestClassifier(n_jobs=n_jobs, **cfg["rf"]).fit(Xtr, ytr)
    plr, prf = lr.predict_proba(Xte)[:, 1], rf.predict_proba(Xte)[:, 1]; out["lr_solver"] = lr_solver; out["lr_n_iter"] = int(np.max(lr.n_iter_)); T["fit_lr_rf"] = time.time() - t0
    t0 = time.time(); oof_lr = oof_pred(oof_lr_model(stage, lr), Xtr, ytr); oof_rf = oof_pred(clone(rf).set_params(n_jobs=n_jobs), Xtr, ytr); T["oof"] = time.time() - t0
    RS = (5, 10); Tau = {"BN": {r: 1 / (1 + r) for r in RS}, "LR": tau_pair(oof_lr, ytr), "RF": tau_pair(oof_rf, ytr)}
    out["tau"] = {m: {str(r): float(Tau[m][r]) for r in RS} for m in Tau}
    log(f"[{name}] LR ({lr_solver}), RF fitted; OOF thresholds {out['tau']} ({T['fit_lr_rf']:.0f}s + {T['oof']:.0f}s)")
    yv = yte[valid]; pv = {"BN": pbn[valid], "LR": plr[valid], "RF": prf[valid]}
    out["auc"] = {"BN": float(roc_auc_score(yv, pbn[valid])), "LR": float(roc_auc_score(yte, plr)), "RF": float(roc_auc_score(yte, prf)), "LR_common": float(roc_auc_score(yv, plr[valid])), "RF_common": float(roc_auc_score(yv, prf[valid]))}
    for nm, other in (("LR", pv["LR"]), ("RF", pv["RF"])):
        obs, dif = paired_boot_full(yv, pv["BN"], other); lo, hi = np.percentile(dif, [2.5, 97.5]); out[f"auc_BN_minus_{nm}"] = dict(observed=float(obs), boot_mean=float(dif.mean()), lo=float(lo), hi=float(hi))
    pt = savings_point(yv, pv, Tau, RS); W = boot_weights(len(yv)); bt = saving_boot(yv, pv, Tau, list(RS), W=W); out["saving"] = {}; out["saving_diff"] = {}
    for j, r in enumerate(RS):
        for m in ("BN", "LR", "RF"):
            lo, hi = pct_ci(100 * bt[m][:, j]); out["saving"][f"{m}_r{r}"] = dict(point=100 * pt[(m, r)][0], lo=lo, hi=hi, flagged=int(pt[(m, r)][1]))
        for m in ("LR", "RF"):
            lo, hi = pct_ci(100 * (bt["BN"][:, j] - bt[m][:, j])); out["saving_diff"][f"BN_minus_{m}_r{r}"] = dict(point=100 * (pt[("BN", r)][0] - pt[(m, r)][0]), lo=lo, hi=hi)
    # per-row blanket sensitivities (S_exp, retracted probability) for flip rates at any r and for J18
    t0 = time.time(); mbv = [b for b in blanket if b not in ("M_Income", "M_Dependents")]; rows_ok = [i for i, ok in zip(test_rows, valid) if ok]
    STATES = {c: sn[c] for c in nodes if c != "Default"}
    sres = sens.analyze_blanket(model, rows_ok, evid, mbv, STATES, 1 / 11, skip_unobserved=True, verbose=False) if mbv else pd.DataFrame()
    T["sexp"] = time.time() - t0; T["total"] = time.time() - t_all; out["seconds"] = T
    os.makedirs(outdir, exist_ok=True)
    sres.to_pickle(f"{outdir}/{name}_sens_rows.pkl")
    pickle.dump(dict(model_edges=[tuple(e) for e in model.edges()], cpds=model.get_cpds(), nodes=nodes, all_nodes=all_nodes, sn=sn, fte=fte, test_rows=np.asarray(test_rows), yte=yte, valid=valid, pbn=pbn, plr=plr, prf=prf,
                     tau=Tau, ytr=ytr, cfg_raw=cfg["raw"], blanket=blanket, drop=list(drop), stage=stage), open(f"{outdir}/{name}_fit.pkl", "wb"))
    json.dump(out, open(f"{outdir}/{name}_fit.json", "w"), indent=1, default=float)
    log(f"[{name}] done in {T['total']/60:.1f} min")
    return out

def load_fit(name, outdir):
    return pickle.load(open(f"{outdir}/{name}_fit.pkl", "rb")), json.load(open(f"{outdir}/{name}_fit.json")), pd.read_pickle(f"{outdir}/{name}_sens_rows.pkl")

# ------------------------------------------------------------------ rq3-style context for a variant
def variant_ctx(fit, lr, rf, name, base=None):
    """A context of the same shape as rq3.load_context() for a freshly fitted variant (BN from saved edges/cpds, LR/RF objects given)."""
    g = get_g(); raw_cols = fit["cfg_raw"]; base = g["build_design"](raw_cols) if base is None else (None if isinstance(base, str) else base)
    bn = DiscreteBayesianNetwork([tuple(e) for e in fit["model_edges"]]); bn.add_cpds(*fit["cpds"]); assert bn.check_model()
    S = {n: list(bn.get_cpds(n).state_names[n]) for n in bn.nodes()}
    ev_cols = [c for c in fit["nodes"] if c != "Default"] + ["M_Income", "M_Dependents"]
    Fs = rq3._states_frame(fit["fte"], ev_cols, S)
    codes = np.column_stack([pd.Index(S[c]).get_indexer(Fs[c]) for c in ev_cols])
    for j, c in enumerate(ev_cols): assert ((codes[:, j] < 0) == Fs[c].isna().to_numpy()).all(), "unseen state in %s" % c
    p0 = {"BN": fit["pbn"], "LR": fit["plr"], "RF": fit["prf"]}
    tau = {"BN": 1 / 6, "LR": fit["tau"]["LR"][5], "RF": fit["tau"]["RF"][5]}
    return dict(name=name, g=g, dfc=g["dfc"], raw_cols=raw_cols, base=base, y=g["y_test"].astype(int), train_idx=g["train_idx"], test_idx=g["test_idx"], models={"LR": lr, "RF": rf},
                p0=p0, tau=tau, bn=bn, S=S, ev_cols=ev_cols, codes=codes, imp_cache={})

# ------------------------------------------------------------------ bootstrap of masking contrasts (notebook 10 design)
B = 2000
_W = {}
def get_W(n):
    if n not in _W:
        rng = np.random.default_rng(42); _W[n] = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.int16)
    return _W[n]
def fixp(p): return np.nan_to_num(np.asarray(p, float), nan=0.0)
def boot_auc(score, y, v, W=None):
    """B weighted AUCs on rows with v == 1 (weights zeroed elsewhere), using the shared multinomial counts."""
    W = get_W(len(y)) if W is None else W; prep = fa.prepare(fixp(score), y); out = np.empty(B); vf = np.asarray(v, float)
    for a in range(0, B, 500): out[a:a + 500] = fa.auc_weighted(prep, W[a:a + 500].astype(float) * vf, chunk=500)
    return out
def point_auc(score, y, v):
    prep = fa.prepare(fixp(score), y); return float(fa.auc_weighted(prep, np.asarray(v, float))[0])

class MaskAcc:
    """Accumulates, per (variable, level, mechanism), the 10-seed mean of [AUC masked - AUC baseline] for each named score set, as point estimate and as B bootstrap values."""
    def __init__(self, y):
        self.y = y; self.n = len(y); self.D = {}; self.P = {}; self.N = {}; self.auc = {}; self.cache = {}
    def add(self, key, seed, v, scores, baselines):
        """scores/baselines: {name: score vector}; both evaluated on rows v."""
        bc = {}
        for nm, s in scores.items():
            b = baselines[nm]
            ck_ = (nm, hash(v.tobytes()), id(b))
            if ck_ not in self.cache: self.cache[ck_] = (boot_auc(b, self.y, v), point_auc(b, self.y, v))
            ab = boot_auc(s, self.y, v); a0, p0 = self.cache[ck_]
            self.D.setdefault((key, nm), np.zeros(B)); self.D[(key, nm)] += ab - a0
            pa = point_auc(s, self.y, v)
            self.P[(key, nm)] = self.P.get((key, nm), 0.0) + (pa - p0)
            self.auc.setdefault((key, nm), []).append((seed, pa, p0))
        self.N[key] = self.N.get(key, 0) + 1
    def finish(self, key, names, ref="BN"):
        out = {}
        for nm in names:
            n = self.N[key]; out[nm] = (self.P[(key, nm)] / n, self.D[(key, nm)] / n)
        res = {}
        for nm in names:
            if nm == ref: continue
            C = out[ref][1] - out[nm][1]; pt = out[ref][0] - out[nm][0]; lo, hi = np.percentile(C, [2.5, 97.5])
            res[nm] = dict(point=float(pt), boot_mean=float(C.mean()), lo=float(lo), hi=float(hi), deg_ref=float(-out[ref][0]), deg_c=float(-out[nm][0]), n_seeds=n)
        return res, {nm: float(-out[nm][0]) for nm in names}

def reading(lo, hi): return "BN degrades less" if lo > 0 else ("BN degrades more" if hi < 0 else "tie")
