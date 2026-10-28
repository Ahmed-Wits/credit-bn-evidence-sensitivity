"""J21 (scoring step): all evidence subsets. python j21.py STAGE SHARD NSHARDS. For each subset S of the blanket variables K observed (the rest unobserved for ALL test rows) it saves, for all 30,000 test rows:
 BN posterior (exact marginalisation, one model); LR_B, RF_B (retrained on complete training data WITHOUT the unobserved variables; thresholds from 5-fold OOF training predictions, grid 0.01, ties to the highest);
 LR_IMP, RF_IMP (the article's stored full LR and RF, thresholds as Table 7, fed values from an IterativeImputer).
IMP imputer: with every row unobserved the imputer cannot be refit to an identically masked training copy (nothing to learn from); it is refit (random_state 42) to a training copy in which each unobserved variable is masked MCAR in 50% of
the training rows (the article's highest level; masks make_mask(n, 0.5, 'mcar', 42 + index)), then applied with all unobserved variables set to missing in every test row. Reported as a deviation.
Output: j21/STAGE_subsetNN.pkl; bootstrap, gaps and criterion in j21_agg.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from pgmpy.inference import VariableElimination
stage, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); name = {"application": "application", "after": "after_full"}[stage]; contribution = {"application": "selection", "after": "monitoring"}[stage]
NJ = int(os.environ.get("RF_JOBS", "3")); RS = (5, 10); LOG = lambda s: print(s, flush=True); sh = lambda s: s.replace(" (bin)", "")
fit, info, _ = load_fit(name, ADD + "/j17"); ctx_bn = variant_ctx(fit, None, None, name, base="skip"); bn, S, ev_cols, codes = ctx_bn["bn"], ctx_bn["S"], ctx_bn["ev_cols"], ctx_bn["codes"]
ctx = rq3.load_context(contribution); g = ctx["g"]; g["bin_natural_category"] = lambda s, max_category=3: rq3._bin_natural(s, "wrap", max_category)
cfg = variant_cfg(stage, []); K = sorted(c for c in fit["blanket"] if not c.startswith("M_")); rawK = [sh(c) for c in K]; ytr = np.asarray(g["y_train"]).astype(int)
J12 = json.load(open(f"{ADD}/j12/{stage}_seed42_gate.json")); TAU_STD = {m: {r: J12["tau"][m][str(r)] for r in RS} for m in ("LR", "RF")}
LRt, RFt = ctx["models"]["LR"], ctx["models"]["RF"]; ENG = VariableElimination(bn); imp_cols = rq3._imp_cols(ctx); des_cols = [c for c in g["DESIGN_ORDER"] if c in ctx["raw_cols"]]
def bn_post(U_idx):
    c2 = codes.copy(); c2[:, U_idx] = -1; u, inv = np.unique(c2, axis=0, return_inverse=True); inv = np.asarray(inv).ravel(); pu = np.full(len(u), np.nan)
    for k, row in enumerate(u):
        ev = {c: S[c][i] for c, i in zip(ev_cols, row) if i >= 0}
        try:
            with np.errstate(invalid="ignore", divide="ignore"): p = float(ENG.query(variables=["Default"], evidence=ev, show_progress=False).get_value(Default=1))
            pu[k] = p if (np.isfinite(p) and 0 <= p <= 1) else np.nan
        except Exception: pass
    return pu[inv], len(u)
def imputed_matrix(U_raw):
    """article's design matrix for the test rows with the variables U_raw unobserved in every row, values from an IterativeImputer (training copy masked MCAR 50%, see header)"""
    tr = ctx["dfc"].loc[ctx["train_idx"], imp_cols].astype(float); te = ctx["dfc"].loc[ctx["test_idx"], imp_cols].astype(float)
    tr["Monthly Income"] = np.log1p(tr["Monthly Income"]); te["Monthly Income"] = np.log1p(te["Monthly Income"])
    for i, v in enumerate(U_raw):
        m_, _ = MK.make_mask(len(tr), 0.5, "mcar", 42 + i); tr.loc[m_, v] = np.nan
        te[v] = np.nan
    imp = IterativeImputer(random_state=42).fit(tr); tei = pd.DataFrame(imp.transform(te), columns=imp_cols, index=te.index); tei["Monthly Income"] = np.expm1(tei["Monthly Income"]).clip(lower=0)
    s_te = pd.DataFrame(index=tei.index)
    for c in des_cols: s_te[c + " (bin)"] = g["cut_edges"](tei[c], g["EDGES"][c]) if c in g["EDGE_VARS"] else rq3._bin_natural(tei[c], "wrap", 3)
    return pd.DataFrame(ctx["base"]["encoder"].transform(s_te.astype(str)), columns=ctx["base"]["Xte"].columns, index=tei.index).values
SUB = list(range(2 ** len(K)))                                  # bit i set = K[i] observed; the last subset (all observed) is the main model
t_all = time.time()
for si in SUB:
    if si % nsh != shard: continue
    t0 = time.time(); obs = [K[i] for i in range(len(K)) if si >> i & 1]; U = [K[i] for i in range(len(K)) if not si >> i & 1]; U_raw = [sh(c) for c in U]; U_idx = [ev_cols.index(c) for c in U]
    full = len(U) == 0
    pb, npat = (fit["pbn"].copy(), 0) if full else bn_post(U_idx)
    raw_B = [c for c in cfg["raw"] if c not in U_raw]; bd = g["build_design"](raw_B); Xtr, Xte = bd["Xtr"].values, bd["Xte"].values
    lrB, sol = lr_fit(cfg["lr"], Xtr, ytr); rfB = RandomForestClassifier(n_jobs=NJ, **cfg["rf"]).fit(Xtr, ytr)
    if full: tauB = {"LR": TAU_STD["LR"], "RF": TAU_STD["RF"]}
    else:
        o_lr = oof_pred(oof_lr_model(stage, lrB), Xtr, ytr); o_rf = oof_pred(clone(rfB).set_params(n_jobs=NJ), Xtr, ytr); tauB = {"LR": tau_pair(o_lr, ytr, RS), "RF": tau_pair(o_rf, ytr, RS)}
    pB = {"LR": lrB.predict_proba(Xte)[:, 1], "RF": rfB.predict_proba(Xte)[:, 1]}
    if full: pI = {"LR": LRt.predict_proba(ctx["base"]["Xte"].values)[:, 1], "RF": RFt.predict_proba(ctx["base"]["Xte"].values)[:, 1]}
    else: Xi = imputed_matrix(U_raw); pI = {"LR": LRt.predict_proba(Xi)[:, 1], "RF": RFt.predict_proba(Xi)[:, 1]}
    pickle.dump(dict(stage=stage, subset=si, observed=[sh(c) for c in obs], unobserved=U_raw, pbn=pb, pB=pB, pI=pI, tauB=tauB, tauI=TAU_STD, solver=sol, n_patterns=npat, y=ctx["y"], rows=fit["test_rows"], n_features_B=int(Xtr.shape[1])), open(f"{ADD}/j21/{stage}_subset{si:02d}.pkl", "wb"))
    LOG(f"[{stage}] subset {si:02d} observed {[sh(c) for c in obs]} done ({time.time()-t0:.0f}s; LR_B {sol}; BN patterns {npat})")
LOG(f"shard {shard} done {(time.time()-t_all)/60:.1f} min")
