"""J20: VOI-guided data collection. python j20.py STAGE   (STAGE application|after). Run once; no variants.
Candidate set K = the blanket variables; base evidence = every variable not in K (Monthly Income / Dependents handled by their missingness indicators as usual); each row observes exactly ONE variable of K (its recorded value).
Policies: P0 nothing; P1 adaptive VOI (argmax_j VOI_j(base), ties to the lower index, no labels); P2 fixed variable j*(r) = highest mean VOI_j(base) on TRAINING rows; P3 adaptive S_exp(j|base) = sum_x w_x |p_x - p0|;
P4 random (20 draws, seeds 0-19); P5 all of K observed (the main model); P6 LR and RF retrained on base + X_j*(r) (thresholds from 5-fold out-of-fold training predictions).
Descriptive only: best single fixed variable by realised TEST cost. Rows: the 29,998 flip-rate rows. Output j20/STAGE_results.json and j20_*.csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from pgmpy.inference import VariableElimination
stage = sys.argv[1]; name = {"application": "application", "after": "after_full"}[stage]; t_all = time.time(); NJ = int(os.environ.get("RF_JOBS", "4"))
RS = (5, 10, 15, 20); LOG = lambda s: print(s, flush=True); sh = lambda s: s.replace(" (bin)", "")
fit, info, _ = load_fit(name, ADD + "/j17"); ctx = variant_ctx(fit, None, None, name, base="skip"); bn, S, ev_cols, codes_te = ctx["bn"], ctx["S"], ctx["ev_cols"], ctx["codes"]
g = get_g(); y_all = ctx["y"]; cfg = variant_cfg(stage, [])
K = [c for c in fit["blanket"] if not c.startswith("M_")]                      # blanket variables, ordered as the blanket list (alphabetical) -> index order for ties
K = sorted(K); KI = [ev_cols.index(c) for c in K]
# ---------------------------------------------------------------- the pattern engine
ENG = VariableElimination(bn)
def tables(codes, U_idx, cand_idx):
    """For rows with evidence codes, variables U (indices) forced unobserved: p0 = P(Y=1|base); per candidate j: w_j (P(X_j=x|base)), px_j (P(Y=1|base, X_j=x)). Unique-pattern computation; returns (inv, p0, {j: (w, px)})."""
    c2 = codes.copy(); c2[:, U_idx] = -1; u, inv = np.unique(c2, axis=0, return_inverse=True); inv = np.asarray(inv).ravel()
    p0 = np.full(len(u), np.nan); T = {j: (np.zeros((len(u), len(S[ev_cols[j]]))), np.full((len(u), len(S[ev_cols[j]])), np.nan)) for j in cand_idx}
    for k, row in enumerate(u):
        ev = {c: S[c][i] for c, i in zip(ev_cols, row) if i >= 0}
        try:
            with np.errstate(invalid="ignore", divide="ignore"): p0[k] = ENG.query(variables=["Default"], evidence=ev, show_progress=False).get_value(Default=1)
        except Exception: continue
        for j in cand_idx:
            v = ev_cols[j]
            try:
                with np.errstate(invalid="ignore", divide="ignore"): f = ENG.query(variables=[v], evidence=ev, show_progress=False)
                w = dict(zip(f.state_names[v], np.asarray(f.values, float)))
            except Exception: continue
            for s_, ws in w.items():
                si = S[v].index(s_); T[j][0][k, si] = ws if np.isfinite(ws) else 0.0
                try:
                    with np.errstate(invalid="ignore", divide="ignore"): T[j][1][k, si] = ENG.query(variables=["Default"], evidence={**ev, v: s_}, show_progress=False).get_value(Default=1)
                except Exception: pass
    return inv, p0, T, u
def voi(p0, w, px, r):
    cb = np.minimum(r * p0, 1 - p0); ca = np.where(w > 0, w * np.minimum(r * px, 1 - px), 0.0).sum(axis=1); return np.maximum(cb - ca, 0.0)
def sexp0(p0, w, px): return np.where(w > 0, w * np.abs(px - p0[:, None]), 0.0).sum(axis=1)

# ---------------------------------------------------------------- common rows (the 29,998 flip-rate rows)
rows_all = fit["test_rows"]
if stage == "after": common = fit["valid"].copy()
else:
    per = pickle.load(open(R + "/selection/predictions/sel_rq2_per_row.pkl", "rb"))["per_row"]
    ga = per[per.variable == "Age (bin)"].set_index("row_index"); gd = per[per.variable == "Debt Ratio (bin)"].set_index("row_index")
    keep = ga.index[ga.retracted_prob.notna().values & gd.retracted_prob.reindex(ga.index).notna().values]; common = np.isin(rows_all, keep)
assert common.sum() == 29998, common.sum()
y = y_all[common]; n = len(y)
# ---------------------------------------------------------------- gates (full evidence)
auc_gate = {"after": (0.8519, 0.8618, 0.8603), "application": (0.6627, 0.6904, 0.6926)}[stage]
v_ = fit["valid"]; got = (roc_auc_score(y_all[v_], fit["pbn"][v_]), roc_auc_score(y_all, fit["plr"]), roc_auc_score(y_all, fit["prf"]))
assert tuple(round(a, 4) for a in got) == auc_gate, ("GATE FAIL full-evidence AUCs", got); LOG(f"PASS GATE full-evidence AUCs BN/LR/RF {[round(a,4) for a in got]} = {auc_gate}")
ref7 = {"after": {5: 35.2, 10: 47.7}, "application": {5: 0.0, 10: 9.9}}[stage]
for r in (5, 10):
    yy = y_all[v_]; fl = fit["pbn"][v_] >= 1 / (1 + r); sv = 100 * dv.saving(dv.cost(yy, fl, r), dv.trivial(yy, r)["cheaper"])
    assert round(sv, 1) == ref7[r], ("GATE FAIL Table 7", r, sv); LOG(f"PASS GATE Table 7 BN saving r={r}: {sv:.2f} = {ref7[r]}")
# VOI gate: the engine with U = {first blanket variable}, candidate = that variable reproduces the J18 values (mean VOI on the 29,998 rows)
gv = K[0]; gj = ev_cols.index(gv); t0 = time.time()
inv_g, p0_g, T_g, _ = tables(codes_te, [gj], [gj]); w_g, px_g = T_g[gj]
j18 = pd.read_csv(ADD + "/j18_voi_by_variable_r.csv"); lab = "Application" if stage == "application" else "After acceptance"
for r in RS:
    vv = voi(p0_g[inv_g], w_g[inv_g], px_g[inv_g], r)[common].mean(); ref = j18[(j18.stage == lab) & (j18.variable == sh(gv)) & (j18.r == r)].mean_voi.iloc[0]
    assert abs(vv - ref) < 1e-9, ("GATE FAIL VOI", gv, r, vv, ref); LOG(f"PASS GATE J18 mean VOI {sh(gv)} r={r}: {vv:.6f} = {ref:.6f}")
LOG(f"gates passed ({time.time()-t0:.0f}s)")
# ---------------------------------------------------------------- test and training tables with all of K unobserved
t0 = time.time(); inv_te, p0_te, T_te, u_te = tables(codes_te, KI, KI); LOG(f"test: {len(u_te)} base patterns ({time.time()-t0:.0f}s)")
tr, te = get_split(42); edges = get_edges(tr); ftr, _ = bn_frame(cfg, tr, edges); Fs = rq3._states_frame(ftr, ev_cols, S); codes_tr = np.column_stack([pd.Index(S[c]).get_indexer(Fs[c]) for c in ev_cols])
t0 = time.time(); inv_tr, p0_tr, T_tr, u_tr = tables(codes_tr, KI, KI); LOG(f"train: {len(u_tr)} base patterns ({time.time()-t0:.0f}s)")
ytr = ytr_ = np.asarray(g["y_train"]).astype(int)
act = {j: codes_te[:, j] for j in KI}                                          # actual (recorded) state of each candidate for each test row
# per-row p_x at the actual value; rows with any undefined needed quantity are dropped from every policy (counted)
PA = np.column_stack([T_te[j][1][inv_te][np.arange(len(codes_te)), act[j]] for j in KI]); ok = common & np.isfinite(p0_te[inv_te]) & np.isfinite(PA).all(axis=1)
for j in KI: ok &= ~((T_te[j][0][inv_te] > 0) & ~np.isfinite(T_te[j][1][inv_te])).any(axis=1)
LOG(f"rows used {int(ok.sum())} of {int(common.sum())}"); assert ok.sum() == common.sum() or abs(ok.sum() - common.sum()) <= 5
sel = ok; ys = y_all[sel]; ns = len(ys); P0 = p0_te[inv_te][sel]; PAs = PA[sel]
Wc = np.random.default_rng(42).multinomial(ns, np.full(ns, 1.0 / ns), size=B).astype(np.float32)      # shared 2,000 resamples of the common rows (default_rng(42), as the earlier row bootstraps)
D = Wc @ ys.astype(np.float32)
def trivial(r): N1 = ys.sum(); return min(r * N1, ns - N1)
def cost_vec(flag, r): return np.where(ys == 1, r * (~flag), flag).astype(float)
def sav(c, r): return 100 * (1 - c.sum() / trivial(r))
def sav_boot(c, r): return 100 * (1 - (Wc @ c.astype(np.float32)) / np.minimum(r * D, ns - D))
RES = {}
for r in RS:
    tau = 1 / (1 + r); R_ = {}
    V = np.column_stack([voi(p0_te[inv_te], T_te[j][0][inv_te], T_te[j][1][inv_te], r)[sel] for j in KI]); SX = np.column_stack([sexp0(p0_te[inv_te], T_te[j][0][inv_te], T_te[j][1][inv_te])[sel] for j in KI])
    jv = np.argmax(np.round(V, 12), axis=1); js = np.argmax(np.round(SX, 12), axis=1)
    # training-chosen fixed variable
    mv = []
    for j in KI:
        vt = voi(p0_tr[inv_tr], T_tr[j][0][inv_tr], T_tr[j][1][inv_tr], r); mv.append(np.nanmean(vt))
    jstar = int(np.argmax(np.round(mv, 12)))
    pick = lambda idx: PAs[np.arange(ns), idx]
    pol = {"P0": P0, "P1": pick(jv), "P2": PAs[:, jstar], "P3": pick(js), "P5": fit["pbn"][sel]}
    cost = {k: cost_vec(p >= tau, r) for k, p in pol.items()}
    rr = np.zeros(ns)
    for dseed in range(20): rd = np.random.RandomState(dseed).randint(0, len(KI), ns); rr += cost_vec(pick(rd) >= tau, r)
    cost["P4"] = rr / 20.0
    for jj, j in enumerate(KI): cost[f"fixed:{sh(K[jj])}"] = cost_vec(PAs[:, jj] >= tau, r)
    R_.update(cost=cost, jstar=jstar, mean_voi_train=[float(m) for m in mv], share_P1=[float((jv == q).mean()) for q in range(len(KI))], share_P3=[float((js == q).mean()) for q in range(len(KI))], tau=tau)
    RES[r] = R_
# ---------------------------------------------------------------- P6: retrained LR and RF with base + X_j*
rawK = [sh(c) for c in K]; base_raw = [c for c in cfg["raw"] if c not in rawK]; P6 = {}; cache = {}
for r in RS:
    jstar = RES[r]["jstar"]
    if jstar not in cache:
        raw = base_raw + [rawK[jstar]]; bd = g["build_design"](raw); Xtr, Xte = bd["Xtr"].values, bd["Xte"].values
        lrm, sol = lr_fit(cfg["lr"], Xtr, ytr); rfm = RandomForestClassifier(n_jobs=NJ, **cfg["rf"]).fit(Xtr, ytr)
        o_lr = oof_pred(oof_lr_model(stage, lrm), Xtr, ytr); o_rf = oof_pred(clone(rfm).set_params(n_jobs=NJ), Xtr, ytr)
        cache[jstar] = dict(plr=lrm.predict_proba(Xte)[:, 1][sel], prf=rfm.predict_proba(Xte)[:, 1][sel], tl=tau_pair(o_lr, ytr, RS), tf=tau_pair(o_rf, ytr, RS), solver=sol, auc_lr=float(roc_auc_score(y_all, lrm.predict_proba(Xte)[:, 1])), auc_rf=float(roc_auc_score(y_all, rfm.predict_proba(Xte)[:, 1])))
        LOG(f"P6 models for j* = {rawK[jstar]} fitted ({sol}); AUC LR {cache[jstar]['auc_lr']:.4f} RF {cache[jstar]['auc_rf']:.4f}; thresholds LR {cache[jstar]['tl']} RF {cache[jstar]['tf']}")
    c_ = cache[jstar]; RES[r]["cost"]["P6_LR"] = cost_vec(c_["plr"] >= c_["tl"][r], r); RES[r]["cost"]["P6_RF"] = cost_vec(c_["prf"] >= c_["tf"][r], r); RES[r]["p6"] = dict(j=rawK[jstar], tau_LR=c_["tl"][r], tau_RF=c_["tf"][r], solver=c_["solver"], auc_LR=c_["auc_lr"], auc_RF=c_["auc_rf"])
# ---------------------------------------------------------------- summaries
OUT = dict(stage=stage, K=rawK, n_rows=int(ns), policies={}, contrasts={}, shares={}, jstar={}, descriptive_best={}, P6={}, patterns=dict(test=int(len(u_te)), train=int(len(u_tr))))
for r in RS:
    cst = RES[r]["cost"]; triv = trivial(r)
    OUT["policies"][r] = {k: dict(cost_per_row=float(c.mean()), saving=sav(c, r), lo=float(np.percentile(sav_boot(c, r), 2.5)), hi=float(np.percentile(sav_boot(c, r), 97.5))) for k, c in cst.items()}
    sb = {k: sav_boot(c, r) for k, c in cst.items() if not k.startswith("fixed")}
    ct = {}
    for a, b in (("P1", "P2"), ("P1", "P3"), ("P1", "P4"), ("P1", "P6_LR"), ("P1", "P6_RF"), ("P1", "P0"), ("P1", "P5"), ("P2", "P6_LR"), ("P2", "P6_RF")):
        d = sb[a] - sb[b]; lo, hi = np.percentile(d, [2.5, 97.5]); ct[f"{a}-{b}"] = dict(diff=sav(cst[a], r) - sav(cst[b], r), lo=float(lo), hi=float(hi))
    OUT["contrasts"][r] = ct; OUT["shares"][r] = dict(P1={rawK[q]: RES[r]["share_P1"][q] for q in range(len(KI))}, P3={rawK[q]: RES[r]["share_P3"][q] for q in range(len(KI))}); OUT["jstar"][r] = dict(variable=rawK[RES[r]["jstar"]], mean_voi_train={rawK[q]: RES[r]["mean_voi_train"][q] for q in range(len(KI))})
    fx = {k: v for k, v in OUT["policies"][r].items() if k.startswith("fixed")}; b = max(fx, key=lambda k: fx[k]["saving"]); OUT["descriptive_best"][r] = dict(variable=b.split(":")[1], **fx[b]); OUT["P6"][r] = RES[r]["p6"]
OUT["seconds"] = time.time() - t_all
json.dump(OUT, open(f"{ADD}/j20/{stage}_results.json", "w"), indent=1, default=float)
for r in (5, 10):
    LOG(f"r={r}: " + "; ".join(f"{k} {v['saving']:.2f}" for k, v in OUT["policies"][r].items() if not k.startswith("fixed")))
    LOG("   " + "; ".join(f"{k} {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]" for k, v in OUT["contrasts"][r].items()))
LOG(f"done {OUT['seconds']/60:.1f} min")
