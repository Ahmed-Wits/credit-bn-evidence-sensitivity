"""J23: expected probability of a decision change (ECP = 1 - SDP) next to S_exp, flip rate and VOI. Same rows as J18 (29,998), main split, both stages, r = 5, 10, 15, 20.
ECP_j(row) = sum_x w_x 1[flag(p_x) != flag(p0)] with p0 = P(Y=1|e_-j), p_x = P(Y=1|e_-j, X_j=x), w_x = P(X_j=x|e_-j), flag(p) = 1[p >= 1/(1+r)]; SDP_j = 1 - ECP_j.
Also the joint SDP of the whole blanket given the base evidence. Writes j23_*.csv and j23/j23_summary.pkl"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from j_tables import *
from addcommon import Part
from scipy.stats import spearmanr, pearsonr, kendalltau
from pgmpy.factors.discrete import TabularCPD
p = Part("j23"); RS = (5, 10, 15, 20); sh = lambda s: s.replace(" (bin)", ""); t_all = time.time(); LOG = lambda s: print(s, flush=True)
# ------------------------------------------------------------------ toy gate (by hand)
toy = DiscreteBayesianNetwork([("X", "Default")])
toy.add_cpds(TabularCPD("X", 2, [[0.6], [0.4]], state_names={"X": ["a", "b"]}), TabularCPD("Default", 2, [[0.95, 0.70], [0.05, 0.30]], evidence=["X"], evidence_card=[2], state_names={"Default": [0, 1], "X": ["a", "b"]}))
eng_t = VariableElimination(toy); inv_t, p0_t, W_t, PX_t = var_tables(eng_t, {"X": ["a", "b"]}, ["X"], np.array([[0]]), 0)
p.gate(f"toy case: p0 = {p0_t[0]:.4f} = 0.15, w = (0.6, 0.4), p_x = (0.05, 0.30)", abs(p0_t[0] - 0.15) < 1e-12 and np.allclose(W_t[0], [0.6, 0.4]) and np.allclose(PX_t[0], [0.05, 0.30]))
for r, (e_h, s_h) in {5: (0.4, 0.6), 10: (0.6, 0.4)}.items():
    e_, s_ = ecp(p0_t, W_t, PX_t, r)
    p.gate(f"toy case r={r}, tau = 1/{r+1}: flag0 = {int(p0_t[0] >= 1/(1+r))}; ECP {e_[0]:.4f} = {e_h} and SDP {s_[0]:.4f} = {s_h} by hand ({'only x = 2' if r == 5 else 'only x = 1'} changes the flag)", abs(e_[0] - e_h) < 1e-12 and abs(s_[0] - s_h) < 1e-12 and abs(e_[0] + s_[0] - 1) < 1e-12)
FR = pd.read_csv(ADD + "/flip_rate_by_r.csv"); J18 = pd.read_csv(ADD + "/j18_voi_by_variable_r.csv")
cells = []; ROW = {}; RANK = []; JOINT = []; maxdisc = 0.0
for stage, name in (("application", "application"), ("after", "after_full")):
    lab = {"application": "Application", "after": "After acceptance"}[stage]; t0 = time.time()
    fit, info, _ = load_fit(name, ADD + "/j17"); ctx = variant_ctx(fit, None, None, name, base="skip"); bn, S, ev_cols, codes = ctx["bn"], ctx["S"], ctx["ev_cols"], ctx["codes"]; eng = VariableElimination(bn)
    rows_all = fit["test_rows"]
    if stage == "after": common = fit["valid"].copy()
    else:
        per = pickle.load(open(R + "/selection/predictions/sel_rq2_per_row.pkl", "rb"))["per_row"]
        ga = per[per.variable == "Age (bin)"].set_index("row_index"); gd = per[per.variable == "Debt Ratio (bin)"].set_index("row_index")
        keep = ga.index[ga.retracted_prob.notna().values & gd.retracted_prob.reindex(ga.index).notna().values]; common = np.isin(rows_all, keep)
    assert common.sum() == 29998
    K = sorted(c for c in fit["blanket"] if not c.startswith("M_")); T = {}
    for v in K:
        j = ev_cols.index(v); inv, P0, W, PX = var_tables(eng, S, ev_cols, codes, j); obs = codes[:, j]
        T[v] = dict(inv=inv, P0=P0, W=W, PX=PX, obs=obs); LOG(f"[{stage}] {sh(v)}: {len(P0)} reduced patterns ({time.time()-t0:.0f}s)")
    ok = common.copy()
    for v in K:
        t = T[v]; pxo = t["PX"][t["inv"], t["obs"]]; ok &= np.isfinite(t["P0"][t["inv"]]) & np.isfinite(pxo) & ~((t["W"][t["inv"]] > 0) & ~np.isfinite(t["PX"][t["inv"]])).any(axis=1)
    p.check(f"{lab}: rows used {int(ok.sum()):,} of {int(common.sum()):,}", ok.sum() == common.sum())
    # ---- per variable and r
    for v in K:
        t = T[v]; inv, obs = t["inv"], t["obs"]; p0 = t["P0"][inv][ok]; w = t["W"][inv][ok]; px = t["PX"][inv][ok]; pxo = px[np.arange(len(p0)), obs[ok]]
        sexp = np.where(w > 0, w * np.abs(px - pxo[:, None]), 0.0); sexp[np.arange(len(p0)), obs[ok]] = 0.0; sexp = sexp.sum(axis=1)
        for r in RS:
            tau = 1 / (1 + r); e, sdp = ecp(p0, w, px, r); vo = voi(p0, w, px, r); flip = ((pxo >= tau) != (p0 >= tau)).astype(float)
            ROW[(lab, sh(v), r)] = dict(ecp=e, sdp=sdp, voi=vo, flip=flip, sexp=sexp)
            p.check(f"{lab} {sh(v)} r={r}: ECP + SDP = 1 (max dev {np.abs(e + sdp - 1).max():.1e}, rows with w sum != 1 ignored)", np.abs(e + sdp - 1).max() < 1e-6)
    LOG(f"[{stage}] per-variable quantities done ({time.time()-t0:.0f}s)")
    # ---- joint SDP of the whole blanket given the base evidence
    t0 = time.time(); KI = [ev_cols.index(v) for v in K]; invj, P0j, Wj, PXj = joint_tables(eng, S, ev_cols, codes, KI); LOG(f"[{stage}] joint blanket tables: {len(P0j)} base patterns, {Wj.shape[1]} joint states ({time.time()-t0:.0f}s)")
    okj = ok & np.isfinite(P0j[invj]) & ~((Wj[invj] > 0) & ~np.isfinite(PXj[invj])).any(axis=1)
    for r in (5, 10):
        e, sdp = ecp(P0j[invj][okj], Wj[invj][okj], PXj[invj][okj], r); ROW[(lab, "JOINT", r)] = dict(sdp=sdp, ecp=e); JOINT.append((lab, r, int(okj.sum())))
    p.check(f"{lab}: joint blanket SDP rows {int(okj.sum()):,} of {int(common.sum()):,} (rows whose joint query is undefined are left out of the joint SDP only)", okj.sum() >= common.sum() - 10)
# ------------------------------------------------------------------ bootstrap (RandomState(42), randint; the same resamples for every column; as the flip-rate analysis)
cols = {}
for (lab, v, r), a in ROW.items():
    if v == "JOINT": continue
    cols[(lab, v, r, "ecp")] = a["ecp"]; cols[(lab, v, r, "flip")] = a["flip"]; cols[(lab, v, r, "diff")] = a["ecp"] - a["flip"]; cols[(lab, v, r, "voi")] = a["voi"]
    cols[(lab, v, r, "voi_pos")] = (a["voi"] > 1e-12).astype(float); cols[(lab, v, r, "ecp_pos")] = (a["ecp"] > 1e-12).astype(float); cols[(lab, v, r, "ecp_pos_exact")] = (a["ecp"] > 0).astype(float)
    if r == 5: cols[(lab, v, 0, "sexp")] = a["sexp"]
keys = list(cols); lens = {len(c) for c in cols.values()}; assert lens == {29998}, lens
Mx = np.column_stack([cols[k] for k in keys]); bm = sens.boot_means([Mx], B=2000, seed=42)[0]; lo_, hi_ = np.percentile(bm, [2.5, 97.5], axis=0); mean_ = Mx.mean(axis=0); ST = {k: (mean_[i], lo_[i], hi_[i]) for i, k in enumerate(keys)}
for lab in ("Application", "After acceptance"):
    vs = sorted({v for (l, v, r) in ROW if l == lab and v != "JOINT"})
    for v in vs:
        for r in RS:
            g_ = lambda q: ST[(lab, v, r, q)]; sx = ST[(lab, v, 0, "sexp")]
            cells.append(dict(stage=lab, variable=v, r=r, mean_ecp=g_("ecp")[0], ecp_lo=g_("ecp")[1], ecp_hi=g_("ecp")[2], flip_rate=g_("flip")[0], flip_lo=g_("flip")[1], flip_hi=g_("flip")[2], ecp_minus_flip=g_("diff")[0], diff_lo=g_("diff")[1], diff_hi=g_("diff")[2],
                              share_voi_pos=g_("voi_pos")[0], voi_pos_lo=g_("voi_pos")[1], voi_pos_hi=g_("voi_pos")[2], share_ecp_pos=g_("ecp_pos")[0], ecp_pos_lo=g_("ecp_pos")[1], ecp_pos_hi=g_("ecp_pos")[2], share_ecp_pos_exact=g_("ecp_pos_exact")[0],
                              mean_sexp=sx[0], sexp_lo=sx[1], sexp_hi=sx[2], mean_voi=g_("voi")[0], voi_lo=g_("voi")[1], voi_hi=g_("voi")[2], abs_diff_ecp_flip_pts=100 * abs(g_("diff")[0])))
CE = pd.DataFrame(cells); CE.to_csv(ADD + "/j23_cells.csv", index=False)
# ---- gates against J10/J18
for _, w in CE.iterrows():
    s = FR[(FR.stage == w.stage) & (FR.variable == w.variable) & (FR.r == w.r)].iloc[0]
    p.gate(f"{w.stage} {w.variable} r={w.r}: flip rate {100*w.flip_rate:.2f}% [{100*w.flip_lo:.2f}, {100*w.flip_hi:.2f}] equals flip_rate_by_r.csv (1e-12)", abs(w.flip_rate - s.flip_rate) < 1e-12 and abs(w.flip_lo - s.lower) < 1e-12 and abs(w.flip_hi - s.upper) < 1e-12)
    q = J18[(J18.stage == w.stage) & (J18.variable == w.variable) & (J18.r == w.r)].iloc[0]
    p.gate(f"{w.stage} {w.variable} r={w.r}: mean VOI {w.mean_voi:.6f} and share VOI > 0 {100*w.share_voi_pos:.2f}% equal J18 ({q.mean_voi:.6f}, {100*q.share_voi_pos:.2f}%); mean S_exp {w.mean_sexp:.4f}", abs(w.mean_voi - q.mean_voi) < 1e-9 and abs(w.share_voi_pos - q.share_voi_pos) < 1e-9 and abs(w.mean_sexp - q.mean_sexp) < 1e-9)
md = float(np.abs(CE.share_ecp_pos - CE.share_voi_pos).max()); mdx = float(np.abs(CE.share_ecp_pos_exact - CE.share_voi_pos).max())
p.check(f"share of rows with ECP > 0 equals the share with VOI > 0 up to ties: maximum discrepancy {md:.2e} (threshold 1e-12 on both), {mdx:.2e} with exact positivity of ECP", md < 1e-3)
# ---- rankings and row-level
for lab in ("Application", "After acceptance"):
    for r in RS:
        d = CE[(CE.stage == lab) & (CE.r == r)].set_index("variable"); vs = list(d.index)
        mets = {"S_exp": d.mean_sexp, "ECP": d.mean_ecp, "VOI": d.mean_voi, "flip rate": d.flip_rate}
        order = {m: tuple(s.round(12).sort_values(ascending=False, kind="stable").index) if s.round(12).nunique() == len(s) else "tied" for m, s in mets.items()}
        for a_, b_ in (("S_exp", "ECP"), ("S_exp", "VOI"), ("S_exp", "flip rate"), ("ECP", "VOI"), ("ECP", "flip rate"), ("VOI", "flip rate")):
            if len(vs) >= 3: sr = float(spearmanr(mets[a_], mets[b_]).statistic); kt = float(kendalltau(mets[a_], mets[b_]).statistic)
            else:
                sa, sb = [np.sign(np.round(mets[m_].iloc[0] - mets[m_].iloc[1], 12)) for m_ in (a_, b_)]; sr = kt = (np.nan if (sa == 0 or sb == 0) else (1.0 if sa == sb else -1.0))
            RANK.append(dict(stage=lab, r=r, metric_a=a_, metric_b=b_, n_variables=len(vs), spearman=sr, kendall_tau=kt, order_a=str(order[a_]), order_b=str(order[b_]), same_order=bool(order[a_] == order[b_] and order[a_] != "tied")))
RKd = pd.DataFrame(RANK); RKd.to_csv(ADD + "/j23_rank_agreement.csv", index=False)
rl = []
for (lab, v, r), a in ROW.items():
    if v == "JOINT": continue
    rl.append(dict(stage=lab, variable=v, r=r, n_rows=len(a["ecp"]), pearson_ecp_voi=float(pearsonr(a["ecp"], a["voi"]).statistic) if np.std(a["voi"]) > 0 and np.std(a["ecp"]) > 0 else np.nan,
                   pearson_ecp_sexp=float(pearsonr(a["ecp"], a["sexp"]).statistic) if np.std(a["ecp"]) > 0 and np.std(a["sexp"]) > 0 else np.nan, pearson_voi_sexp=float(pearsonr(a["voi"], a["sexp"]).statistic) if np.std(a["voi"]) > 0 else np.nan,
                   pearson_ecp_flip=float(pearsonr(a["ecp"], a["flip"]).statistic) if np.std(a["flip"]) > 0 and np.std(a["ecp"]) > 0 else np.nan))
RL = pd.DataFrame(rl); RL.to_csv(ADD + "/j23_row_level.csv", index=False)
jr = []
for (lab, v, r), a in ROW.items():
    if v != "JOINT": continue
    bj = sens.boot_means([a["sdp"][:, None]], B=2000, seed=42)[0][:, 0]; lo, hi = np.percentile(bj, [2.5, 97.5]); jr.append(dict(stage=lab, r=r, n_rows=len(a["sdp"]), mean_joint_sdp=float(a["sdp"].mean()), lo=float(lo), hi=float(hi), mean_joint_ecp=1 - float(a["sdp"].mean())))
JR = pd.DataFrame(jr); JR.to_csv(ADD + "/j23_joint_sdp.csv", index=False)
# ---- criterion (descriptive)
n_agree = int((CE.abs_diff_ecp_flip_pts <= 2.0).sum()); n_cells = len(CE)
diff_cells = [(l, r) for (l, r), g in RKd[(RKd.metric_a == "S_exp") & (RKd.metric_b == "ECP")].groupby(["stage", "r"]) for _ in [0] if not g.same_order.iloc[0]]
crit = dict(n_agree=n_agree, n_cells=n_cells, agreement=bool(n_agree >= 16), rank_differs_cells=[(l, int(r)) for (l, r) in diff_cells], rank_differs=len(diff_cells) >= 1, max_disc=md, max_disc_exact=mdx)
LOG(str(crit)); LOG(f"J23 done {(time.time()-t_all)/60:.1f} min")
print(CE[["stage", "variable", "r", "mean_ecp", "flip_rate", "ecp_minus_flip", "share_voi_pos", "share_ecp_pos", "mean_sexp", "mean_voi"]].round(4).to_string()); print(JR.round(4)); print(RKd[(RKd.metric_a == "S_exp") & (RKd.metric_b == "ECP")].round(2).to_string())
pickle.dump(dict(CE=CE, RK=RKd, RL=RL, JR=JR, crit=crit, checks=p.checks), open(ADD + "/j23/j23_summary.pkl", "wb")) if os.makedirs(ADD + "/j23", exist_ok=True) is None else None
