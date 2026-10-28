"""J18: myopic value of information (VOI) of each blanket variable against evidence sensitivity S_exp and flip rate. BN only, main split, both stages.
Cost of a wrong flag 1, of a miss r.  For a test row with evidence e (X_j observed), e_-j = e without X_j:
  p0 = P(Y=1|e_-j);  p_x = P(Y=1|e_-j, X_j=x);  w_x = P(X_j=x|e_-j)
  cost_before = min(r*p0, 1-p0);  cost_after = sum_x w_x*min(r*p_x, 1-p_x);  VOI = cost_before - cost_after >= 0.
Flag from e: 1[p_{x_obs} >= 1/(1+r)]; flag from e_-j: 1[p0 >= 1/(1+r)]; flip = the two differ (the J10 definition).
Realised value (descriptive): realised cost with the e_-j flag minus realised cost with the e flag, true labels, per row (positive = observing X_j helped on the test set).
Usage: python j18.py   (needs j17/application_fit.*, j17/after_full_fit.*; writes j18_*.csv, parts/j18.{json,md}, j18_snumbers_rows.csv)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from addcommon import Part, checks_md, sgn
from pgmpy.inference import VariableElimination
from scipy.stats import spearmanr, kendalltau, pearsonr
p = Part("j18"); J17 = ADD + "/j17"; RS = (5, 10, 15, 20); short = lambda s: s.replace(" (bin)", "")

# ------------------------------------------------------------------ the VOI function and the toy case verified by hand
def voi_rows(p0, px, wx, r):
    """p0: (n,), px, wx: (n, K) (w = 0 where a state has no mass; px then ignored). Returns cost_before, cost_after, VOI."""
    cb = np.minimum(r * p0, 1 - p0)
    ca = (np.where(wx > 0, wx * np.minimum(r * px, 1 - px), 0.0)).sum(axis=1)
    return cb, ca, cb - ca
def query_variable(model, var, ev_rest, target="Default", eng=None):
    """p0, w (dict state->prob), px (dict state->P(target=1|rest, var=state)) for one reduced evidence set"""
    eng = VariableElimination(model) if eng is None else eng; f = eng.query(variables=[var], evidence=ev_rest, show_progress=False)
    w = dict(zip(f.state_names[var], np.asarray(f.values, float)))
    q0 = eng.query(variables=[target], evidence=ev_rest, show_progress=False).get_value(**{target: 1})
    def _pq(s):
        try:
            with np.errstate(invalid="ignore", divide="ignore"): return eng.query(variables=[target], evidence={**ev_rest, var: s}, show_progress=False).get_value(**{target: 1})
        except Exception: return np.nan
    px = {s: _pq(s) for s in w}          # every state, including those with w = 0 (the observed value can have w_x = 0 numerically; its posterior is still the stored one)
    return float(q0), w, px
from pgmpy.factors.discrete import TabularCPD
toy = DiscreteBayesianNetwork([("X", "Default")])
toy.add_cpds(TabularCPD("X", 2, [[0.6], [0.4]], state_names={"X": ["a", "b"]}),
             TabularCPD("Default", 2, [[0.95, 0.70], [0.05, 0.30]], evidence=["X"], evidence_card=[2], state_names={"Default": [0, 1], "X": ["a", "b"]}))
assert toy.check_model()
q0, w, px = query_variable(toy, "X", {})
hand = {5: dict(p0=0.15, cb=0.75, ca=0.6 * 0.25 + 0.4 * 0.70, voi=0.32), 10: dict(p0=0.15, cb=0.85, ca=0.6 * 0.5 + 0.4 * 0.70, voi=0.27)}
p.gate("toy case (X ~ (0.6, 0.4); P(Y=1|X) = (0.05, 0.30)): p0 = 0.6*0.05 + 0.4*0.30 = 0.15 equals the network's P(Y=1)", abs(q0 - 0.15) < 1e-12)
p.gate("toy case: sum_x w_x p_x = p0 (law of total probability)", abs(sum(w[s] * px[s] for s in w) - q0) < 1e-12)
for r in (5, 10):
    cb, ca, vo = voi_rows(np.array([q0]), np.array([[px["a"], px["b"]]]), np.array([[w["a"], w["b"]]]), r)
    p.gate(f"toy case r={r}: cost_before {cb[0]:.4f} = {hand[r]['cb']:.4f}, cost_after {ca[0]:.4f} = {hand[r]['ca']:.4f}, VOI {vo[0]:.4f} = {hand[r]['voi']:.4f} (by hand: r=5: 0.75 - (0.6*0.25 + 0.4*0.70) = 0.32; r=10: 0.85 - (0.6*0.5 + 0.4*0.70) = 0.27)",
           abs(cb[0] - hand[r]["cb"]) < 1e-12 and abs(ca[0] - hand[r]["ca"]) < 1e-12 and abs(vo[0] - hand[r]["voi"]) < 1e-12)

# ------------------------------------------------------------------ per-stage computation on the real BNs
def stage_rows(name, label):
    fit, info, sres = load_fit(name, J17); ctx = variant_ctx(fit, None, None, name, base="skip"); bn, S, ev_cols, codes = ctx["bn"], ctx["S"], ctx["ev_cols"], ctx["codes"]
    mbv = [b for b in fit["blanket"] if b not in ("M_Income", "M_Dependents")]; y = ctx["y"]; base_p = fit["pbn"]; valid = fit["valid"]; out = {}
    t0 = time.time(); eng = VariableElimination(bn)
    for v in mbv:
        j = ev_cols.index(v); K = len(S[v]); c2 = codes.copy(); obs = c2[:, j].copy(); c2[:, j] = -1
        u, inv = np.unique(c2, axis=0, return_inverse=True); inv = np.asarray(inv).ravel()
        P0 = np.full(len(u), np.nan); PX = np.full((len(u), K), np.nan); WX = np.zeros((len(u), K))
        for k, row in enumerate(u):
            ev = {c: S[c][i] for c, i in zip(ev_cols, row) if i >= 0}
            try:
                with np.errstate(invalid="ignore", divide="ignore"):
                    q0, w, px = query_variable(bn, v, ev, eng=eng)
                P0[k] = q0
                for s, ws in w.items(): WX[k, S[v].index(s)] = ws if np.isfinite(ws) else 0.0
                for s, pp in px.items(): PX[k, S[v].index(s)] = pp
            except Exception:
                P0[k] = np.nan
        p0 = P0[inv]; px_ = PX[inv]; wx = WX[inv]
        bad = ~np.isfinite(p0) | (obs < 0) | ((wx > 0) & ~np.isfinite(px_)).any(axis=1) | ~valid
        pxo = px_[np.arange(len(obs)), np.maximum(obs, 0)]; ok = ~bad & np.isfinite(pxo)
        out[v] = dict(p0=p0, px=np.where(wx > 0, px_, 0.0), wx=wx, obs=obs, ok=ok, pbase=pxo, n_patterns=len(u))
        print(f"[{name}] {short(v)}: {len(u)} reduced evidence patterns, rows used {int(ok.sum())} ({(time.time()-t0):.0f}s)", flush=True)
    common = np.all([out[v]['ok'] for v in mbv], axis=0)         # the J10 row set: rows where every blanket variable's retraction is defined (29,998)
    for v in mbv: out[v]['ok'] = common
    return fit, info, sres, mbv, y, out
RES = {}
for name, label in (("application", "Application"), ("after_full", "After acceptance")): RES[label] = stage_rows(name, label)

# ------------------------------------------------------------------ gates: rows, flip rates, S_exp, total probability
FR = pd.read_csv(ADD + "/flip_rate_by_r.csv"); J = {}
for label, (fit, info, sres, mbv, y, out) in RES.items():
    sx = sres.groupby("variable").S_exp.mean()
    for v in mbv:
        o = out[v]; ok = o["ok"]
        # total probability: sum_x w_x p_x = p0
        tp = np.abs((o["wx"] * o["px"]).sum(axis=1) - o["p0"])[ok].max()
        p.gate(f"{label} {short(v)}: law of total probability sum_x w_x p_x = p0 on {int(ok.sum()):,} rows (max abs deviation {tp:.2e})", tp < 1e-9)
        p.gate(f"{label} {short(v)}: P(Y=1|e) recomputed from the same VE queries equals the stored BN posterior (max abs diff {np.abs(o['pbase'][ok] - fit['pbn'][ok]).max():.2e})", np.abs(o["pbase"][ok] - fit["pbn"][ok]).max() < 1e-9)
        # S_exp recomputed from the same quantities
        sexp = (o["wx"] * np.abs(o["px"] - o["pbase"][:, None]) * (o["wx"] > 0)).copy(); sexp[np.arange(len(ok)), np.maximum(o["obs"], 0)] = 0.0; sexp = sexp.sum(axis=1)
        srow = sres[sres.variable == v].set_index("row_index").S_exp; rows_ = fit["test_rows"]; ref = pd.Series(sexp, index=rows_)[ok]
        dS = np.abs(ref.reindex(srow.index).to_numpy() - srow.to_numpy()); dS = dS[np.isfinite(dS)]
        p.gate(f"{label} {short(v)}: S_exp from the VOI quantities equals analyze_blanket S_exp row by row (max abs diff {dS.max():.2e} on {len(dS):,} rows)", dS.max() < 1e-9)
        for r in RS:
            tau = 1 / (1 + r); flip = ((o["pbase"] >= tau) != (o["p0"] >= tau))[ok].mean()
            s = FR[(FR.stage == label) & (FR.variable == short(v)) & (FR.r == r)].iloc[0]
            p.gate(f"{label} {short(v)} r={r}: flip rate {100*flip:.2f}% on {int(ok.sum()):,} rows equals flip_rate_by_r.csv ({100*s.flip_rate:.2f}%, {int(s.n_rows):,} rows)", abs(flip - s.flip_rate) < 1e-12 and int(ok.sum()) == int(s.n_rows))

# ------------------------------------------------------------------ the numbers: per variable and r
rows = []; ROWLEVEL = {}; boot_in = {}
for label, (fit, info, sres, mbv, y, out) in RES.items():
    ok0 = out[mbv[0]]["ok"]
    for v in mbv: assert (out[v]["ok"] == ok0).all()
    yy = y[ok0].astype(float); pi = yy.mean()
    for v in mbv:
        o = out[v]; ok = ok0; p0, px, wx, pb = o["p0"][ok], o["px"][ok], o["wx"][ok], o["pbase"][ok]; obsx = o["obs"][ok]
        sexp = (wx * np.abs(px - pb[:, None])); sexp[np.arange(len(pb)), obsx] = 0.0; sexp = sexp.sum(axis=1)
        for r in RS:
            tau = 1 / (1 + r); cb, ca, vo = voi_rows(p0, px, wx, r); assert vo.min() > -1e-12, vo.min(); vo = np.maximum(vo, 0.0)      # float noise only
            fl_e, fl_m = pb >= tau, p0 >= tau; flip = (fl_e != fl_m)
            real = (r * yy * (~fl_m) + (1 - yy) * fl_m) - (r * yy * (~fl_e) + (1 - yy) * fl_e)       # cost with e_-j flag minus cost with e flag
            # does some alternative value change the flag?
            alt = ((np.where(wx > 0, px, np.nan) >= tau) != fl_m[:, None])
            alt = np.nan_to_num(alt.astype(float)).max(axis=1) > 0
            wobs = wx[np.arange(len(pb)), obsx]; ROWLEVEL[(label, short(v), r)] = dict(wobs=wobs, sexp=sexp, voi=vo, flip=flip.astype(float), real=real, alt=alt, cb=cb, ca=ca)
            for nm, arr in (("sexp", sexp), ("voi", vo), ("voi_pos", (vo > 1e-12).astype(float)), ("flip", flip.astype(float)), ("real", real), ("cost_before", cb)):
                boot_in[(label, short(v), r, nm)] = arr
# one bootstrap with shared resamples (RandomState(42), randint, B = 2000) for every column
keys = list(boot_in); M = np.column_stack([boot_in[k] for k in keys]); n_rows = {lab: int(len(boot_in[k])) for k in keys for lab in [k[0]]}
# the two stages have different row counts only if the row sets differ; here both are 29,998 -> one matrix
assert len({len(v) for v in boot_in.values()}) == 1
bm = sens.boot_means([M], B=2000, seed=42)[0]; lo_, hi_ = np.percentile(bm, [2.5, 97.5], axis=0); mean_ = M.mean(axis=0)
STAT = {k: (mean_[i], lo_[i], hi_[i]) for i, k in enumerate(keys)}
res_rows = []
for label, (fit, info, sres, mbv, y, out) in RES.items():
    ok0 = out[mbv[0]]["ok"]; yy = y[ok0].astype(float); pi = float(yy.mean())
    for v in mbv:
        for r in RS:
            g_ = lambda nm: STAT[(label, short(v), r, nm)]
            triv = min(r * pi, 1 - pi)
            res_rows.append(dict(stage=label, variable=short(v), r=r, n_rows=int(ok0.sum()), mean_sexp=g_("sexp")[0], mean_sexp_lo=g_("sexp")[1], mean_sexp_hi=g_("sexp")[2],
                                 flip_rate=g_("flip")[0], flip_lo=g_("flip")[1], flip_hi=g_("flip")[2],
                                 mean_voi=g_("voi")[0], voi_lo=g_("voi")[1], voi_hi=g_("voi")[2], mean_cost_before=g_("cost_before")[0],
                                 voi_share_of_cheaper_trivial=g_("voi")[0] / triv, voi_share_lo=g_("voi")[1] / triv, voi_share_hi=g_("voi")[2] / triv, trivial_cost_per_row=triv, default_rate_rows=pi,
                                 voi_share_of_cost_before=g_("voi")[0] / g_("cost_before")[0],
                                 share_voi_pos=g_("voi_pos")[0], share_voi_pos_lo=g_("voi_pos")[1], share_voi_pos_hi=g_("voi_pos")[2],
                                 realised_value=g_("real")[0], realised_lo=g_("real")[1], realised_hi=g_("real")[2]))
RESdf = pd.DataFrame(res_rows); RESdf.to_csv(ADD + "/j18_voi_by_variable_r.csv", index=False)

# ------------------------------------------------------------------ rankings
rk = []
for label in RES:
    for r in RS:
        d = RESdf[(RESdf.stage == label) & (RESdf.r == r)].set_index("variable"); vs = list(d.index)
        mets = {"S_exp": d.mean_sexp, "flip rate": d.flip_rate, "VOI": d.mean_voi, "realised value": d.realised_value}
        rankv = {m: s.rank(ascending=False, method="min").astype(int).to_dict() for m, s in mets.items()}
        for a, b in (("S_exp", "flip rate"), ("S_exp", "VOI"), ("flip rate", "VOI"), ("S_exp", "realised value"), ("flip rate", "realised value"), ("VOI", "realised value")):
            if len(vs) >= 3: sp_ = spearmanr(mets[a], mets[b]); kt = kendalltau(mets[a], mets[b]); sr, kr = float(sp_.statistic), float(kt.statistic)
            else:
                sa, sb = [np.sign(np.round(mets[m_].iloc[0] - mets[m_].iloc[1], 12)) for m_ in (a, b)]       # two variables: agreement is +1 / -1; undefined (NaN) if either metric is tied
                sr = kr = (np.nan if (sa == 0 or sb == 0) else (1.0 if sa == sb else -1.0))
            rk.append(dict(stage=label, r=r, metric_a=a, metric_b=b, n_variables=len(vs), spearman=sr, kendall_tau=kr, ranks_a=str(rankv[a]), ranks_b=str(rankv[b])))
RK = pd.DataFrame(rk); RK.to_csv(ADD + "/j18_rank_agreement.csv", index=False)
dis = []
for label in RES:
    for r in RS:
        d = RESdf[(RESdf.stage == label) & (RESdf.r == r)].set_index("variable"); vs = list(d.index)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                a, b = vs[i], vs[j]; sgn_ = {m: np.sign(np.round(d.loc[a, c] - d.loc[b, c], 12)) for m, c in (("S_exp", "mean_sexp"), ("flip rate", "flip_rate"), ("VOI", "mean_voi"), ("realised value", "realised_value"))}
                if len(set(sgn_.values())) > 1:
                    hi_ = lambda m: "tie" if sgn_[m] == 0 else (a if sgn_[m] > 0 else b)
                    dis.append(dict(stage=label, r=r, pair=f"{a} vs {b}", higher_by_S_exp=hi_("S_exp"), higher_by_flip=hi_("flip rate"), higher_by_VOI=hi_("VOI"), higher_by_realised=hi_("realised value")))
DIS = pd.DataFrame(dis); DIS.to_csv(ADD + "/j18_ranking_disagreements.csv", index=False)

# ------------------------------------------------------------------ row level
rl = []
for (label, v, r), a in ROWLEVEL.items():
    n_ = len(a["voi"]); pos = a["voi"] > 1e-12; fl = a["flip"] > 0
    pr = pearsonr(a["sexp"], a["voi"]).statistic if np.std(a["voi"]) > 0 else np.nan; sr_ = spearmanr(a["sexp"], a["voi"]).statistic if np.std(a["voi"]) > 0 else np.nan
    rl.append(dict(stage=label, variable=v, r=r, n_rows=n_, pearson_sexp_voi=float(pr), spearman_sexp_voi=float(sr_), share_voi_pos=float(pos.mean()), share_flip=float(fl.mean()),
                   share_voi_pos_no_flip=float((pos & ~fl).mean()), share_flip_voi_zero=float((fl & ~pos).mean()), share_alt_value_changes_flag=float(a["alt"].mean()), share_alt_but_no_voi=float((a["alt"] & ~pos).mean()),
                   share_voi_pos_but_not_alt=float((pos & ~a["alt"]).mean()), mean_voi_given_flip=float(a["voi"][fl].mean()) if fl.any() else np.nan, mean_voi_given_no_flip_pos=float(a["voi"][pos & ~fl].mean()) if (pos & ~fl).any() else np.nan))
RL = pd.DataFrame(rl); RL.to_csv(ADD + "/j18_row_level_relations.csv", index=False)
nz = 0; wmax = 0.0
for k_, a in ROWLEVEL.items():
    z = (a["flip"] > 0) & ~(a["voi"] > 1e-12); nz += int(z.sum()); wmax = max(wmax, float(a["wobs"][z].max()) if z.any() else 0.0)
p.check(f"every row with flip = 1 has VOI > 0, except {nz} row-variable-r combinations in which the observed value has (numerically) zero predictive probability w_obs given the other evidence (largest w_obs among them {wmax:.1e}); VOI is an expectation under w and gives such a value no weight", wmax < 1e-9)
print(RESdf.round(5).to_string()); print(RK.round(3).to_string()); print(DIS.to_string()); print(RL.round(4).to_string())
pickle.dump(dict(RES=RESdf, RK=RK, DIS=DIS, RL=RL, checks=p.checks), open(ADD + "/j18/j18_summary.pkl", "wb"))
p.write("(report text written by j18_report.py)")
