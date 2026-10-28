"""J24 Stage B (ii), Application only: the J21 all-subsets comparison (4 subsets of {Age, Debt Ratio}) for the structures NB and TAN (and BIC as a gate against J21), against LR_B, RF_B, LR_IMP, RF_IMP (the saved J21 comparator scores), with gap_S - gap_full and its paired interval.
Writes j24_subset_results.csv, j24_gap_vs_full.csv, j24/j24_subsets_summary.pkl"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from j_tables import ev_dict
from addcommon import Part
from pgmpy.inference import VariableElimination
p = Part("j24s"); RS = (5, 10); LOG = lambda s: print(s, flush=True); sh = lambda s: s.replace(" (bin)", "")
fit, _i, _s = load_fit("application", ADD + "/j17"); D = {int(f[-6:-4]): pickle.load(open(f"{ADD}/j21/{f}", "rb")) for f in sorted(os.listdir(ADD + "/j21")) if f.startswith("application_subset") and f.endswith(".pkl")}; full = max(D)
per = pickle.load(open(R + "/selection/predictions/sel_rq2_per_row.pkl", "rb"))["per_row"]; ga = per[per.variable == "Age (bin)"].set_index("row_index"); gd = per[per.variable == "Debt Ratio (bin)"].set_index("row_index")
keep = ga.index[ga.retracted_prob.notna().values & gd.retracted_prob.reindex(ga.index).notna().values]; common = np.isin(fit["test_rows"], keep); assert common.sum() == 29998
y_all = D[full]["y"]; ys = y_all[common]; ns = len(ys); Wc = np.random.default_rng(42).multinomial(ns, np.full(ns, 1.0 / ns), size=B).astype(np.int16); ones = np.ones(ns); Dw = Wc.astype(np.float32) @ ys.astype(np.float32)
K = ["Age", "Debt Ratio"]; STR = {}
for sname in ("NB", "TAN"):
    d = pickle.load(open(f"{ADD}/j24/application_{sname}.pkl", "rb")); bn_ = DiscreteBayesianNetwork([tuple(e) for e in d["model_edges"]]); bn_.add_cpds(*d["cpds"]); assert bn_.check_model()
    Fs = rq3._states_frame(fit["fte"], d["ev_cols"], d["S"]); cod = np.column_stack([pd.Index(d["S"][c]).get_indexer(Fs[c]) for c in d["ev_cols"]]); STR[sname] = dict(S=d["S"], ev_cols=d["ev_cols"], codes=cod, eng=VariableElimination(bn_), bn=bn_)
def post(sname, U):
    st = STR[sname]; c2 = st["codes"].copy(); c2[:, [st["ev_cols"].index(u + " (bin)") for u in U]] = -1; u_, inv = np.unique(c2, axis=0, return_inverse=True); inv = np.asarray(inv).ravel(); pu = np.full(len(u_), np.nan)
    for k, row in enumerate(u_):
        try:
            with np.errstate(invalid="ignore", divide="ignore"): pu[k] = float(st["eng"].query(variables=["Default"], evidence=ev_dict(st["S"], st["ev_cols"], row), show_progress=False).get_value(Default=1))
        except Exception: pass
    return pu[inv]
SC = {}
for si, dd in D.items():
    SC[("BIC", si)] = dd["pbn"]
    for sname in ("NB", "TAN"): SC[(sname, si)] = dd["pbn"].copy() if si == full and False else post(sname, dd["unobserved"]); assert np.isfinite(SC[(sname, si)][common]).all()
LOG("structure posteriors per subset done")
comp = lambda dd, k: {"LR_B": dd["pB"]["LR"], "RF_B": dd["pB"]["RF"], "LR_IMP": dd["pI"]["LR"], "RF_IMP": dd["pI"]["RF"]}[k]
tauc = lambda dd, k, r: {"LR_B": dd["tauB"]["LR"][r], "RF_B": dd["tauB"]["RF"][r], "LR_IMP": dd["tauI"]["LR"][r], "RF_IMP": dd["tauI"]["RF"][r]}[k]
CN = ["LR_B", "RF_B", "LR_IMP", "RF_IMP"]; AB = {}; PT = {}
for si, dd in D.items():
    for k in CN: s = comp(dd, k)[common]; AB[(k, si)] = boot_auc(s, ys, ones, W=Wc); PT[(k, si)] = point_auc(s, ys, ones)
    for sname in ("BIC", "NB", "TAN"): s = SC[(sname, si)][common]; AB[(sname, si)] = boot_auc(s, ys, ones, W=Wc); PT[(sname, si)] = point_auc(s, ys, ones)
rows = []; gaps = []
for sname in ("BIC", "NB", "TAN"):
    gfull = {m: (PT[(f"{m}_B", full)] - PT[(sname, full)], AB[(f"{m}_B", full)] - AB[(sname, full)]) for m in ("LR", "RF")}
    for si, dd in sorted(D.items()):
        row = dict(structure=sname, subset=si, observed=", ".join(dd["observed"]) or "none", n_observed=len(dd["observed"]), auc_S=PT[(sname, si)])
        for k in CN:
            row[f"auc_{k}"] = PT[(k, si)]; df_ = AB[(sname, si)] - AB[(k, si)]; lo, hi = np.percentile(df_, [2.5, 97.5]); row[f"aucdiff_S_minus_{k}"] = PT[(sname, si)] - PT[(k, si)]; row[f"aucdiff_S_minus_{k}_lo"] = float(lo); row[f"aucdiff_S_minus_{k}_hi"] = float(hi)
        for r in RS:
            trv = min(r * ys.sum(), ns - ys.sum())
            def sv(pv, tau): fl = pv[common] >= tau; c = np.where(ys == 1, r * (~fl), fl).astype(float); return 100 * (1 - c.sum() / trv), 100 * (1 - (Wc.astype(np.float32) @ c.astype(np.float32)) / np.minimum(r * Dw, ns - Dw))
            sp, sb = sv(SC[(sname, si)], 1 / (1 + r)); row[f"saving_S_r{r}"] = sp; row[f"saving_S_r{r}_lo"], row[f"saving_S_r{r}_hi"] = [float(x) for x in np.percentile(sb, [2.5, 97.5])]
            for k in CN:
                cp, cb = sv(comp(dd, k), tauc(dd, k, r)); lo, hi = np.percentile(sb - cb, [2.5, 97.5]); row[f"savdiff_S_minus_{k}_r{r}"] = sp - cp; row[f"savdiff_S_minus_{k}_r{r}_lo"] = float(lo); row[f"savdiff_S_minus_{k}_r{r}_hi"] = float(hi)
        rows.append(row)
        for m in ("LR", "RF"):
            gS = PT[(f"{m}_B", si)] - PT[(sname, si)]; gd_ = (AB[(f"{m}_B", si)] - AB[(sname, si)]) - gfull[m][1]; lo, hi = np.percentile(gd_, [2.5, 97.5])
            gaps.append(dict(structure=sname, subset=si, observed=row["observed"], model=m, gap_S=gS, gap_full=gfull[m][0], gap_S_minus_gap_full=gS - gfull[m][0], lo=float(lo), hi=float(hi), sig_positive=bool(lo > 0), sig_negative=bool(hi < 0)))
RW = pd.DataFrame(rows); GP = pd.DataFrame(gaps); RW.to_csv(ADD + "/j24_subset_results.csv", index=False); GP.to_csv(ADD + "/j24_gap_vs_full.csv", index=False)
# gate: the BIC rows equal the J21 Application rows
J21 = pd.read_csv(ADD + "/j21_subset_results.csv"); J21 = J21[J21.stage == "application"].set_index("subset"); wmax = 0.0
for _, w in RW[RW.structure == "BIC"].iterrows():
    q = J21.loc[w.subset]
    for k in CN: wmax = max(wmax, abs(w[f"aucdiff_S_minus_{k}"] - q[f"aucdiff_BN_minus_{k}"]), abs(w[f"aucdiff_S_minus_{k}_lo"] - q[f"aucdiff_BN_minus_{k}_lo"]), abs(w[f"savdiff_S_minus_{k}_r10"] - q[f"savdiff_BN_minus_{k}_r10"]))
p.gate(f"BIC rows through the J24 code equal the J21 Application rows (AUC and saving differences, bounds; max abs diff {wmax:.2e}, tol 1e-9)", wmax < 1e-9)
crit = {}
for sname in ("NB", "TAN"):
    x = GP[(GP.structure == sname) & (GP.subset != full)]; r_ = RW[(RW.structure == sname) & (RW.subset != full)]
    crit[sname] = dict(**{f"sig_positive_{m}_B": int(x[x.model == m].sig_positive.sum()) for m in ("LR", "RF")}, **{f"not_sig_positive_{m}_B": int((~x[x.model == m].sig_positive).sum()) for m in ("LR", "RF")},
                       **{f"above_{k}": int((r_[f"aucdiff_S_minus_{k}_lo"] > 0).sum()) for k in ("LR_IMP", "RF_IMP")}, **{f"below_{k}": int((r_[f"aucdiff_S_minus_{k}_hi"] < 0).sum()) for k in ("LR_IMP", "RF_IMP")}, n_nonfull=len(r_))
print(RW[["structure", "subset", "observed", "auc_S", "auc_LR_B", "auc_RF_B", "auc_LR_IMP", "auc_RF_IMP"]].round(4).to_string()); print(GP.round(4).to_string()); print(crit)
pickle.dump(dict(crit=crit, checks=p.checks), open(ADD + "/j24/j24_subsets_summary.pkl", "wb"))
