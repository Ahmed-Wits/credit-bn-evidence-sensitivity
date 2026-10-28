"""J21 aggregation: AUC/saving per subset with paired bootstrap, gap_S - gap_full, criterion. Reads j21/STAGE_subsetNN.pkl. Writes j21_subset_results.csv, j21_gap_vs_full.csv, j21/j21_summary.pkl"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from addcommon import Part
p = Part("j21"); RS = (5, 10); rows = []; gaps = []; SUMM = {}
for stage in ("application", "after"):
    files = sorted(f for f in os.listdir(ADD + "/j21") if f.startswith(stage + "_subset") and f.endswith(".pkl"))
    if not files: continue
    D = {int(f[-6:-4]): pickle.load(open(f"{ADD}/j21/{f}", "rb")) for f in files}; nK = max(D) .bit_length(); full = max(D)
    fit, info, _ = load_fit({"application": "application", "after": "after_full"}[stage], ADD + "/j17"); y_all = D[full]["y"]
    if stage == "after": common = fit["valid"].copy()
    else:
        per = pickle.load(open(R + "/selection/predictions/sel_rq2_per_row.pkl", "rb"))["per_row"]
        ga = per[per.variable == "Age (bin)"].set_index("row_index"); gd = per[per.variable == "Debt Ratio (bin)"].set_index("row_index")
        keep = ga.index[ga.retracted_prob.notna().values & gd.retracted_prob.reindex(ga.index).notna().values]; common = np.isin(fit["test_rows"], keep)
    assert common.sum() == 29998 and len(D) == 2 ** nK, (common.sum(), len(D))
    ys = y_all[common]; ns = len(ys); Wc = np.random.default_rng(42).multinomial(ns, np.full(ns, 1.0 / ns), size=B).astype(np.int16); ones = np.ones(ns)
    Dw = Wc.astype(np.float32) @ ys.astype(np.float32)
    # ---- gates at S = all observed (full evidence)
    d = D[full]; ref = {"after": (0.8519, 0.8618, 0.8603), "application": (0.6627, 0.6904, 0.6926)}[stage]; v_ = fit["valid"]
    a_ = (roc_auc_score(y_all[v_], d["pbn"][v_]), roc_auc_score(y_all, d["pB"]["LR"]), roc_auc_score(y_all, d["pB"]["RF"]))
    p.gate(f"{stage}: full-evidence AUC BN/LR/RF {[round(x,4) for x in a_]} = {ref} (the retrained full models equal the article's)", tuple(round(x, 4) for x in a_) == ref)
    dl = np.abs(d["pB"]["LR"] - d["pI"]["LR"]).max(); dr = np.abs(d["pB"]["RF"] - d["pI"]["RF"]).max()
    p.gate(f"{stage}: retrained full LR/RF equal the stored article models' predictions (max abs diff {dl:.1e} / {dr:.1e}, tol 1e-6)", dl < 1e-6 and dr < 1e-6)
    ref7 = {"after": {5: (35.2, 35.0, 35.1), 10: (47.7, 49.0, 49.1)}, "application": {5: (0.0, 0.0, 1.1), 10: (9.9, 16.1, 15.7)}}[stage]
    rowsT = v_ if stage == "after" else np.ones(len(y_all), bool)
    for r in RS:
        yy = y_all[rowsT]; tr_ = dv.trivial(yy, r)["cheaper"]
        got = tuple(round(100 * dv.saving(dv.cost(yy, p_[rowsT] >= t_, r), tr_), 1) for p_, t_ in ((d["pbn"], 1 / (1 + r)), (d["pI"]["LR"], d["tauI"]["LR"][r]), (d["pI"]["RF"], d["tauI"]["RF"][r])))
        p.gate(f"{stage}: Table 7 savings at r={r} BN/LR/RF {got} = {ref7[r]}", got == ref7[r])
    # ---- per subset
    def sc(dd, k):
        return {"BN": dd["pbn"], "LR_B": dd["pB"]["LR"], "RF_B": dd["pB"]["RF"], "LR_IMP": dd["pI"]["LR"], "RF_IMP": dd["pI"]["RF"]}[k]
    def tauof(dd, k, r):
        return {"BN": 1 / (1 + r), "LR_B": dd["tauB"]["LR"][r], "RF_B": dd["tauB"]["RF"][r], "LR_IMP": dd["tauI"]["LR"][r], "RF_IMP": dd["tauI"]["RF"][r]}[k]
    NAMES = ["BN", "LR_B", "RF_B", "LR_IMP", "RF_IMP"]; AB = {}; PT = {}
    for si, dd in sorted(D.items()):
        for k in NAMES:
            s = sc(dd, k)[common]; AB[(si, k)] = boot_auc(s, ys, ones, W=Wc); PT[(si, k)] = point_auc(s, ys, ones)
    gfull = {m: (PT[(full, f"{m}_B")] - PT[(full, "BN")], AB[(full, f"{m}_B")] - AB[(full, "BN")]) for m in ("LR", "RF")}
    for si, dd in sorted(D.items()):
        row = dict(stage=stage, subset=si, n_observed=len(dd["observed"]), observed=", ".join(dd["observed"]) or "none", unobserved=", ".join(dd["unobserved"]) or "none", n_rows=ns, lr_solver_B=dd["solver"])
        for k in NAMES: row[f"auc_{k}"] = PT[(si, k)]
        for k in NAMES[1:]:
            df_ = AB[(si, "BN")] - AB[(si, k)]; lo, hi = np.percentile(df_, [2.5, 97.5]); row[f"aucdiff_BN_minus_{k}"] = PT[(si, "BN")] - PT[(si, k)]; row[f"aucdiff_BN_minus_{k}_lo"] = float(lo); row[f"aucdiff_BN_minus_{k}_hi"] = float(hi)
        for r in RS:
            trv = min(r * ys.sum(), ns - ys.sum()); SB = {}; SP = {}
            for k in NAMES:
                fl = sc(dd, k)[common] >= tauof(dd, k, r); c = np.where(ys == 1, r * (~fl), fl).astype(float); SP[k] = 100 * (1 - c.sum() / trv)
                SB[k] = 100 * (1 - (Wc.astype(np.float32) @ c.astype(np.float32)) / np.minimum(r * Dw, ns - Dw)); lo, hi = np.percentile(SB[k], [2.5, 97.5]); row[f"saving_{k}_r{r}"] = SP[k]; row[f"saving_{k}_r{r}_lo"] = float(lo); row[f"saving_{k}_r{r}_hi"] = float(hi)
            for k in NAMES[1:]:
                lo, hi = np.percentile(SB["BN"] - SB[k], [2.5, 97.5]); row[f"savdiff_BN_minus_{k}_r{r}"] = SP["BN"] - SP[k]; row[f"savdiff_BN_minus_{k}_r{r}_lo"] = float(lo); row[f"savdiff_BN_minus_{k}_r{r}_hi"] = float(hi)
        rows.append(row)
        for m in ("LR", "RF"):
            gS = PT[(si, f"{m}_B")] - PT[(si, "BN")]; gb = AB[(si, f"{m}_B")] - AB[(si, "BN")]; gd = gb - gfull[m][1]; lo, hi = np.percentile(gd, [2.5, 97.5])
            gaps.append(dict(stage=stage, subset=si, observed=row["observed"], n_observed=row["n_observed"], model=m, gap_S=gS, gap_full=gfull[m][0], gap_S_minus_gap_full=gS - gfull[m][0], lo=float(lo), hi=float(hi), sig_positive=bool(lo > 0), sig_negative=bool(hi < 0)))
    # ---- criterion (After acceptance; Application descriptive)
    R_ = pd.DataFrame([r for r in rows if r["stage"] == stage]); G_ = pd.DataFrame([g_ for g_ in gaps if g_["stage"] == stage]); nonfull = R_.subset != full
    crit = {}
    for m in ("LR", "RF"):
        gm = G_[(G_.model == m) & (G_.subset != full)]
        crit[f"not_sig_positive_{m}_B"] = int((~gm.sig_positive).sum()); crit[f"sig_positive_{m}_B"] = int(gm.sig_positive.sum())
        crit[f"BN_above_{m}_IMP"] = int((R_[nonfull][f"aucdiff_BN_minus_{m}_IMP_lo"] > 0).sum()); crit[f"BN_below_{m}_IMP"] = int((R_[nonfull][f"aucdiff_BN_minus_{m}_IMP_hi"] < 0).sum())
    first = crit["not_sig_positive_LR_B"] >= 14 and crit["not_sig_positive_RF_B"] >= 14 if stage == "after" else None
    second = crit["BN_above_LR_IMP"] >= 5 and crit["BN_above_RF_IMP"] >= 5 if stage == "after" else None
    neg = (crit["sig_positive_LR_B"] > 4 or crit["sig_positive_RF_B"] > 4)
    verdict = "HELPFUL" if (first and second) else ("FLAT" if first else ("NEGATIVE" if neg else "NEITHER HELPFUL, FLAT NOR NEGATIVE (see counts)"))
    SUMM[stage] = dict(crit=crit, first=first, second=second, neg=neg, verdict=verdict, n_subsets=len(D), n_models_retrained=len(D) - 1)
    print(stage, SUMM[stage])
pd.DataFrame(rows).to_csv(ADD + "/j21_subset_results.csv", index=False); pd.DataFrame(gaps).to_csv(ADD + "/j21_gap_vs_full.csv", index=False)
pickle.dump(dict(SUMM=SUMM, checks=p.checks), open(ADD + "/j21/j21_summary.pkl", "wb"))
