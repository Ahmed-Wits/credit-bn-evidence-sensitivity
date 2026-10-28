"""J14: audit of every printed paired difference: observed difference on the common rows against the printed value and the bootstrap mean.
Gate: the printed values (point and interval) must reproduce first."""
from addcommon import *
import pickle
from costlib import *
p = Part("j14")
L = lambda f: pickle.load(open(R + "/" + f, "rb"))
def nums(disp):
    d = disp.replace(MINUS, "-").replace(" pts", "").replace("%", "").replace("[", "").replace("]", "").replace(",", ""); return [float(x) for x in d.split()]
def dec_of(disp):
    s = disp.split("[")[0].strip().split()[0].replace("%", ""); return len(s.split(".")[1]) if "." in s else 0
ITEMS = []   # dict(table, id, quantity, printed, observed, boot_mean, ...)
def add(table, lid, quantity, printed, observed, boot_mean, ci_ok, note="", unit="AUC difference", dec=4):
    printed_is = ("n/a (no bootstrap printed)" if boot_mean is None else None)
    if boot_mean is not None:
        mo = abs(round(observed, dec) - printed) < 1e-9; mb = abs(round(boot_mean, dec) - printed) < 1e-9
        printed_is = "consistent with both at the printed precision" if mo and mb else ("observed difference" if mo else ("bootstrap mean" if mb else "NEITHER"))
    else:
        printed_is = "observed difference" if abs(round(observed, dec) - printed) < 1e-9 else "NEITHER"
    ITEMS.append(dict(table=table, id=lid, quantity=quantity, printed=printed, observed=observed, boot_mean=boot_mean, ci_ok=ci_ok, printed_is=printed_is, unit=unit, dec=dec, note=note,
                      disc_po=abs(printed - observed), disc_ob=(abs(observed - boot_mean) if boot_mean is not None else None), disc_pb=(abs(printed - boot_mean) if boot_mean is not None else None)))
def auc_item(table, lid, qlabel, y, pa, pb, sign=1, note=""):
    disp = LED.loc[lid, "display"]; pv, lo, hi = nums(disp); obs, dif = paired_boot_full(y, pa, pb); l_, h_ = np.percentile(dif, [2.5, 97.5]); bm = float(dif.mean())
    ok = abs(round(l_, 4) - lo) < 1e-9 and abs(round(h_, 4) - hi) < 1e-9
    p.gate(f"{lid} printed interval {disp.split('[')[1]} reproduced from the saved predictions", ok, f"recomputed [{l_:+.4f}, {h_:+.4f}]")
    add(table, lid, qlabel, pv, obs, bm, ok, note)
# ------------------------------------------------ Application (Table 4 and S-Numbers): predictions saved
bp = L("selection/predictions/sel_baseline_test_predictions.pkl"); sb = L("selection/predictions/sel_bn_test_predictions.pkl"); yA = np.asarray(bp["y"]).astype(int)
PA = {"BN": np.asarray(sb["p"], float), "LR": np.asarray(bp["p_lr"], float), "RF": np.asarray(bp["p_rf"], float)}
sbt = pd.read_csv(R + "/selection/tables/sel_baselines_bootstrap.csv").set_index("comparison")
for lid, a, b, tbl in (("N-SEL-004", "BN", "LR", "Table 4"), ("N-SEL-005", "BN", "RF", "Table 4"), ("N-SEL-006", "LR", "RF", "S-Numbers (AUC)")):
    auc_item(tbl, lid, f"Application test AUC difference, {a} minus {b}", yA, PA[a], PA[b])
    o = ITEMS[-1]; st = sbt.loc[f"{a} - {b}"]
    p.check(f"{lid}: stored `diff` in sel_baselines_bootstrap.csv ({st['diff']:.6f}) equals the observed difference ({o['observed']:.6f}), not a bootstrap mean", abs(st["diff"] - o["observed"]) < 1e-9)
# ------------------------------------------------ After acceptance (Table 6 and S-Numbers): predictions saved
bb = L("full_variable/predictions/bn_test_predictions.pkl")["bn_test_probs"]; ab = L("full_variable/predictions/ablation_predictions.pkl"); sp = L("Data/processed/split_indices.pkl"); yM = np.asarray(sp["y_test"]).astype(int)
v = bb.notna().to_numpy(); yv = yM[v]; BN = bb.to_numpy(float)[v]; LR = np.asarray(ab["LR_all"], float); RF = np.asarray(ab["RF_all"], float)
auc_item("Table 6", "N-MON-007", "After-acceptance test AUC difference, LR minus BN", yv, LR[v], BN, note="29,998 common rows")
auc_item("Table 6", "N-MON-008", "After-acceptance test AUC difference, RF minus BN", yv, RF[v], BN, note="29,998 common rows")
# N-MON-009: LR minus RF: try the 30,000 rows and the 29,998 rows
disp = LED.loc["N-MON-009", "display"]; pv, lo, hi = nums(disp)
for nm, yy, a_, b_ in (("30,000 rows", yM, LR, RF), ("29,998 rows", yv, LR[v], RF[v])):
    obs, dif = paired_boot_full(yy, a_, b_); l_, h_ = np.percentile(dif, [2.5, 97.5])
    if abs(round(l_, 4) - lo) < 1e-9 and abs(round(h_, 4) - hi) < 1e-9: break
p.gate(f"N-MON-009 printed interval reproduced from the saved predictions ({nm})", abs(round(l_, 4) - lo) < 1e-9 and abs(round(h_, 4) - hi) < 1e-9, f"recomputed [{l_:+.4f}, {h_:+.4f}]")
add("S-Numbers (AUC)", "N-MON-009", "After-acceptance test AUC difference, LR minus RF", pv, obs, float(dif.mean()), True, note=nm)
for lid, key, qlab in (("N-MON-041", "LR_noID", "Ablation paired AUC difference, LR all variables minus LR without Income and Dependents"), ("N-MON-042", "RF_noID", "Ablation paired AUC difference, RF all variables minus RF without Income and Dependents")):
    full = LR if "LR" in key else RF; auc_item("S-Numbers (AUC)", lid, qlab, yM, full, np.asarray(ab[key], float), note="30,000 rows")
for lid, key, full, sgnf in (("N-MON-038", "LR_noID", LR, -1), ("N-MON-039", "RF_noID", RF, -1)):
    obs = roc_auc_score(yM, np.asarray(ab[key], float)) - roc_auc_score(yM, full); pv = nums(LED.loc[lid, "display"])[0]
    add("S-Numbers (AUC)", lid, LED.loc[lid, "quantity"], pv, obs, None, True, note="no interval printed", dec=5)
# ------------------------------------------------ continuous reference rows (predictions of the continuous models were not saved: observed difference from the stored AUCs on the same rows)
for stage, f, ids in (("Application", "selection/tables/sel_continuous_reference", {"N-SEL-011": ("continuous LR minus binned LR", "all", "LR_cont", "LR_binned"), "N-SEL-012": ("continuous RF minus binned RF", "all", "RF_cont", "RF_binned"), "N-SEL-013": ("continuous LR minus BN", "bn_valid", "LR_cont", "BN"), "N-SEL-014": ("continuous RF minus BN", "bn_valid", "RF_cont", "BN")}),
                      ("After acceptance", "monitoring_rq3/mon_continuous_reference", {"N-MON-012": ("continuous LR minus binned LR", "all", "LR_cont", "LR_binned"), "N-MON-013": ("continuous RF minus binned RF", "all", "RF_cont", "RF_binned"), "N-MON-014": ("continuous LR minus BN", "bn_valid", "LR_cont", "BN"), "N-MON-015": ("continuous RF minus BN", "bn_valid", "RF_cont", "BN")})):
    T = pd.read_csv(R + "/" + f + "_table.csv"); Bt = pd.read_csv(R + "/" + f + "_bootstrap.csv")
    for lid, (cmpn, rs, a_, b_) in ids.items():
        auc = lambda m: float(T[(T.model == m) & (T.row_set == rs)].auc.iloc[0]); obs = auc(a_) - auc(b_); st = Bt[(Bt.comparison == cmpn) & (Bt.row_set == rs)].iloc[0]; pv = nums(LED.loc[lid, "display"])[0]
        p.gate(f"{lid} printed value and interval equal the stored bootstrap row and the stored AUC difference ({obs:+.6f})", abs(round(st["diff"], 4) - pv) < 1e-9 and abs(st["diff"] - obs) < 1e-6 and abs(round(st.ci_lo, 4) - nums(LED.loc[lid, "display"])[1]) < 1e-9, f"stored diff {st['diff']:.6f}")
        add("S-Numbers (AUC)", lid, f"{stage}: " + LED.loc[lid, "quantity"], pv, obs, None, True, note="predictions of the continuous-feature models were not saved: bootstrap mean not recomputable without refitting; the stored `diff` equals the observed difference of the stored AUCs")
# ------------------------------------------------ cost savings: observed (printed) against the J2 bootstrap mean
comb = pd.read_csv(R + "/selection/tables/sel_decision_value_combined.csv")
ST = {"Application": ("selection", yA, PA), "After acceptance": ("monitoring", yv, {"BN": BN, "LR": LR[v], "RF": RF[v]})}
RSC = list(range(1, 21)); BOOT = {}
for stage, (key, y_, P_) in ST.items():
    Tm = {m: {r: float(comb[(comb.contribution == key) & (comb.model == m) & (comb.r == r)].tau.iloc[0]) for r in RSC} for m in P_}
    pt = savings_point(y_, P_, Tm, RSC); BOOT[stage] = (pt, saving_boot(y_, P_, Tm, RSC))
def bm_sav(stage, m, r): return float(100 * BOOT[stage][1][m][:, r - 1].mean())
def bm_dif(stage, a, b, r): return float(100 * (BOOT[stage][1][a][:, r - 1] - BOOT[stage][1][b][:, r - 1]).mean())
def obs_sav(stage, m, r): return 100 * BOOT[stage][0][(m, r)][0]
IDS_T7 = {("Application", "BN", 5): "N-SEL-042", ("Application", "LR", 5): "N-SEL-043", ("Application", "RF", 5): "N-SEL-044", ("After acceptance", "BN", 5): "N-MON-045", ("After acceptance", "LR", 5): "N-MON-046", ("After acceptance", "RF", 5): "N-MON-047"}
for st_, base in (("Application", 0), ("After acceptance", 9)):
    for k, r in enumerate((10,)):
        for mi, m in enumerate(("BN", "LR", "RF")): IDS_T7[(st_, m, r)] = "N-XST-%03d" % (base + mi + 1)
for (stage, m, r), lid in IDS_T7.items():
    disp = LED.loc[lid, "display"]; pv = float(disp.rstrip("%")); dec = dec_of(disp)
    p.gate(f"Table 7 {stage} {m} r={r}: printed saving {disp} equals the observed saving ({obs_sav(stage, m, r):.4f}%)", abs(round(obs_sav(stage, m, r), dec) - pv) < 1e-9)
    add("Table 7", lid, f"Cost saving, {m}, {stage}, r={r}", pv, obs_sav(stage, m, r), bm_sav(stage, m, r), True, unit="percent", dec=dec, note="point value printed = observed saving (no bootstrap mean printed)")
for lid, (stage, a, b, r) in {"N-SEL-057": ("Application", "BN", "LR", 10), "N-SEL-058": ("Application", "BN", "RF", 8), "N-MON-056": ("After acceptance", "BN", "LR", 7), "N-MON-057": ("After acceptance", "BN", "RF", 3), "N-MON-059": ("After acceptance", "BN", "LR", 2), "N-MON-060": ("After acceptance", "BN", "RF", 11)}.items():
    pv = nums(LED.loc[lid, "display"])[0]; ob = 100 * (BOOT[stage][0][(a, r)][0] - BOOT[stage][0][(b, r)][0])
    p.gate(f"{lid}: printed {LED.loc[lid,'display']} equals the observed saving difference ({ob:+.4f} points)", abs(round(ob, 2) - pv) < 1e-9)
    add("S-Numbers (cost)", lid, f"{stage}: saving difference {a} minus {b}, r={r} (points)", pv, ob, bm_dif(stage, a, b, r), True, unit="percentage points", dec=2, note="printed = observed difference of observed savings")
# ------------------------------------------------ RQ3: observed seed-mean difference against the bootstrap mean
SA = pd.read_csv(R + "/selection/tables/sel_rq3_row_bootstrap_summary.csv"); SM = pd.read_csv(ADD + "/mon_rq3_row_bootstrap_summary.csv")
sumA = pd.read_csv(R + "/selection/tables/sel_rq3_summary_v2.csv"); sumA["level"] = sumA.level.round(2)
led_rq = LED[(LED.index.str.startswith("N-RQ3")) & (LED.imputer == "refit-imputed")]
def rq3_rows(df, stage_name, table, led_stage):
    n = 0
    for _, r in df.iterrows():
        m = led_rq[(led_rq.stage == led_stage) & (led_rq.variable == r.variable) & (led_rq.mechanism == r.mechanism) & (led_rq.level == "%d%%" % round(r.level * 100)) & (led_rq.comparator == r.contrast[-2:])]
        assert len(m) == 1; lid = m.index[0]; disp = m.display.iloc[0]
        pv = nums(disp)[0] if "[" in disp else nums(disp)[0]
        okp = abs(round(r.point_estimate, 4) - pv) < 1e-9 and (led_stage != "rq3-selection" or abs(round(r.ci_lo, 4) - nums(disp)[1]) < 1e-9)
        p.gate(f"{lid} ({stage_name} {r.variable} {r.mechanism} {int(r.level*100)}% {r.contrast}): printed value (and interval) equals the observed seed-mean difference and the stored bootstrap row", okp, f"{r.point_estimate:+.6f} vs bootstrap mean {r.boot_mean:+.6f}")
        add(table, lid, f"{stage_name} AUC-loss difference {r.contrast}, {r.variable} masked, {r.mechanism}, {int(r.level*100)}%", pv, float(r.point_estimate), float(r.boot_mean), True, note="mean over 10 mask seeds; printed and plotted point = observed")
rq3_rows(SA, "Application", "Figure 5 and S-Numbers (RQ3, Application)", "rq3-selection")
rq3_rows(SM, "After acceptance", "Figure S8 (RQ3, After acceptance)", "rq3-monitoring")
# median-imputed rows: printed = observed seed mean (no bootstrap exists)
ok_med = True; nmed = 0
sumM = pd.read_csv(R + "/monitoring_rq3/mon_rq3_summary_v2.csv"); sumM["level"] = sumM.level.round(2)
for lid, row in LED[(LED.index.str.startswith("N-RQ3")) & (LED.imputer == "median")].iterrows():
    S_ = sumA if row.stage == "rq3-selection" else sumM
    g = S_[(S_.variable == row.variable) & (S_.mechanism == row.mechanism) & (S_.level.round(2) == float(row.level.rstrip("%")) / 100)].set_index("variant").mean_delta_auc
    ob = g["BN"] - g["%s_median" % row.comparator]; ok_med &= abs(round(ob, 4) - nums(row.display)[0]) < 1e-9; nmed += 1
p.gate(f"all {nmed} median-imputed RQ3 'minus' rows: printed value equals the observed seed-mean difference (no bootstrap exists for them)", ok_med and nmed == 72)
# ------------------------------------------------ summary
D = pd.DataFrame(ITEMS); D.to_csv(ADD + "/paired_difference_audit.csv", index=False)
grp = D.groupby("table").agg(n=("id", "size"), max_printed_vs_observed=("disc_po", "max"), max_observed_vs_bootmean=("disc_ob", "max"), max_printed_vs_bootmean=("disc_pb", "max"))
overall = dict(po=float(D.disc_po.max()), ob=float(D.disc_ob.dropna().max()), pb=float(D.disc_pb.dropna().max()))
neither = D[D.printed_is == "NEITHER"]
p.check("every audited printed value equals the observed difference or the bootstrap mean at its printed precision (no 'NEITHER')", len(neither) == 0, str(list(neither.id)))
SRC = "Submissions/4. Article/WRITEUP/analysis_additions/paired_difference_audit.csv"
for _, r in D.iterrows():
    p.row("cross-stage", f"Paired-difference audit: {r.quantity} ({r.table}, {r.id}): absolute discrepancy between the observed difference and the bootstrap mean" if r.boot_mean is not None else f"Paired-difference audit: {r.quantity} ({r.table}, {r.id}): absolute discrepancy between the printed value and the observed difference",
          (r.disc_ob if r.boot_mean is not None else r.disc_po), f"{(r.disc_ob if r.boot_mean is not None else r.disc_po):.6f}", r.unit, SRC, f"ledger id {r.id} | observed vs bootstrap mean", derived="Y",
          notes=f"printed {r.printed}; observed difference on the common rows {r.observed:.6f}; bootstrap mean {('%.6f' % r.boot_mean) if r.boot_mean is not None else 'not available'}; printed value is the {r.printed_is}; {r.note}", display_rule="txt", n="", caveat="C6" if "RQ3" in r.table else "")
for t_, r in grp.iterrows():
    p.row("cross-stage", f"Paired-difference audit, maximum discrepancy in {t_} (observed vs bootstrap mean; printed vs observed)", float(np.nanmax([r.max_observed_vs_bootmean if pd.notna(r.max_observed_vs_bootmean) else 0, r.max_printed_vs_observed])), f"{max(r.max_observed_vs_bootmean if pd.notna(r.max_observed_vs_bootmean) else 0, r.max_printed_vs_observed):.6f}", "mixed units (AUC points, percent, percentage points)", SRC, f"table={t_} | max over {int(r.n)} items", derived="Y", notes=f"{int(r.n)} items; max |printed - observed| {r.max_printed_vs_observed:.6f}; max |observed - bootstrap mean| {('%.6f' % r.max_observed_vs_bootmean) if pd.notna(r.max_observed_vs_bootmean) else 'n/a'}", display_rule="txt")
p.row("cross-stage", "Paired-difference audit, overall maximum discrepancy between the observed difference and the bootstrap mean", overall["ob"], f"{overall['ob']:.6f}", "mixed units", SRC, "max over all audited items", derived="Y", notes=f"overall max |printed - observed| {overall['po']:.6f}; overall max |printed - bootstrap mean| {overall['pb']:.6f}", display_rule="txt")
fmt = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.6f}"
md = ["## J14. Audit of the printed paired differences", "",
      "**Method.** For every paired difference printed in the article (Tables 4, 6, 7) or in Table S-Numbers, and for the Fig 5 and Fig S8 contrasts, the observed difference on the common rows is recomputed from the saved predictions (AUC differences, savings) or taken from the stored summaries (continuous-reference models: predictions were not saved, so only the stored AUC difference is available), the bootstrap mean is recomputed with the original procedure where predictions exist (`RandomState(42)`, 2,000 resamples for AUC differences; the J2 multinomial bootstrap for savings; the stored `boot_mean` for RQ3), and both are compared with the printed value. **Gate (printed values reproduce first): PASSED** for every item (interval, and the printed point equals the observed difference or the bootstrap mean at its printed precision).", "",
      "**Which value is printed.** Application AUC differences (Table 4, `sel_baselines_bootstrap.csv`) and the continuous-reference differences print the **observed** difference. The After-acceptance Table 6 differences and the ablation differences (S-Numbers) are recorded in the ledger as the **mean of the bootstrap differences**; at the printed precision this is distinguishable from the observed difference only for N-MON-007 (LR minus BN: printed +0.0099 = bootstrap mean 0.009862; observed 0.009806 rounds to +0.0098); for the other rows both round to the same printed value. Savings (Table 7) and the saving differences in S-Numbers are **observed** values (the J2 intervals are new). RQ3 points (Figs 5 and S8, S-Numbers) are the **observed** mean over the 10 mask seeds; the stored bootstrap mean is different and is not printed.", "",
      "| Table | items | max abs(printed - observed) | max abs(observed - bootstrap mean) | max abs(printed - bootstrap mean) |", "|---|---|---|---|---|"] + \
     [f"| {t_} | {int(r.n)} | {fmt(r.max_printed_vs_observed)} | {fmt(r.max_observed_vs_bootmean)} | {fmt(r.max_printed_vs_bootmean)} |" for t_, r in grp.iterrows()] + \
     [f"| **Overall** | {len(D)} | {overall['po']:.6f} | {overall['ob']:.6f} | {overall['pb']:.6f} |", "",
      "(Units: AUC differences and AUC-loss differences in AUC points; savings in percent; saving differences in percentage points.)", "",
      "### Item list", "", "| Table | Ledger id | Quantity | Printed | Observed | Bootstrap mean | Printed value is |", "|---|---|---|---|---|---|---|"] + \
     [f"| {r.table} | {r.id} | {r.quantity} | {r.printed:g} | {r.observed:.6f} | {fmt(r.boot_mean)} | {r.printed_is} |" for _, r in D.iterrows()] + ["", checks_md(p), "",
      "**Notes.** (a) The only printed value that differs from the observed difference at its printed precision is N-MON-007 (Table 6, LR minus BN): +0.0099 is the bootstrap mean, the observed difference is +0.0098 (0.009806); 67 of the 108 items are consistent with both, 40 equal the observed difference only (no bootstrap mean printed or recomputable), 1 equals the bootstrap mean only. The savings discrepancies in Table 7 (up to 0.05 percentage points) are the rounding of the percent to one decimal. (b) Median-imputed RQ3 rows (72) print the observed seed-mean difference; no bootstrap exists for them. (c) The continuous-reference rows cannot be audited against a bootstrap mean without refitting the continuous models (not done).", ""]
p.write("\n".join(md))
print(grp.to_string()); print(overall)
