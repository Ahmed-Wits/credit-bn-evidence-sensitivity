"""Tier B (supplement) figures S1-S6. S7 is in figS7.py. Plots only from saved outputs."""
import os
from common import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, roc_auc_score

SRC = {}
STATS = {}   # extra numbers for the verify report

# ------------------------------------------------------------------ S1
def s1():
    NAME = "figS1_roc"
    P = pd.read_pickle(R + "/selection/predictions/sel_baseline_test_predictions.pkl")
    y = P["y"]; sc = {"BN": P["p_bn"], "LR": P["p_lr"], "RF": P["p_rf"]}
    ids = {"BN": "N-SEL-001", "LR": "N-SEL-002", "RF": "N-SEL-003"}
    fig = plt.figure(figsize=(W1, 83.7 * MM)); ax = fig.add_axes([0.17, 0.15, 0.78, 0.80])
    ax.plot([0, 1], [0, 1], color="0.45", lw=0.8, ls=(0, (1, 2)), label="Chance", zorder=1)
    for m in ("RF", "LR", "BN"):
        fpr, tpr, _ = roc_curve(y, sc[m]); auc = roc_auc_score(y, sc[m])
        check(NAME, f"AUC {m}", auc, lv(ids[m]), 4, "ledger " + ids[m])
        lab = f"{m} (AUC {auc:.4f})"; check_text(NAME, f"{m} legend label", lab, f"{m} (AUC {rnd(lv(ids[m]),4):.4f})", "ledger " + ids[m])
        kw = dict(markevery=None) if m == "BN" else dict(markevery=0.09)
        ax.plot(fpr, tpr, color=COL[m], ls=LS[m], marker=MK[m], ms=(3.2 if m == "BN" else 3.0), mew=0, lw=(1.7 if m == "BN" else 1.0), label=lab, zorder=(4 if m == "BN" else 3), **kw)
        if m == "BN": STATS["S1_BN_distinct_scores"] = int(len(np.unique(sc[m])))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    h, l = ax.get_legend_handles_labels(); order = [l.index(x) for x in sorted(l, key=lambda s: ("Chance" in s, ["BN", "LR", "RF"].index(s[:2]) if s[:2] in ("BN", "LR", "RF") else 9))]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower right", frameon=False, handlelength=2.6, borderaxespad=0.5)
    ax.grid(color="0.93", lw=0.5, zorder=0); ax.set_axisbelow(True)
    NOTES[NAME] = ["Old figure had no title; legend AUCs are the ledger values (BN 0.6627, LR 0.6904, RF 0.6926). The BN curve is piecewise linear because the BN produces at most 25 distinct scores (%d distinct values here); its markers sit at the vertices, LR and RF carry markers at regular intervals along the curve only to identify them without colour." % STATS["S1_BN_distinct_scores"]]
    SRC[NAME] = "selection/predictions/sel_baseline_test_predictions.pkl (y, p_bn, p_lr, p_rf)"
    return fig, NAME

# ------------------------------------------------------------------ S2
def s2():
    NAME = "figS2_calibration"
    P = pd.read_pickle(R + "/selection/predictions/sel_baseline_test_predictions.pkl"); y = P["y"]
    H = 70 * MM; fig = plt.figure(figsize=(W2, H))
    Lm, Rm, Tm, Bm, gap = 0.62, 0.10, 0.08, 0.50, 0.70
    aw = (W2 - Lm - Rm - gap) / 2; ah = H - Tm - Bm
    a1, a2 = [fig.add_axes([(Lm + k * (aw + gap)) / W2, Bm / H, aw / W2, ah / H]) for k in range(2)]
    check(NAME, "BN mean predicted probability", P["p_bn"].mean(), lv("N-SEL-007"), 4, "ledger N-SEL-007")
    check(NAME, "observed test default rate", y.mean(), lv("N-SEL-008"), 4, "ledger N-SEL-008")
    d = pd.DataFrame({"p": P["p_bn"], "y": y}).groupby("p").agg(obs=("y", "mean"), n=("y", "size")).reset_index()
    STATS["S2_BN_distinct"] = len(d); assert d.n.sum() == 30000
    a1.plot([0, 0.15], [0, 0.15], color="0.45", lw=0.8, ls=(0, (1, 2)), zorder=1)
    k = 6.0
    a1.scatter(d.p, d.obs, s=d.n / k * 0.0 + np.sqrt(d.n) * 0.55, color=COL["BN"], alpha=0.85, edgecolor="white", linewidth=0.3, zorder=3, label="BN")
    a1.set_xlim(0, 0.15); a1.set_ylim(0, 0.15)
    a2.plot([0, 1], [0, 1], color="0.45", lw=0.8, ls=(0, (1, 2)), zorder=1)
    for m, col in (("LR", "p_lr"), ("RF", "p_rf")):
        q = pd.qcut(pd.Series(P[col]), 10, duplicates="drop"); g = pd.DataFrame({"p": P[col], "y": y}).groupby(q.values, observed=True).mean()
        a2.plot(g.p, g.y, color=COL[m], ls=LS[m], marker=MK[m], ms=3.6, lw=1.0, mew=0, zorder=3)
        STATS["S2_%s_mean_pred" % m] = float(P[col].mean())
    a2.set_xlim(0, 1); a2.set_ylim(0, 1)
    for a in (a1, a2):
        a.set_xlabel("Predicted probability of default"); a.set_ylabel("Observed default rate"); a.grid(color="0.93", lw=0.5, zorder=0); a.set_axisbelow(True)
    a1.text(0.02, 0.97, "(a) BN", transform=a1.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
    a2.text(0.02, 0.97, "(b) LR and RF", transform=a2.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
    # legends
    ns = [200, 2000, 10000]
    hs = [Line2D([], [], color="0.45", lw=0.8, ls=(0, (1, 2)), label="Perfect calibration")] + \
         [Line2D([], [], color=COL["BN"], marker="o", ls="", ms=np.sqrt(np.sqrt(n) * 0.55) , label=f"BN, n = {n:,}") for n in ns]
    a1.legend(handles=hs, loc="lower right", frameon=False, borderaxespad=0.4, handletextpad=0.5, labelspacing=0.45)
    hs2 = [Line2D([], [], color="0.45", lw=0.8, ls=(0, (1, 2)), label="Perfect calibration")] + \
          [Line2D([], [], color=COL[m], ls=LS[m], marker=MK[m], ms=3.6, lw=1.0, mew=0, label=m) for m in ("LR", "RF")]
    a2.legend(handles=hs2, loc="center left", frameon=False, borderaxespad=0.4)
    NOTES[NAME] = ["Old figure carried panel titles 'BN (25 distinct predicted values)' and 'LR and RF (class_weight balanced)' and a footer 'Test set, n = 30,000, observed default rate 0.0668. Mean predicted: BN 0.0669, LR %.4f, RF %.4f. Balanced class weights inflate LR and RF probabilities, so only their ranking is meaningful; the BN is calibrated.' (removed; panel letters now inside the panels)." % (STATS["S2_LR_mean_pred"], STATS["S2_RF_mean_pred"]),
                   "Each panel keeps its own axis limits as in the old figure: BN 0 to 0.15, LR/RF 0 to 1. The LR and RF points lie well below the diagonal (predicted probabilities too high) because both models were fitted with balanced class weights, which inflates their probabilities; only their ranking is meaningful.",
                   "BN: one point per distinct predicted value (%d values), marker area proportional to the square root of the number of test rows n (legend gives n = 200, 2,000, 10,000). LR and RF: ten bins of predicted probability (deciles), mean predicted against observed rate." % STATS["S2_BN_distinct"]]
    SRC["figS2_calibration"] = "selection/predictions/sel_baseline_test_predictions.pkl"
    return fig, NAME

# ------------------------------------------------------------------ S3 / S4
def _masking(name, summ_path, variables, panel_labels, nrow, ncol, H, stage):
    S = pd.read_csv(R + "/" + summ_path); S["level"] = S["level"].round(2)
    fig = plt.figure(figsize=(W2, H))
    Lm, Rm, Tm, Bm, gx, gy = 0.98, 0.10, 0.08, 0.80, 0.62, 0.45
    aw = (W2 - Lm - Rm - gx * (ncol - 1)) / ncol; ah = (H - Tm - Bm - gy * (nrow - 1)) / nrow
    axs = []
    for i, (v, lab) in enumerate(zip(variables, panel_labels)):
        r, c = divmod(i, ncol)
        ax = fig.add_axes([(Lm + c * (aw + gx)) / W2, (Bm + (nrow - 1 - r) * (ah + gy)) / H, aw / W2, ah / H]); axs.append(ax)
        for model, var_ in (("BN", "BN"), ("LR", "LR_primary"), ("RF", "RF_primary")):
            for mech, ls in (("mcar", "-"), ("dependent", "--")):
                q = S[(S.variable == v) & (S.mechanism == mech) & (S.variant == var_)].sort_values("level"); assert len(q) == 3
                x = q.level.values * 100
                ax.fill_between(x, q.mean_delta_auc - q.sd_delta_auc, q.mean_delta_auc + q.sd_delta_auc, color=COL[model], alpha=0.12, lw=0, zorder=1)
                ax.plot(x, q.mean_delta_auc, color=COL[model], ls=ls, marker=MK[model], ms=3.4, mew=0, lw=(1.5 if model == "BN" else 0.9), zorder=(4 if model == "BN" else 3))
        ax.axhline(0, color="0.5", lw=0.6, zorder=0)
        ax.set_xticks([20, 40, 50]); ax.set_xticklabels(["20%", "40%", "50%"]); ax.set_xlim(17, 53)
        ax.text(0.03, 0.04, lab, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
        ax.grid(axis="y", color="0.93", lw=0.5, zorder=0); ax.set_axisbelow(True)
        if c == 0: ax.set_ylabel("Change in test AUC\nvs 0% masking")
        if r == nrow - 1: ax.set_xlabel("Share of test rows masked")
    # ledger assertions: BN minus comparator = ledger paired differences (refit-imputed rows)
    L = LED[(LED.stage == stage) & (LED.imputer == "refit-imputed")]
    n_ok = 0
    for v in variables:
        for mech in ("mcar", "dependent"):
            for lev in (0.2, 0.4, 0.5):
                for comp, var_ in (("LR", "LR_primary"), ("RF", "RF_primary")):
                    g = S[(S.variable == v) & (S.mechanism == mech) & (S.level == lev)].set_index("variant").mean_delta_auc
                    diff = g["BN"] - g[var_]
                    m = L[(L.variable == v) & (L.mechanism == mech) & (L.level == "%d%%" % round(lev * 100)) & (L.comparator == comp)]
                    assert len(m) == 1, (v, mech, lev, comp, len(m)); lid = m.index[0]
                    check(name, f"{v} {mech} {int(lev*100)}% BN minus {comp} (from plotted means)", diff, lv(lid), 4, "ledger " + lid)
                    # also the stored paired summary row
                    pr = S[(S.variable == v) & (S.mechanism == mech) & (S.level == lev) & (S.variant == "BN_minus_%s_primary" % comp)].mean_delta_auc.iloc[0]
                    check(name, f"{v} {mech} {int(lev*100)}% stored BN_minus_{comp}_primary row", pr, lv(lid), 4, "ledger " + lid)
    hs = [Line2D([], [], color=COL[m], marker=MK[m], ls="-", ms=3.4, mew=0, lw=(1.5 if m == "BN" else 0.9), label=m) for m in ("BN", "LR", "RF")] + \
         [Line2D([], [], color="0.3", ls="-", lw=0.9, label="MCAR"), Line2D([], [], color="0.3", ls="--", lw=0.9, label="Dependent")]
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=2.2, handlelength=2.6, borderaxespad=0.15)
    return fig, axs

def s3():
    NAME = "figS3_masking_application"
    fig, _ = _masking(NAME, "selection/tables/sel_rq3_summary_v2.csv", ["Age", "Debt Ratio"], ["(a) Age", "(b) Debt Ratio"], 1, 2, 62 * MM, "rq3-selection")
    NOTES[NAME] = ["Old figure (sel_rq3_auc_degradation_v2.png) panel titles 'Age' and 'Debt Ratio' replaced by '(a) Age', '(b) Debt Ratio' inside the panels.",
                   "Lines are means over 10 mask seeds of the change in test AUC relative to no masking; bands are the standard deviation across the 10 seeds (they reflect mask randomness only; the SD columns are not in the ledger). MCAR solid, dependent dashed. LR and RF are the primary (refit-imputed) variants. The plotted BN minus LR and BN minus RF differences of the means are asserted against the ledger paired-difference rows (24 rows) and against the stored BN_minus_* summary rows."]
    SRC[NAME] = "selection/tables/sel_rq3_summary_v2.csv"
    return fig, NAME

def s4():
    NAME = "figS4_masking_after_acceptance"
    vs = ["Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due"]
    fig, _ = _masking(NAME, "monitoring_rq3/mon_rq3_summary_v2.csv", vs, ["(a) " + vs[0], "(b) " + vs[1], "(c) " + vs[2], "(d) " + vs[3]], 2, 2, 100 * MM, "rq3-monitoring")
    NOTES[NAME] = ["Old figure (mon_rq3_auc_degradation_v2.png) panel titles replaced by panel letters and names inside the panels; panels ordered as in Fig 2b (Credit Utilisation Rate, 30-59 Days Past Due, 90+ Days Late, 60-89 Days Past Due).",
                   "Lines are means over 10 mask seeds; bands are SD across seeds (mask randomness only). No intervals were computed at this stage. Differences of the plotted means are asserted against the 48 ledger paired-difference rows."]
    SRC[NAME] = "monitoring_rq3/mon_rq3_summary_v2.csv"
    return fig, NAME

# ------------------------------------------------------------------ S5 / S6
def _box(ax, data_by_var, labels, ymin, ymax, tick_lab=None):
    stats = []; pos = np.arange(1, len(labels) + 1)
    for v in labels:
        x = np.asarray(data_by_var[v]); q = np.percentile(x, [5, 25, 50, 75, 95, 99])
        stats.append(dict(med=q[2], q1=q[1], q3=q[3], whislo=q[0], whishi=q[4], fliers=[], mean=x.mean(), p99=q[5], n=len(x), nzero=int((x == 0).sum())))
    ax.set_yscale("log")
    ax.bxp([dict(med=s["med"], q1=s["q1"], q3=s["q3"], whislo=s["whislo"], whishi=s["whishi"], fliers=[]) for s in stats], positions=pos, widths=0.5, showfliers=False,
           patch_artist=True, boxprops=dict(facecolor=COL["BN"], alpha=0.35, edgecolor=COL["BN"], lw=0.9), medianprops=dict(color="black", lw=1.3),
           whiskerprops=dict(color=COL["BN"], lw=0.9), capprops=dict(color=COL["BN"], lw=0.9), capwidths=0.22)
    ax.scatter(pos, [s["mean"] for s in stats], marker="D", s=14, color="black", zorder=5, linewidths=0, label="Mean")
    ax.scatter(pos, [s["p99"] for s in stats], marker="x", s=18, color="black", zorder=5, linewidths=0.9, label="99th percentile")
    ax.set_ylim(ymin, ymax); ax.set_xlim(0.4, len(labels) + 0.6)
    ax.set_xticks(pos); ax.set_xticklabels(tick_lab or labels)
    ax.set_ylabel(SEXP + " (log scale)")
    ax.grid(axis="y", which="major", color="0.93", lw=0.5, zorder=0); ax.set_axisbelow(True)
    hs = [Patch(facecolor=COL["BN"], alpha=0.35, edgecolor=COL["BN"], label="25th to 75th percentile"),
          Line2D([], [], color=COL["BN"], lw=0.9, label="5th to 95th percentile"), Line2D([], [], color="black", lw=1.3, label="Median"),
          Line2D([], [], marker="D", color="black", ls="", ms=3.6, mew=0, label="Mean"), Line2D([], [], marker="x", color="black", ls="", ms=4, mew=0.9, label="99th percentile")]
    ax.legend(handles=hs, loc="upper right", frameon=True, framealpha=1.0, edgecolor="0.7", fancybox=False, borderaxespad=0.3, labelspacing=0.3, handlelength=1.6)
    return stats

def s5():
    NAME = "figS5_sexp_distribution_application"
    per = pd.read_pickle(R + "/selection/predictions/sel_rq2_per_row.pkl")["per_row"]
    tab = pd.read_csv(R + "/selection/tables/sel_rq2_sensitivity_table.csv").set_index("variable")
    # the four zero variables: asserted but NOT drawn
    for v in ("Monthly Income", "Number of Dependents", "Real Estate Loans", "Open Credit Lines"):
        s = per[per.variable == v + " (bin)"].S_exp.dropna()
        CHECKS.append(dict(figure=NAME, item=f"omitted variable {v}: S_exp = 0 on every row ({len(s):,} rows)", plotted="omitted", reference="all 0", decimals="", source="sel_rq2_per_row.pkl", ok=bool((s == 0).all())))
    data = {v: per[per.variable == v + " (bin)"].S_exp.dropna().to_numpy() for v in ("Age", "Debt Ratio")}
    H = 70 * MM; fig = plt.figure(figsize=(W1, H)); ax = fig.add_axes([0.20, 0.13, 0.76, 0.84])
    stats = _box(ax, data, ["Age", "Debt Ratio"], 1e-3, 3e-1)
    ids = {"Age": "N-SEL-016", "Debt Ratio": "N-SEL-017"}
    for v, s in zip(("Age", "Debt Ratio"), stats):
        check(NAME, f"{v} mean", s["mean"], lv(ids[v]), 5, "ledger " + ids[v])
        check(NAME, f"{v} median", s["med"], tab.loc[v, "median"], 5, "sel_rq2_sensitivity_table.csv")
        check(NAME, f"{v} 99th percentile", s["p99"], tab.loc[v, "p99"], 5, "sel_rq2_sensitivity_table.csv")
        STATS["S5_" + v] = dict(n=s["n"], nzero=s["nzero"], q=[s[k] for k in ("whislo", "q1", "med", "q3", "whishi", "p99")], mean=s["mean"])
    ax.set_xlabel("")
    NOTES[NAME] = ["Only Age and Debt Ratio are drawn. Monthly Income, Number of Dependents, Real Estate Loans, Open Credit Lines: S_exp = 0 on every row.",
                   "Statistics are over test rows, each row counted once: Age n = %d (2 test rows have undefined S_exp and are excluded), Debt Ratio n = %d. Box = 25th to 75th percentile, whiskers = 5th to 95th percentile, line = median, diamond = mean, cross = 99th percentile; vertical axis is logarithmic." % (STATS["S5_Age"]["n"], STATS["S5_Debt Ratio"]["n"]),
                   "On the log axis one Debt Ratio row with S_exp exactly 0 cannot be shown; it is included in all statistics (it affects no plotted value).",
                   "Old figure used a symlog axis, the six variables, and tick labels with '(n=...)'; the old y label was 'S_exp (symlog axis; linear below 0.001)'."]
    SRC[NAME] = "selection/predictions/sel_rq2_per_row.pkl (S_exp); selection/tables/sel_rq2_sensitivity_table.csv"
    return fig, NAME

def s6():
    NAME = "figS6_sexp_distribution_after_acceptance"
    b = pd.read_pickle(R + "/full_variable/per_row/vo_blanket_original_per_row.pkl")
    tab = pd.read_csv(R + "/full_variable/tables/rq2_sensitivity_table.csv"); tab = tab[tab.measure == "S_exp"]; tab["v"] = tab.variable.str.replace(" (bin)", "", regex=False); tab = tab.set_index("v")
    pl = pd.read_csv(R + "/full_variable/tables/rq2_pattern_level_check.csv"); pl["v"] = pl.variable.str.replace(" (bin)", "", regex=False); pl = pl.set_index("v")
    order = ["Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due"]
    ids = dict(zip(order, ["N-MON-016", "N-MON-018", "N-MON-020", "N-MON-022"]))
    data = {}
    for v in order:
        g = b[b.variable == v + " (bin)"]
        assert g.row_index.is_unique and g.S_exp.notna().all() and len(g) == 29998
        data[v] = g.S_exp.to_numpy()
        STATS["S6_" + v] = dict(rows=len(g), unique_rows=int(g.row_index.nunique()), nzero=int((g.S_exp == 0).sum()), skipped=int(g.skipped.sum()),
                                 pattern_level=bool(pl.loc[v, "S_exp_pattern_level"]), n_patterns=int(pl.loc[v, "n_patterns"]), spread=float(pl.loc[v, "S_exp_within_pattern_spread"]),
                                 distinct=int(pl.loc[v, "distinct_S_exp_values"]))
    # the per-row S column of rq2_sensitivity_per_row.pkl is S_max, not S_exp (documented in the report)
    pr = pd.read_pickle(R + "/full_variable/per_row/rq2_sensitivity_per_row.pkl")
    smax = pd.read_csv(R + "/full_variable/tables/rq2_sensitivity_table.csv"); smax = smax[smax.measure == "S_max"]; smax["v"] = smax.variable.str.replace(" (bin)", "", regex=False); smax = smax.set_index("v")
    for v in order:
        STATS["S6_" + v]["rq2_per_row_S_mean"] = float(pr[pr.variable == v + " (bin)"].S.mean()); STATS["S6_" + v]["table_S_max_mean"] = float(smax.loc[v, "mean"]); STATS["S6_" + v]["table_S_exp_mean"] = float(tab.loc[v, "mean"])
    H = 80 * MM; fig = plt.figure(figsize=(W1, H)); ax = fig.add_axes([0.20, 0.17, 0.76, 0.80])
    ticks = ["Credit\nUtilisation\nRate", "30-59 Days\nPast Due", "90+ Days\nLate", "60-89 Days\nPast Due"]
    stats = _box(ax, data, order, 3e-4, 8.0, tick_lab=ticks)
    for v, s in zip(order, stats):
        check(NAME, f"{v} mean", s["mean"], lv(ids[v]), 4, "ledger " + ids[v])
        check(NAME, f"{v} mean vs table mean (full precision)", s["mean"], tab.loc[v, "mean"], 8, "rq2_sensitivity_table.csv")
        check(NAME, f"{v} median", s["med"], tab.loc[v, "median"], 5, "rq2_sensitivity_table.csv")
        check(NAME, f"{v} 99th percentile", s["p99"], tab.loc[v, "p99"], 5, "rq2_sensitivity_table.csv")
        STATS["S6_" + v].update(q=[s[k] for k in ("whislo", "q1", "med", "q3", "whishi", "p99")], mean=s["mean"], n=s["n"])
    NOTES[NAME] = ["Tick labels no longer carry '(row-level)' or '(pattern-level)'. Definition to put in the caption or text: pattern-level means that every test row sharing a blanket pattern (the same values of the variables in Default's Markov blanket) has the same S_exp; row-level means that the value also depends on evidence outside the blanket. From rq2_pattern_level_check.csv: 30-59 Days Past Due and 60-89 Days Past Due are pattern-level (within-pattern spread 0); Credit Utilisation Rate and 90+ Days Late are row-level (within-pattern spread 0.269 and 0.074).",
                   "Statistics are over the 29,998 test rows where the BN is defined, each row counted once (not over unique blanket patterns). Box = 25th to 75th percentile, whiskers = 5th to 95th percentile, line = median, diamond = mean, cross = 99th percentile; log axis.",
                   "On the log axis 1 row (30-59 Days Past Due) and 6 rows (90+ Days Late) with S_exp exactly 0 cannot be shown; they are included in all statistics and affect no plotted value.",
                   "Old figure: six boxes with row-level/pattern-level tick labels and a symlog axis; replaced by four boxes with a log axis."]
    SRC[NAME] = "full_variable/per_row/vo_blanket_original_per_row.pkl (S_exp); full_variable/tables/rq2_sensitivity_table.csv; rq2_pattern_level_check.csv"
    return fig, NAME

MAKERS = [s1, s2, s3, s4, s5, s6]
if __name__ == "__main__":
    for mk in MAKERS:
        os.makedirs(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out"), exist_ok=True); f, n = mk(); print(n, audit(f, n)); f.savefig(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out", "%s.png" % n), dpi=200)
    print([c for c in CHECKS if not c["ok"]], len(CHECKS))
