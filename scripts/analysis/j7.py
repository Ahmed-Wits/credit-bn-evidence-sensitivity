"""J7: summary statistics of S_exp for the variables plotted in S5 and S6 (test rows, each row counted once)."""
from addcommon import *
import pickle
p = Part("j7")
P_APP = "selection/predictions/sel_rq2_per_row.pkl"; P_MON = "full_variable/per_row/vo_blanket_original_per_row.pkl"
app = pickle.load(open(R + "/" + P_APP, "rb"))["per_row"]; mon = pd.read_pickle(R + "/" + P_MON)
tabA = pd.read_csv(R + "/selection/tables/sel_rq2_sensitivity_table.csv").set_index("variable")
tabM = pd.read_csv(R + "/full_variable/tables/rq2_sensitivity_table.csv"); tabM = tabM[tabM.measure == "S_exp"].set_index("variable")
IDS = {"Age": "N-SEL-016", "Debt Ratio": "N-SEL-017", "Credit Utilisation Rate": "N-MON-016", "30-59 Days Past Due": "N-MON-018", "90+ Days Late": "N-MON-020", "60-89 Days Past Due": "N-MON-022"}
ITEMS = [("selection", "Application", app, P_APP, v, v + " (bin)", tabA, v) for v in ("Age", "Debt Ratio")] + \
        [("monitoring", "After acceptance", mon, P_MON, v, v + " (bin)", tabM, v + " (bin)") for v in ("Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due")]
STATS = [("5th percentile", 5), ("25th percentile", 25), ("median", 50), ("75th percentile", 75), ("95th percentile", 95), ("99th percentile", 99)]
tab = ["| Stage | Variable | rows | 5th | 25th | median | 75th | 95th | 99th | mean | mean / median |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for st, lab, df, src, nm, key, tb, tk in ITEMS:
    g = df[df.variable == key]; x = g.S_exp.dropna().to_numpy(); assert g.row_index.is_unique
    n = len(x); q = {k: float(np.percentile(x, k)) for _, k in STATS}; mean = float(x.mean()); med = q[50]; ratio = mean / med
    dec = 5 if st == "selection" else 4
    # reproduction
    p.check(f"{lab} {nm}: {n:,} rows, each test row once (row_index unique)", n == int(tb.loc[tk, "n_rows"]) or (nm == "Debt Ratio" and n == 30000), f"table n_rows {int(tb.loc[tk,'n_rows'])}")
    p.check(f"{lab} {nm}: mean {mean:.{dec}f} equals ledger {IDS[nm]} ({LED.loc[IDS[nm], 'display'].split(' ')[0]})", f"{mean:.{dec}f}" == LED.loc[IDS[nm], "display"].split(" ")[0])
    mcol = "mean_S_exp" if st == "selection" else "mean"
    p.near(f"{lab} {nm}: mean equals the stored table mean", mean, tb.loc[tk, mcol], 1e-9)
    p.near(f"{lab} {nm}: median equals the stored table median", med, tb.loc[tk, "median"], 1e-9)
    p.near(f"{lab} {nm}: 99th percentile equals the stored table p99", q[99], tb.loc[tk, "p99"], 1e-9)
    if st == "monitoring":
        p.near(f"{lab} {nm}: 95th percentile equals the stored table p95", q[95], tb.loc[tk, "p95"], 1e-9)
    for sname, k in STATS + [("mean", None)]:
        val = mean if k is None else q[k]
        p.row(st, f"Summary of S_exp over test rows, {nm}, {lab}: {sname}", val, f"{val:.{dec}f}", "probability", src, f"variable={key} | S_exp | {sname} over rows (np.percentile, linear interpolation)" if k else f"variable={key} | S_exp | mean over rows",
              derived="Y", notes=f"{n:,} test rows, each row counted once; same statistics as the plotted S5/S6 boxes (box 25th to 75th, whiskers 5th to 95th, cross 99th)", display_rule=("s5" if st == "selection" else "s4"), n=n, model="BN", variable=nm, caveat="C7")
    p.row(st, f"Ratio of mean S_exp to median S_exp, {nm}, {lab}", ratio, f"{ratio:.2f}", "ratio", src, f"variable={key} | mean(S_exp) / median(S_exp)", derived="Y", notes=f"{n:,} test rows; mean {mean:.{dec+1}f}, median {med:.{dec+1}f}; the mean exceeds the median because the distribution is right-skewed", display_rule="r2", n=n, model="BN", variable=nm, caveat="C7")
    tab.append(f"| {lab} | {nm} | {n:,} | " + " | ".join(f"{q[k]:.{dec}f}" for _, k in STATS[:5]) + f" | {q[99]:.{dec}f} | {mean:.{dec}f} | {ratio:.2f} |")
md = ["## J7. S5 and S6 summary statistics", "",
      "**Method.** For each plotted variable, the median, mean and the 5th, 25th, 75th, 95th and 99th percentiles of S_exp over test rows, each row counted once (`numpy.percentile`, linear interpolation), and the ratio mean / median. Application: Age and Debt Ratio from `%s` (`S_exp`; Age: 2 rows with undefined S_exp excluded; Debt Ratio: 1 row has S_exp = 0, kept). After acceptance: the four blanket variables from `%s` (`S_exp`, 29,998 rows each; `rq2_sensitivity_per_row.pkl` is not used because its column `S` is S_max)." % (P_APP, P_MON), "", *tab, "", checks_md(p), "",
      "**Notes.** (a) Rows are test rows, not unique blanket patterns. (b) Means, medians and 99th percentiles reproduce the stored tables and ledger exactly. (c) The mean-to-median ratios After acceptance are 1.35 (Credit Utilisation Rate), 3.92 (30-59 Days Past Due), 8.52 (90+ Days Late) and 6.96 (60-89 Days Past Due); the supplement's 'about 8.6 times' for 90+ Days Late should read about 8.5 (8.6 comes from the rounded means 0.0223 / 0.0026). (d) Ledger `unit` for the ratio rows is `ratio`, a new descriptive label.", ""]
p.write("\n".join(md))
