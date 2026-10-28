"""J5: flag-transition tables, Application BN (Age and Debt Ratio retracted, r = 5, 10, 15, 20)."""
from addcommon import *
import pickle
p = Part("j5")
SRCP = "selection/predictions/sel_rq2_per_row.pkl"
o = pickle.load(open(R + "/" + SRCP, "rb")); per = o["per_row"]; tg = o["tau_grid"]
pa = pd.read_csv(R + "/selection/tables/sel_voi_partA_table.csv")
cells = pd.read_csv(R + "/selection/tables/sel_default_cells.csv")
comb = pd.read_csv(R + "/selection/tables/sel_decision_value_combined.csv"); comb = comb[(comb.contribution == "selection") & (comb.model == "BN")].set_index("r")
sb = pickle.load(open(R + "/selection/predictions/sel_bn_test_predictions.pkl", "rb")); pbn = np.asarray(sb["p"], float)
RS = [5, 10, 15, 20]
V = {"Age": "Age (bin)", "Debt Ratio": "Debt Ratio (bin)"}
d = {v: per[per.variable == k].set_index("row_index") for v, k in V.items()}
# population: rows whose retracted probability is defined for BOTH variables (the 29,998 rows of Part A)
common = d["Age"].retracted_prob.notna() & d["Debt Ratio"].retracted_prob.notna().reindex(d["Age"].index)
idx = d["Age"].index[common.values]
p.check("population of the Part A table: rows with defined baseline and retracted probabilities", len(idx) == 29998 == int(pa.n_rows.iloc[0]), f"{len(idx)} rows; table n_rows {int(pa.n_rows.iloc[0])}")
p.check("Debt Ratio has %d rows with defined retracted probability in the per-row file (the 2 rows excluded for Age are excluded for both)" % int(d["Debt Ratio"].retracted_prob.notna().sum()), True)
LEDID = {("Age", 10): "N-SEL-018", ("Debt Ratio", 10): "N-SEL-019"}
tab = {}
SRCF = "selection/tables/sel_voi_partA_table.csv"
for var in V:
    g = d[var].loc[idx]
    for r in RS:
        tau = tg[r]; p.near(f"threshold r={r}: tau_grid equals 1/(1+r)", tau, 1.0 / (1 + r), 1e-15)
        fa = (g.baseline_prob.values >= tau); fr = (g.retracted_prob.values >= tau)
        both, a_only, r_only, neither = int((fa & fr).sum()), int((fa & ~fr).sum()), int((~fa & fr).sum()), int((~fa & ~fr).sum())
        n = len(g); tab[(var, r)] = (both, a_only, r_only, neither, n)
        t = pa[(pa.variable == var) & (pa.r == r)].iloc[0]
        p.check(f"{var} r={r}: counts sum to n = {n}", both + a_only + r_only + neither == n)
        p.check(f"{var} r={r}: (flagged then not) + (not then flagged) = {a_only + r_only} equals the Part A flip count {int(t.n_flips)}", a_only + r_only == int(t.n_flips))
        p.near(f"{var} r={r}: flip rate {(a_only + r_only) / n:.6f} equals the stored flip rate", (a_only + r_only) / n, t.flip_rate, 1e-12)
        p.check(f"{var} r={r}: share flagged with all evidence equals share_baseline_ge_tau", abs((both + a_only) / n - t.share_baseline_ge_tau) < 1e-12)
        if (var, r) in LEDID:
            lid = LEDID[(var, r)]; disp = LED.loc[lid, "display"]
            p.check(f"{var} r=10: flip rate printed {100*(a_only+r_only)/n:.1f}% equals ledger {lid} ({disp.split(' ')[0]})", f"{100*(a_only+r_only)/n:.1f}%" == disp.split(" ")[0])
        nm = {"both": "flagged with all evidence and flagged after retracting the variable", "a_only": "flagged with all evidence, not flagged after retracting the variable",
              "r_only": "not flagged with all evidence, flagged after retracting the variable", "neither": "not flagged with all evidence and not flagged after retracting the variable"}
        for key, val in (("both", both), ("a_only", a_only), ("r_only", r_only), ("neither", neither)):
            p.row("selection", f"Flag transition, Application BN, {var} retracted, r={r} (threshold 1/{r+1}): test rows {nm[key]}", val, f"{val:,}", "rows", SRCP,
                  f"variable={var} (bin) | tau=1/(1+{r}) | baseline_prob >= tau vs retracted_prob >= tau | cell={key}", derived="Y", notes=f"2x2 table over the {n:,} test rows with defined baseline and retracted probability; sum of the two off-diagonal cells equals the stored Part A flip count {int(t.n_flips)} ({SRCF})",
                  display_rule="int", n=n, model="BN", variable=var, level=f"r={r}", caveat="C3;C9")
# per-r summary
sumr = {}
for r in RS:
    tau = 1.0 / (1 + r)
    ncell = int((cells.p_default_em.values >= tau - 1e-15).sum()); fl = int((pbn >= tau).sum())
    sumr[r] = (tau, ncell, fl)
    p.check(f"r={r}: BN flagged test rows {fl:,} equals the stored `flagged` in sel_decision_value_combined.csv", fl == int(comb.loc[r, "flagged"]))
    p.row("selection", f"Number of the 25 Age x Debt Ratio cells with probability at or above the flag threshold 1/(1+r), r={r}", ncell, str(ncell), "cells", "selection/tables/sel_default_cells.csv", f"p_default_em >= 1/(1+{r}) | count over 25 cells", derived="Y", notes="threshold 1/%d = %.4f; probabilities are the EM lookup-table values (N-FIG-001 to 025)" % (r + 1, tau), display_rule="int", n=25, model="BN", level=f"r={r}", caveat="C3")
    p.row("selection", f"Test rows the BN flags at r={r} (threshold 1/{r+1})", fl, f"{fl:,}", "rows", "selection/tables/sel_decision_value_combined.csv", f"contribution=selection | model=BN | r={r} | flagged", derived="N", notes="stored column `flagged`, recomputed from sel_bn_test_predictions.pkl as sum(p >= 1/(1+r)) and equal", display_rule="int", n=30000, model="BN", level=f"r={r}", caveat="C3")
p.check("r=5: 0 cells at or above 1/6 and 0 flagged rows (ledger N-SEL-031 = 0)", sumr[5][1] == 0 and sumr[5][2] == 0 == int(float(LED.loc["N-SEL-031", "value"])))
p.check("ledger thresholds N-SEL-027 and N-SEL-028 equal 1/6 and 1/11 to 4 dp", round(1/6, 4) == round(lv := float(LED.loc["N-SEL-027", "value"]), 4) and round(1/11, 4) == round(float(LED.loc["N-SEL-028", "value"]), 4))
md = ["## J5. Flag-transition tables (Application BN)", "",
      "**Method.** For each of Age and Debt Ratio and each r, the flag with all evidence is `baseline_prob >= 1/(1+r)` and the flag after retracting the variable is `retracted_prob >= 1/(1+r)` (the BN posteriors by exact variable elimination, saved per test row in `%s`; `tau_grid` in the file equals 1/(1+r)). The 2x2 counts cross the two flags over the 29,998 test rows for which both posteriors are defined (the population of the Part A table; 2 test rows have zero-probability retracted evidence and are excluded; Age and Debt Ratio are always observed, so every other row is included). The same rows give the stored Part A flip rates." % SRCP, "",
      "**Data.** `%s`; `%s`; `selection/tables/sel_default_cells.csv`; `selection/tables/sel_decision_value_combined.csv`; `selection/predictions/sel_bn_test_predictions.pkl`." % (SRCP, SRCF), ""]
for var in V:
    md += [f"**{var} retracted**", "", "| r | threshold | both flagged | flagged, then not | not, then flagged | neither | n | flip rate |", "|---|---|---|---|---|---|---|---|"]
    for r in RS:
        b, a_, r_, ne, n = tab[(var, r)]; md.append(f"| {r} | 1/{r+1} = {1/(1+r):.4f} | {b:,} | {a_:,} | {r_:,} | {ne:,} | {n:,} | {100*(a_+r_)/n:.1f}% |")
    md.append("")
md += ["**Per r (Application BN)**", "", "| r | threshold 1/(1+r) | cells of 25 at or above it | test rows flagged (of 30,000) |", "|---|---|---|---|"] + [f"| {r} | {sumr[r][0]:.4f} | {sumr[r][1]} | {sumr[r][2]:,} |" for r in RS] + ["", checks_md(p), "",
       "**Notes.** (a) At r = 5 no cell reaches 1/6 (highest cell 0.1194), so nothing is flagged and there are no transitions: all 29,998 rows are in the 'neither' cell. (b) The flagged-then-not and not-then-flagged counts show the direction of the flips, which the flip rate alone does not; the flip rate equals their sum over n. (c) 'Test rows flagged' counts all 30,000 test rows (BN defined on all of them at Application).", ""]
p.write("\n".join(md))
