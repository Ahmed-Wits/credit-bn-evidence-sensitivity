"""J6: mean Delta_j next to mean S_exp, row-by-row check Delta_j <= S_exp, ratio of means."""
from addcommon import *
import pickle
p = Part("j6")
P_APP = "selection/predictions/sel_rq2_per_row.pkl"; P_MON = "full_variable/per_row/vo_blanket_original_per_row.pkl"
app = pickle.load(open(R + "/" + P_APP, "rb"))["per_row"]; mon = pd.read_pickle(R + "/" + P_MON)
tabA = pd.read_csv(R + "/selection/tables/sel_voi_partA_table.csv"); tabA = tabA[tabA.r == 10].set_index("variable")
tabM = pd.read_csv(R + "/full_variable/tables/rq2_ext_partA_table.csv"); tabM = tabM.drop_duplicates("variable").set_index("variable")
sensA = pd.read_csv(R + "/selection/tables/sel_rq2_sensitivity_table.csv").set_index("variable")
sensM = pd.read_csv(R + "/full_variable/tables/rq2_sensitivity_table.csv"); sensM = sensM[sensM.measure == "S_exp"].set_index("variable")
IDS = {"Age": "N-SEL-016", "Debt Ratio": "N-SEL-017", "Credit Utilisation Rate": "N-MON-016", "30-59 Days Past Due": "N-MON-018", "90+ Days Late": "N-MON-020", "60-89 Days Past Due": "N-MON-022"}
out = []
_ra = app[app.variable == "Age (bin)"].set_index("row_index").retracted_prob.notna(); _rd = app[app.variable == "Debt Ratio (bin)"].set_index("row_index").retracted_prob.notna()
COMMON = set(_ra.index[_ra.values & _rd.reindex(_ra.index).values])
assert len(COMMON) == 29998
def do(stage, src, df, name, var_key, label, tab, sens, dec):
    tk = var_key if stage == "monitoring" else name
    g = df[df.variable == var_key]; ok = g.S_exp.notna() & g.delta.notna(); g = g[ok]
    delta, sexp = g.delta.values, g.S_exp.values
    viol = float(np.max(delta - sexp)); viol_pos = max(viol, 0.0)
    md_, ms_ = float(delta.mean()), float(sexp.mean()); ratio = md_ / ms_ if ms_ > 0 else float("nan")
    scol = "mean_S_exp" if "mean_S_exp" in sens.columns else "mean"
    p.near(f"{label}: mean S_exp equals the stored table mean ({sens.loc[tk, scol]:.6f})", ms_, sens.loc[tk, scol], 1e-9)
    gm = g[g.row_index.isin(COMMON)] if stage == "selection" else g
    p.near(f"{label}: mean Delta over the Part A rows ({len(gm):,}) equals the stored mean_delta ({tab.loc[tk, 'mean_delta']:.6f})", gm.delta.mean(), tab.loc[tk, "mean_delta"], 1e-9)
    if name in IDS: p.check(f"{label}: mean S_exp printed to the ledger precision equals {IDS[name]} ({LED.loc[IDS[name], 'display'].split(' ')[0]})", f"{ms_:.{dec}f}" == LED.loc[IDS[name], "display"].split(" ")[0], f"{ms_:.{dec+2}f}")
    p.check(f"{label}: Delta_j <= S_exp on all {len(g):,} rows (maximum of Delta_j - S_exp = {viol:+.3e})", viol_pos <= 1e-12)
    out.append(dict(stage=stage, name=name, n=len(g), md=md_, ms=ms_, ratio=ratio, viol=viol, equal=float(np.mean(np.isclose(delta, sexp, rtol=0, atol=1e-12))), src=src))
for nm in ("Age", "Debt Ratio"):
    do("selection", P_APP, app, nm, nm + " (bin)", f"Application {nm}", tabA, sensA, 5)
for nm in ("Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due"):
    do("monitoring", P_MON, mon, nm, nm + " (bin)", f"After acceptance {nm}", tabM, sensM, 4)
# non-blanket Application variables (stored delta = 0 and S_exp = 0)
zero_rows = []
for nm in ("Monthly Income", "Number of Dependents", "Real Estate Loans", "Open Credit Lines"):
    g = app[(app.variable == nm + " (bin)")]; g = g[g.S_exp.notna()]
    p.check(f"Application {nm} (outside the blanket): S_exp = 0 and stored Delta = 0 on all {len(g):,} rows where observed", bool((g.S_exp == 0).all() and (g.delta.fillna(0) == 0).all()))
    zero_rows.append((nm, len(g)))
for o in out:
    q = f"{'Application' if o['stage']=='selection' else 'After acceptance'}"
    st = o["stage"]; nm = o["name"]; dec = 5 if st == "selection" else 4
    note = f"over the {o['n']:,} test rows where {nm} is observed and Delta and S_exp are defined; Delta_j = |P(Y=1 | e minus {nm}) - P(Y=1 | e)|; S_exp = probability-weighted mean shift; Delta_j <= S_exp by the triangle inequality, asserted row by row (max Delta - S_exp = {o['viol']:+.2e}); Delta equals S_exp (1e-12) on {100*o['equal']:.1f}% of rows"
    p.row(st, f"Mean Delta_j (effect of retracting the variable on P(Y=1)), {nm}, {q}", o["md"], f"{o['md']:.{dec}f}", "probability", o["src"], f"variable={nm} (bin) | mean of delta", derived="Y", notes=note, display_rule=("s5" if st == "selection" else "s4"), n=o["n"], model="BN", variable=nm, caveat="C7")
    p.row(st, f"Ratio of mean Delta_j to mean S_exp, {nm}, {q}", o["ratio"], f"{o['ratio']:.3f}", "ratio", o["src"], f"variable={nm} (bin) | mean(delta) / mean(S_exp)", derived="Y", notes=note + "; ratio of the means, not the mean of row ratios", display_rule="d4" if False else "txt", n=o["n"], model="BN", variable=nm, caveat="C7")
    p.row(st, f"Largest violation of Delta_j <= S_exp over test rows (max of Delta_j minus S_exp), {nm}, {q}", o["viol"], f"{o['viol']:+.2e}", "probability", o["src"], f"variable={nm} (bin) | max(delta - S_exp)", derived="Y", notes="expected at most 1e-12 (non-positive up to rounding); " + note, display_rule="sci", n=o["n"], model="BN", variable=nm, caveat="C7")
for nm, n in zero_rows:
    p.row("selection", f"Mean Delta_j, {nm} (outside Default's Markov blanket), Application", 0.0, "0.00000", "probability", P_APP, f"variable={nm} (bin) | mean of delta", derived="Y", notes=f"stored Delta = 0 and S_exp = 0 on all {n:,} rows where the variable is observed; ratio undefined", display_rule="s5", n=n, model="BN", variable=nm, caveat="C7")
md = ["## J6. Delta_j against S_exp", "",
      "**Method.** For each blanket variable, over the test rows where it is observed and both quantities are stored: mean Delta_j = |P(Y=1 | e_-j) - P(Y=1 | e)| (the retraction effect), mean S_exp (the probability-weighted mean shift when the variable is set to each of its states), the ratio of the two means, and the largest row-wise violation of Delta_j <= S_exp. By the triangle inequality (Delta_j is the absolute value of the weighted average of the signed shifts, S_exp the weighted average of their absolute values) Delta_j <= S_exp must hold on every row.", "",
      "**Data.** Application: `%s` (per test row and variable: `delta`, `S_exp`, `baseline_prob`, `retracted_prob`). After acceptance: `%s` (per test row and blanket variable: `delta`, `S_exp`, `S_max`; stored with the original, no-smoothing parameters). The per-row retracted probability P(Y=1 | e_-j) is therefore saved; nothing was re-run. For the eight After-acceptance variables outside the blanket the per-row Delta is not stored (the blanket file holds the four blanket variables only); their S_exp is 0 on every row (ledger N-MON-025 to 028, 051 to 054), so Delta_j <= S_exp forces Delta_j = 0 (d-separation) but it is not independently stored." % (P_APP, P_MON), "",
      "| Stage | Variable | rows | mean Delta_j | mean S_exp | ratio of means | max(Delta - S_exp) | rows with Delta = S_exp |", "|---|---|---|---|---|---|---|---|"]
for o in out:
    dec = 5 if o["stage"] == "selection" else 4
    md.append(f"| {'Application' if o['stage']=='selection' else 'After acceptance'} | {o['name']} | {o['n']:,} | {o['md']:.{dec}f} | {o['ms']:.{dec}f} | {o['ratio']:.3f} | {o['viol']:+.2e} | {100*o['equal']:.1f}% |")
md += ["", "Application variables outside the blanket (Monthly Income, Number of Dependents, Real Estate Loans, Open Credit Lines): stored Delta = 0 and S_exp = 0 on every row where observed; ratio undefined.", "", checks_md(p), "",
       "**Notes.** (a) The maximum violation is reported as `max(Delta - S_exp)`, so a value near zero or negative means the inequality holds; the table shows the actual values. (b) The ratio is the ratio of the means (mean Delta over mean S_exp), not the mean of row ratios. (c) Application uses the 29,998 rows of the Part A table for the Delta means (2 test rows have zero-probability retracted evidence) and every row with a defined S_exp for the S_exp mean, as in the ledger (Age: 29,998 rows, Debt Ratio: 30,000 rows); the table above uses rows where both are defined.", ""]
p.write("\n".join(md))
