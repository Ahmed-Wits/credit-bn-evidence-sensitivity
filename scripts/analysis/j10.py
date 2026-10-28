"""J10: flip rates at tau_r = 1/(1+r), r = 1..20, both stages, with the existing bootstrap (sensitivity.boot_means: RandomState(42), B = 2000, one shared row-resample sequence)."""
from addcommon import *
import pickle
sys.path.insert(0, R + "/shared/src")
import importlib.util
spec = importlib.util.spec_from_file_location("sens", R + "/shared/src/sensitivity.py"); sens = importlib.util.module_from_spec(spec); spec.loader.exec_module(sens)
p = Part("j10")
RS = list(range(1, 21))
# ------------- data
per = pickle.load(open(R + "/selection/predictions/sel_rq2_per_row.pkl", "rb"))["per_row"]
pa = pd.read_csv(R + "/selection/tables/sel_voi_partA_table.csv")
ga = per[per.variable == "Age (bin)"].set_index("row_index"); gd = per[per.variable == "Debt Ratio (bin)"].set_index("row_index")
keep = ga.index[ga.retracted_prob.notna().values & gd.retracted_prob.reindex(ga.index).notna().values]
assert len(keep) == 29998
APP = {"Age": ga.loc[keep], "Debt Ratio": gd.loc[keep]}
mon = pd.read_pickle(R + "/full_variable/per_row/vo_partA_retraction_per_row.pkl")
MV = ["Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due"]
MON = {v: mon[mon.variable == v + " (bin)"].reset_index(drop=True) for v in MV}
for v in MV: assert len(MON[v]) == 29998 and MON[v].retracted_prob.notna().all() and MON[v].baseline_prob.notna().all()
tabM = pd.read_csv(R + "/full_variable/tables/rq2_ext_partA_table.csv").set_index("variable")
def flips(base, ret, tau): return ((base >= tau) != (ret >= tau)).astype(float)
def boot(cols):
    bm = sens.boot_means([np.column_stack(cols)], B=2000, seed=42)[0]
    return np.percentile(bm, [2.5, 97.5], axis=0)
# ------------- GATES
cols = []; keys = []
for var in ("Age", "Debt Ratio"):
    for r in (5, 10, 15, 20):
        cols.append(flips(APP[var].baseline_prob.to_numpy(), APP[var].retracted_prob.to_numpy(), 1 / (1 + r))); keys.append((var, r))
ci = boot(cols)
for j, (var, r) in enumerate(keys):
    t = pa[(pa.variable == var) & (pa.r == r)].iloc[0]
    p.gate_near(f"Application {var} r={r}: flip rate equals the stored Table 3 value ({100*t.flip_rate:.2f}%)", cols[j].mean(), t.flip_rate, 1e-12)
    p.gate_near(f"Application {var} r={r}: bootstrap lower bound equals the stored one", ci[0, j], t.flip_ci_lo, 1e-12)
    p.gate_near(f"Application {var} r={r}: bootstrap upper bound equals the stored one", ci[1, j], t.flip_ci_hi, 1e-12)
IDF = {"Credit Utilisation Rate": "N-MON-017", "30-59 Days Past Due": "N-MON-019", "90+ Days Late": "N-MON-021", "60-89 Days Past Due": "N-MON-023"}
colsm = [flips(MON[v].baseline_prob.to_numpy(), MON[v].retracted_prob.to_numpy(), 0.17) for v in MV]; cim = boot(colsm)
for j, v in enumerate(MV):
    t = tabM.loc[v + " (bin)"]
    p.gate_near(f"After acceptance {v}: flip rate at the locked 0.17 equals Table 5 ({100*t.flip_rate:.2f}%, ledger {IDF[v]})", colsm[j].mean(), t.flip_rate, 1e-12)
    p.gate(f"After acceptance {v}: printed rate {100*colsm[j].mean():.2f}% equals ledger {IDF[v]}", f"{100*colsm[j].mean():.2f}%" == LED.loc[IDF[v], "display"].split(" ")[0])
    p.gate_near(f"After acceptance {v}: bootstrap lower bound at 0.17 equals the stored one", cim[0, j], t.flip_ci_low, 1e-12)
    p.gate_near(f"After acceptance {v}: bootstrap upper bound at 0.17 equals the stored one", cim[1, j], t.flip_ci_high, 1e-12)
# ------------- new numbers
recs = []
for stage, D, names in (("Application", APP, ["Age", "Debt Ratio"]), ("After acceptance", MON, MV)):
    for v in names:
        base, ret = D[v].baseline_prob.to_numpy(), D[v].retracted_prob.to_numpy()
        cs = [flips(base, ret, 1 / (1 + r)) for r in RS]; cc = boot(cs)
        for k, r in enumerate(RS):
            recs.append(dict(stage=stage, variable=v, r=r, tau=1 / (1 + r), flip_rate=float(cs[k].mean()), lower=float(cc[0, k]), upper=float(cc[1, k]), n_rows=len(base), n_flips=int(cs[k].sum()),
                             share_flagged_all_evidence=float((base >= 1 / (1 + r)).mean())))
F = pd.DataFrame(recs); F.to_csv(ADD + "/flip_rate_by_r.csv", index=False)
# the r=5,10,15,20 values for Application equal the gate values (already asserted); r=5,10 After acceptance extra comparison
def f_(stage, v, r): return F[(F.stage == stage) & (F.variable == v) & (F.r == r)].iloc[0]
# ------------- flagged counts of the After-acceptance BN
bb = pickle.load(open(R + "/full_variable/predictions/bn_test_predictions.pkl", "rb"))["bn_test_probs"]; pm = bb.dropna()
p.check("BN defined on 29,998 test rows", len(pm) == 29998)
flag = {t: int((pm >= t).sum()) for t in (1 / 6, 0.17, 1 / 11)}
comb = pd.read_csv(R + "/selection/tables/sel_decision_value_combined.csv"); cm = comb[(comb.contribution == "monitoring") & (comb.model == "BN")].set_index("r")
p.check("BN flagged rows at 1/6 and 1/11 equal the stored `flagged` of sel_decision_value_combined.csv (r = 5, 10)", flag[1 / 6] == int(cm.loc[5, "flagged"]) and flag[1 / 11] == int(cm.loc[10, "flagged"]), f"{flag[1/6]:,} / {flag[1/11]:,}")
p.check("ledger N-MON-045 (BN saving at r=5 flags 3,257 rows) agrees", "3257" in LED.loc["N-MON-045", "quantity"] and flag[1 / 6] == 3257)
# ------------- ledger rows
SRC = "Submissions/4. Article/WRITEUP/analysis_additions/flip_rate_by_r.csv"
CIM = "95% percentile bootstrap, 2000 resamples, seed 42 (sensitivity.boot_means: RandomState(42), one shared row-resample sequence for all matrices; same procedure as the stored flip-rate intervals)"
for stage, st_led, names in (("Application", "selection", ["Age", "Debt Ratio"]), ("After acceptance", "monitoring", MV)):
    for v in names:
        for r in (5, 10, 15, 20):
            x = f_(stage, v, r)
            p.row(st_led, f"Flip rate at r={r} (threshold 1/{r+1}) when the variable is retracted, {v}, {stage}, with 95% interval", float(x.flip_rate),
                  f"{100*x.flip_rate:.2f}% [{100*x.lower:.2f}%, {100*x.upper:.2f}%]" if st_led == "monitoring" else f"{100*x.flip_rate:.1f}% [{100*x.lower:.1f}%, {100*x.upper:.1f}%]", "%", SRC,
                  f"stage={stage} | variable={v} | r={r} | tau=1/(1+r) | flip_rate (code/j10.py)", derived="Y", ci_lo=float(x.lower), ci_hi=float(x.upper), ci_method=CIM, n=int(x.n_rows), model="BN", variable=v, level=f"r={r}",
                  caveat="C3;C7;C9", notes=f"flip = (baseline p >= 1/(1+r)) != (p with the variable retracted >= 1/(1+r)); {int(x.n_flips):,} of {int(x.n_rows):,} rows; threshold fixed in advance at both stages ({'iteration-79 BN' if st_led=='monitoring' else 'application BN'}); gate: Application r=5,10,15,20 equal Table 3 and the stored intervals, After acceptance at the locked 0.17 equals Table 5",
                  display_rule=("pct2" if st_led == "monitoring" else "pct1"))
for t, lab, loc in ((1 / 6, "1/6", "bn_test_probs >= 1/6"), (0.17, "0.17 (the earlier locked threshold)", "bn_test_probs >= 0.17"), (1 / 11, "1/11", "bn_test_probs >= 1/11")):
    p.row("monitoring", f"Test rows the After-acceptance BN flags at threshold {lab}", flag[t], f"{flag[t]:,}", "rows", "full_variable/predictions/bn_test_predictions.pkl", loc, derived="Y",
          notes="count over the 29,998 test rows where the BN is defined; 1/6 and 1/11 equal the stored `flagged` of sel_decision_value_combined.csv (r = 5, 10); 0.17 is the locked threshold chosen on the test set (not used in the revised article)", display_rule="int", n=29998, model="BN", level=lab, caveat="C1;C3")
# comparison table of the three thresholds
cmp_rows = []
for v in MV:
    base, ret = MON[v].baseline_prob.to_numpy(), MON[v].retracted_prob.to_numpy()
    cmp_rows.append((v, [flips(base, ret, t).mean() for t in (1 / 6, 0.17, 1 / 11)]))
def tbl(stage, names, rs=(5, 10, 15, 20)):
    L = ["| Variable | " + " | ".join(f"r = {r}" for r in rs) + " |", "|---|" + "---|" * len(rs)]
    for v in names: L.append(f"| {v} | " + " | ".join(f"{100*f_(stage, v, r).flip_rate:.2f}% [{100*f_(stage, v, r).lower:.2f}, {100*f_(stage, v, r).upper:.2f}]" for r in rs) + " |")
    return L
md = ["## J10. Flip rates at tau_r = 1/(1+r), both stages", "",
      "**Method.** Flip = (posterior with all evidence >= 1/(1+r)) differs from (posterior after retracting the variable >= 1/(1+r)), over the 29,998 test rows for which both posteriors are defined (Application: `selection/predictions/sel_rq2_per_row.pkl`; After acceptance: `full_variable/per_row/vo_partA_retraction_per_row.pkl`, the stored iteration-79 BN). Intervals: the bootstrap used for the existing flip-rate intervals, `shared/src/sensitivity.py::boot_means` (`numpy.random.RandomState(42)`, B = 2,000 resamples of the 29,998 test rows by `randint`, the same resample sequence for every column; 95% percentile interval of the resampled mean). Because the resample sequence depends only on n and the seed, the intervals are on exactly the same resamples as Tables 3 and 5.", "",
      "**Gate (tolerance 1e-12): PASSED.** Application r = 5, 10, 15, 20 for Age and Debt Ratio: flip rate, lower and upper bound equal the stored Table 3 values (`sel_voi_partA_table.csv`). After acceptance at the locked 0.17: flip rate, lower and upper bound equal Table 5 / `rq2_ext_partA_table.csv` (4.58, 4.63, 3.09, 1.40 percent; ledger N-MON-017, 019, 021, 023 reproduced as printed).", "",
      "**Output.** `flip_rate_by_r.csv` (stage, variable, r, tau, flip_rate, lower, upper, n_rows, n_flips, share_flagged_all_evidence; 120 rows = 6 variables x 20 r). Ledger rows for r = 5, 10, 15, 20 at both stages (24 rows) and the three flagged counts (3 rows).", "",
      "### Application (Table 3 replacement; flip rate [95% interval])", "", *tbl("Application", ["Age", "Debt Ratio"]), "",
      "### After acceptance at 1/(1+r) (replacement for Table 5; flip rate [95% interval])", "", *tbl("After acceptance", MV), "",
      "### After acceptance: the same flip rate at three thresholds", "", "| Variable | at 1/6 (r = 5) | at 0.17 (locked, Table 5) | at 1/11 (r = 10) |", "|---|---|---|---|"] + \
     [f"| {v} | {100*a:.2f}% | {100*b:.2f}% | {100*c:.2f}% |" for v, (a, b, c) in cmp_rows] + \
     ["", "Test rows flagged by the After-acceptance BN: **%s at 1/6, %s at 0.17, %s at 1/11** (of 29,998)." % (f"{flag[1/6]:,}", f"{flag[0.17]:,}", f"{flag[1/11]:,}"), "", checks_md(p), "",
      "**Notes.** (a) The flip rate is a necessary-condition quantity (a flip needs one posterior at or above and one below the threshold); at Application it is 0 for r = 1 to 7 because no cell reaches 1/(1+r) for r <= 7. (b) 'Observed' rows: the blanket variables are always observed at both stages, so every row with defined posteriors is used. (c) No new ledger `stage` values; source_locator names the CSV columns.", ""]
p.write("\n".join(md))
