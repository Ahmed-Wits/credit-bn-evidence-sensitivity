from addcommon import *
p = Part("j1")
SRC = "Submissions/4. Article/WRITEUP/figures_final/code/s7_em_history.csv"
import pickle
h = pd.read_csv(R + "/" + SRC)
d = pickle.load(open(R + "/full_variable/models/bn_final_model_iteration79.pkl", "rb"))
tr, te = h.train_ll.values, h.test_ll.values
# reproduction checks against stored values
p.check("100 iterations in the history", len(h) == 100)
p.near("train LL at iteration 79 equals stored value", tr[78], d["train_ll_at_peak"], 1e-3, "rerun %.6f stored %.6f" % (tr[78], d["train_ll_at_peak"]))
p.near("test LL at iteration 79 equals stored value", te[78], d["test_ll_at_peak"], 1e-3, "rerun %.6f stored %.6f" % (te[78], d["test_ll_at_peak"]))
gtr = np.diff(tr); gte = np.diff(te)   # gain at iteration k+1 = LL(k+1) - LL(k)
min_gain = float(gtr.min()); nondec = bool((gtr >= 0).all())
first_small = next((i + 2 for i, g in enumerate(gtr) if g < 1e-4), None)
am = int(np.argmax(te)) + 1
rows = []
def mk(q, v, disp, unit, loc, notes="", rule="txt"): rows.append(dict(q=q, v=v, disp=disp, unit=unit, loc=loc, notes=notes, rule=rule))
mk("After-acceptance EM rerun: train log-likelihood is non-decreasing over iterations 1 to 100", str(nondec), str(nondec), "True/False", "all(diff(train_ll) >= 0)", "S7 rerun (code/s7_em_rerun.py); no early stop, no tolerance")
mk("After-acceptance EM rerun: minimum per-iteration gain in train log-likelihood (iterations 2 to 100)", min_gain, f"{min_gain:.6f}", "log-likelihood units", "min(diff(train_ll)) over iterations 2..100", "attained at iteration %d" % (int(np.argmin(gtr)) + 2), "d6")
mk("After-acceptance EM rerun: train log-likelihood gain at iteration 79 (LL(79) minus LL(78))", float(gtr[78 - 1]), f"{gtr[77]:.6f}", "log-likelihood units", "train_ll[79] - train_ll[78]", "", "d6")
mk("After-acceptance EM rerun: train log-likelihood gain at iteration 100 (LL(100) minus LL(99))", float(gtr[98]), f"{gtr[98]:.6f}", "log-likelihood units", "train_ll[100] - train_ll[99]", "", "d6")
mk("After-acceptance EM rerun: test log-likelihood gain at iteration 79 (LL(79) minus LL(78))", float(gte[77]), sgn(gte[77], 6), "log-likelihood units", "test_ll[79] - test_ll[78]", "positive: the test LL still rose into iteration 79", "d6")
mk("After-acceptance EM rerun: test log-likelihood gain at iteration 100 (LL(100) minus LL(99))", float(gte[98]), sgn(gte[98], 6), "log-likelihood units", "test_ll[100] - test_ll[99]", "negative: the test LL falls after the peak", "d6")
mk("After-acceptance EM rerun: first iteration at which the train gain falls below 1e-4", (first_small if first_small else "none within 100"), (str(first_small) if first_small else "none within 100"), "iteration", "first k in 2..100 with train_ll[k] - train_ll[k-1] < 1e-4",
   "train gain at iteration 100 is %.6f" % gtr[98], "txt")
mk("After-acceptance EM rerun: test log-likelihood at iteration 100 minus the peak", float(te[99] - te.max()), sgn(te[99] - te.max(), 6), "log-likelihood units", "test_ll[100] - max(test_ll)", "peak = %.6f at iteration %d" % (te.max(), am), "d6")
mk("After-acceptance EM rerun: iteration with the highest test log-likelihood", am, str(am), "iteration", "argmax(test_ll) + 1", "equals the iteration used (79)", "int")
for r in rows:
    p.row("monitoring", r["q"], r["v"], r["disp"], r["unit"], SRC, r["loc"], derived="Y", notes=(r["notes"] or "statistic of the 100-iteration S7 rerun history; no early stop"), display_rule=r["rule"], n=100)
p.check("argmax of the test log-likelihood is 79", am == 79)
p.check("test log-likelihood gain at 79 is positive and at 80 negative", gte[77] > 0 and gte[78] < 0, "gain(79) %.6f, gain(80) %.6f" % (gte[77], gte[78]))
md = ["## J1. EM history statistics (S7 rerun)", "",
      "**Method.** Statistics of the per-iteration train and test log-likelihood of the 100-iteration After-acceptance EM rerun (`figures_final/code/s7_em_history.csv`, produced by `figures_final/code/s7_em_rerun.py`: stored 24-edge structure, ML start on the 96,251 complete rows, no prior, no early stop). Gain at iteration k is LL(k) minus LL(k-1).", "",
      "**Data.** `%s`. Reproduction checks: iteration-79 values equal the stored values in `full_variable/models/bn_final_model_iteration79.pkl` (see table). No ledger row exists for these quantities, so the check is against the stored pickle." % SRC, "",
      "| Statistic | Value |", "|---|---|"] + ["| %s | %s |" % (r["q"].replace("After-acceptance EM rerun: ", ""), r["disp"]) for r in rows] + ["", checks_md(p), "",
      "**Deviations.** None. Note: the minimum per-iteration train gain is positive (%.6f at iteration %d), so the train log-likelihood is strictly increasing over all 100 iterations; the train gain is still above 1e-4 at iteration 100, so the first iteration with gain below 1e-4 is 'none within 100'. The ledger `stage` for these rows is `monitoring` (After acceptance); `unit` for the log-likelihood quantities is a new descriptive label." % (min_gain, int(np.argmin(gtr)) + 2), ""]
p.write("\n".join(md))
