"""J2: paired test-row bootstrap intervals for the Fig 4 cost-saving pipeline (both stages)."""
from addcommon import *
import pickle
sys.path.insert(0, R + "/shared/src")
import decision_value as dv
p = Part("j2")
RS = list(range(1, 21))
CSV = "selection/tables/sel_decision_value_combined.csv"
comb = pd.read_csv(R + "/" + CSV)
sp = pickle.load(open(R + "/Data/processed/split_indices.pkl", "rb")); y_tr = np.asarray(sp["y_train"]).astype(int); y_te = np.asarray(sp["y_test"]).astype(int)

# ---------------- rebuild predictions exactly as in 08_story_extras cell index 2 (saved outputs only)
sb = pickle.load(open(R + "/selection/predictions/sel_bn_test_predictions.pkl", "rb")); bp = pickle.load(open(R + "/selection/predictions/sel_baseline_test_predictions.pkl", "rb"))
ys = np.asarray(sb["y"]).astype(int); assert np.array_equal(ys, y_te)
Ps = {"BN": np.asarray(sb["p"], float), "LR": np.asarray(bp["p_lr"], float), "RF": np.asarray(bp["p_rf"], float)}
oofm = pickle.load(open(R + "/monitoring_rq3/mon_oof_predictions.pkl", "rb"))["oof"]
bbp = pickle.load(open(R + "/full_variable/predictions/bn_test_predictions.pkl", "rb"))["bn_test_probs"]; ab = pickle.load(open(R + "/full_variable/predictions/ablation_predictions.pkl", "rb"))
v = bbp.notna().to_numpy(); assert int(v.sum()) == 29998
ym = y_te[v]; Pm = {"BN": bbp.to_numpy(float)[v], "LR": np.asarray(ab["LR_all"], float)[v], "RF": np.asarray(ab["RF_all"], float)[v]}
STAGES = {"selection": dict(y=ys, P=Ps, label="Application", n=len(ys), oof={"LR": bp["oof_lr"], "RF": bp["oof_rf"]}),
          "monitoring": dict(y=ym, P=Pm, label="After acceptance", n=len(ym), oof={"LR": oofm["LR"], "RF": oofm["RF"]})}

def taus_for(st):
    d = comb[comb.contribution == st]
    T = {m: {int(r): float(d[(d.model == m) & (d.r == r)].tau.iloc[0]) for r in RS} for m in ("BN", "LR", "RF")}
    return T

res = {}
for st, S in STAGES.items():
    T = taus_for(st); y, P = S["y"], S["P"]
    # (1) thresholds: recompute from OOF train predictions and compare with the stored tau column
    ok_t = True
    for m in ("LR", "RF"):
        for r in RS:
            t, *_ = dv.oof_tau(S["oof"][m], y_tr, r, "highest"); ok_t &= abs(t - T[m][r]) < 1e-12
    for r in RS: ok_t &= abs(T["BN"][r] - 1.0 / (1 + r)) < 1e-12
    p.check(f"{S['label']}: LR/RF thresholds recomputed from saved OOF predictions equal the stored tau column (r = 1..20), BN = 1/(1+r)", ok_t)
    cur = dv.curve(y, P, T, RS)
    # (2) point estimates equal the stored csv
    d = comb[comb.contribution == st]
    m_ = cur.merge(d, on=["r", "model"], suffixes=("", "_csv"))
    p.check(f"{S['label']}: recomputed cost, flagged count and saving equal sel_decision_value_combined.csv for all 60 (model, r) cells",
            len(m_) == 60 and (m_.cost == m_.cost_csv).all() and (m_.flagged == m_.flagged_csv).all() and np.allclose(m_.saving, m_.saving_csv, atol=1e-15), "max |saving diff| = %.1e" % np.abs(m_.saving - m_.saving_csv).max())
    # (3) bootstrap
    n = len(y); W = boot_weights(n)   # B x n
    D = W @ y.astype(np.float32)      # defaulters in each resample
    sav = {}
    for m in ("BN", "LR", "RF"):
        s_ = np.empty((B_BOOT, len(RS)))
        for j, r in enumerate(RS):
            fl = P[m] >= T[m][r]
            fn = W @ ((y == 1) & ~fl).astype(np.float32); fp = W @ ((y == 0) & fl).astype(np.float32)
            cost = r * fn + fp; cheaper = np.minimum(r * D, n - D)
            s_[:, j] = 1.0 - cost / cheaper
        sav[m] = s_
    res[st] = dict(sav=sav, cur=cur, n=n, point={(m, r): float(cur[(cur.model == m) & (cur.r == r)].saving.iloc[0]) for m in ("BN", "LR", "RF") for r in RS})

def pt(st, m, r): return res[st]["point"][(m, r)]
def sav_ci(st, m, r): x = 100 * res[st]["sav"][m][:, r - 1]; return pct_ci(x)
def diff_ci(st, a, b, r): x = 100 * (res[st]["sav"][a][:, r - 1] - res[st]["sav"][b][:, r - 1]); return pct_ci(x)
def diff_pt(st, a, b, r): return 100 * (pt(st, a, r) - pt(st, b, r))

# reproduction against existing ledger rows (point estimates)
IDS = {("selection", "BN", 5): "N-SEL-042", ("selection", "LR", 5): "N-SEL-043", ("selection", "RF", 5): "N-SEL-044", ("monitoring", "BN", 5): "N-MON-045", ("monitoring", "LR", 5): "N-MON-046", ("monitoring", "RF", 5): "N-MON-047"}
for st, base in (("selection", 0), ("monitoring", 9)):
    for k, r in enumerate((10, 15, 20)):
        for mi, m in enumerate(("BN", "LR", "RF")): IDS[(st, m, r)] = "N-XST-%03d" % (base + 3 * k + mi + 1)
PEAK = {("selection", "BN"): "N-SEL-061", ("selection", "LR"): "N-SEL-062", ("selection", "RF"): "N-SEL-063", ("monitoring", "BN"): "N-MON-061", ("monitoring", "LR"): "N-MON-062", ("monitoring", "RF"): "N-MON-063"}
for (st, m, r), lid in IDS.items():
    disp = LED.loc[lid, "display"]; dec = len(disp.rstrip("%").split(".")[1]) if "." in disp else 0
    p.check(f"reproduce ledger {lid}: {STAGES[st]['label']} {m} saving at r={r}", abs(round(100 * pt(st, m, r) + 1e-12, dec) - float(disp.rstrip('%'))) < 1e-9, f"{100*pt(st,m,r):.4f}% vs {disp}")
for (st, m), lid in PEAK.items():
    disp = LED.loc[lid, "display"]
    p.check(f"reproduce ledger {lid}: {STAGES[st]['label']} {m} peak saving at r=14", abs(round(100 * pt(st, m, 14), 2) - float(disp.rstrip('%'))) < 1e-9, f"{100*pt(st,m,14):.4f}% vs {disp}")
NAMED = [("selection", "RF", 8, "N-SEL-058", -7.49), ("selection", "LR", 10, "N-SEL-057", -6.24), ("monitoring", "RF", 3, "N-MON-057", 0.83), ("monitoring", "LR", 2, "N-MON-059", -1.92), ("monitoring", "RF", 11, "N-MON-060", -1.81)]
for st, c, r, lid, art in NAMED:
    dpt = diff_pt(st, "BN", c, r)
    p.check(f"reproduce article/ledger {lid}: {STAGES[st]['label']} BN minus {c} at r={r} = {art:+.2f} points", abs(round(dpt, 2) - art) < 1e-9 and abs(round(dpt, 2) - float(LED.loc[lid, 'display'].replace(MINUS, '-').replace(' pts', ''))) < 1e-9, f"{dpt:+.4f} vs ledger {LED.loc[lid,'display']}")
# the named extremes are extremes over r = 1..20 of the point differences
for st, c, r, lid, art in NAMED:
    allr = {rr: diff_pt(st, "BN", c, rr) for rr in RS}
    ext = min(allr, key=allr.get) if art < 0 else max(allr, key=allr.get)
    p.check(f"{STAGES[st]['label']} BN minus {c}: r={r} is the {'minimum' if art < 0 else 'maximum'} over r = 1..20", ext == r, f"extreme at r={ext}")

NOTE = "paired test-row bootstrap (B = 2000, default_rng(42) multinomial resample counts, same resample for every model and r); predictions and thresholds fixed; cheaper trivial policy re-evaluated within each resample; 95% percentile interval"
CIM = "95% percentile interval, 2000 test-row bootstrap resamples (seed 42), paired across models; predictions and thresholds fixed"
SRCF = "selection/tables/sel_decision_value_combined.csv"
out_md = []
def sav_row(st, m, r, rule, dec):
    lo, hi = sav_ci(st, m, r); x = 100 * pt(st, m, r)
    p.row("cross-stage", f"Cost saving vs the cheaper trivial policy with 95% interval, {m}, {STAGES[st]['label']}, r={r}", pt(st, m, r), f"{x:.{dec}f}% [{lo:.{dec}f}%, {hi:.{dec}f}%]", "%", SRCF,
          f"contribution={st} | model={m} | r={r} | saving | bootstrap interval (code/j2.py)", derived="Y", ci_lo=lo / 100, ci_hi=hi / 100, ci_method=CIM, n=res[st]["n"], model=m, level=f"r={r}", caveat="C3;C5",
          notes="point estimate equals the stored saving; interval from j2.py; " + NOTE, display_rule=rule)
    return x, lo, hi
def diff_row(st, a, b, r, ctx=""):
    lo, hi = diff_ci(st, a, b, r); x = diff_pt(st, a, b, r)
    tested = "yes-tie" if lo <= 0 <= hi else "yes-significant"
    p.row("cross-stage", f"Cost saving difference {a} minus {b} with 95% interval, {STAGES[st]['label']}, r={r} (percentage points){ctx}", x, f"{sgn(x,2,True)} [{sgn(lo,2)}, {sgn(hi,2)}]", "percentage points", SRCF,
          f"contribution={st} | model={a},{b} | r={r} | saving difference | bootstrap interval (code/j2.py)", derived="Y", ci_lo=lo, ci_hi=hi, ci_method=CIM, n=res[st]["n"], model=a, comparator=b, level=f"r={r}", tested=tested, caveat="C3;C5",
          notes="point estimate equals the stored saving difference; " + NOTE + ("; interval %s 0" % ("includes" if tested == "yes-tie" else "excludes")), display_rule="txt")
    return x, lo, hi
tab1 = ["| Stage | r | BN | LR | RF |", "|---|---|---|---|---|"]
for st in ("selection", "monitoring"):
    for r in (5, 10):
        cells = []
        for m in ("BN", "LR", "RF"):
            x, lo, hi = sav_row(st, m, r, "pct1", 1); cells.append(f"{x:.1f}% [{lo:.1f}%, {hi:.1f}%]")
        tab1.append(f"| {STAGES[st]['label']} | {r} | " + " | ".join(cells) + " |")
tab2 = ["| Stage | r | BN minus LR | BN minus RF |", "|---|---|---|---|"]
for st in ("selection", "monitoring"):
    for r in (5, 10):
        cells = []
        for c in ("LR", "RF"):
            x, lo, hi = diff_row(st, "BN", c, r); cells.append(f"{sgn(x,2,True)} [{sgn(lo,2)}, {sgn(hi,2)}]")
        tab2.append(f"| {STAGES[st]['label']} | {r} | " + " | ".join(cells) + " |")
tab3 = ["| Stage | Difference | r | Point (points) | 95% interval | Interval excludes 0? |", "|---|---|---|---|---|---|"]
for st, c, r, lid, art in NAMED:
    x, lo, hi = diff_row(st, "BN", c, r, " (named extreme in the article)")
    tab3.append(f"| {STAGES[st]['label']} | BN minus {c} | {r} | {sgn(x,2)} | [{sgn(lo,2)}, {sgn(hi,2)}] | {'no' if lo <= 0 <= hi else 'yes'} |")
tab4 = ["| Stage | Model | Peak saving at r = 14 | 95% interval |", "|---|---|---|---|"]
for st in ("selection", "monitoring"):
    for m in ("BN", "LR", "RF"):
        x, lo, hi = sav_row(st, m, 14, "pct2", 2); tab4.append(f"| {STAGES[st]['label']} | {m} | {x:.2f}% | [{lo:.2f}%, {hi:.2f}%] |")
# bootstrap sanity: interval contains the point estimate for every row
inside = []
for r in p.rows:
    v_ = float(r["value"]); lo_ = float(r["ci_lo"]); hi_ = float(r["ci_hi"])
    inside.append(lo_ - 1e-9 <= v_ <= hi_ + 1e-9)
p.check("all %d point estimates lie inside their 95%% intervals" % len(inside), all(inside))
md = ["## J2. Cost-saving intervals (Fig 4 pipeline)", "",
      "**Method.** Same pipeline as Fig 4 (08_story_extras cell index 2, `shared/src/decision_value.py`): flag = p >= tau; cost = r x FN + FP; saving = 1 - cost / cost of the cheaper trivial policy (accept-all r x D, reject-all N - D). Thresholds are the stored `tau` column of `sel_decision_value_combined.csv`: BN 1/(1+r); LR and RF the out-of-fold-train cost-minimising value on the 0.01 grid, ties to the highest (selection: out-of-fold predictions in `sel_baseline_test_predictions.pkl`; After acceptance: `monitoring_rq3/mon_oof_predictions.pkl`). After acceptance uses the 29,998 test rows where the BN is defined (BN test probabilities from `full_variable/predictions/bn_test_predictions.pkl`, LR/RF from `ablation_predictions.pkl` keys LR_all, RF_all). Intervals: test rows resampled with replacement (B = 2,000, `numpy.random.default_rng(42)`, multinomial resample counts, the same resample for every model and every r), predictions and thresholds fixed, the cheaper trivial policy re-evaluated inside each resample, 95% percentile interval. No model, threshold or OOF prediction was refitted: thresholds were first recomputed from the saved OOF predictions and shown to equal the stored ones.", "",
      "**Data.** %s; `Data/processed/split_indices.pkl` (y_train, y_test); predictions pickles listed above." % SRCF, "",
      "### (i) Saving with 95% interval", "", *tab1, "", "### (ii) Differences in saving, BN minus LR and BN minus RF (percentage points)", "", *tab2, "",
      "### (iii) The article's named extremes", "", *tab3, "", "### (iv) Peak savings at r = 14", "", *tab4, "", checks_md(p), "",
      "**Deviations / notes.** (a) The bootstrap draws resample counts with `default_rng(42).multinomial` (the convention of the Phase 8 test-row bootstrap, `10_continuous_ref_and_rq3_ci` cell 7), not `RandomState(42).randint` as in `sensitivity.paired_bootstrap`; both are 2,000 test-row resamples with seed 42. (b) The cheaper trivial policy is recomputed inside every resample (accept-all at small r, reject-all above r* = 13.96/13.98), as the saving is defined relative to it. (c) For the differences the intervals use the paired resample; they are intervals for the test-set saving difference with models and thresholds fixed, so they do not include retraining variability and, for LR and RF After acceptance, do not include the test-set selection of the locked thresholds. (d) In the ledger rows the saving is stored as a fraction (as in the existing rows) and differences as percentage points; `ci_lo`/`ci_hi` follow the same units as `value`. (e) The five named extremes were chosen as the extreme of the point difference over r = 1..20, so their intervals are not corrected for that selection; read them descriptively. At Application r = 5 the BN and LR both flag no row, so their saving and the difference are exactly 0 in every resample (degenerate interval). (f) The ledger `stage` is `cross-stage` as for the existing saving rows; the stage is in the quantity text and in `source_locator`.", ""]
p.write("\n".join(md))
