"""C1 data: saving and saving differences with 95% paired bootstrap intervals for EVERY integer r = 1..20, both stages (the J2 run stored intervals only for selected r).
Gate: values at r = 5 and r = 10 (savings, differences) equal ledger_additions_v4.csv rows N-ADD-010 to 029; the named-extreme rows N-ADD-030 to 034 reproduce; point estimates equal the stored csv."""
from addcommon import *
import pickle
from costlib import *
p = Part("c1")
RS = list(range(1, 21))
comb = pd.read_csv(R + "/selection/tables/sel_decision_value_combined.csv")
sp = pickle.load(open(R + "/Data/processed/split_indices.pkl", "rb")); y_te = np.asarray(sp["y_test"]).astype(int)
sb = pickle.load(open(R + "/selection/predictions/sel_bn_test_predictions.pkl", "rb")); bp = pickle.load(open(R + "/selection/predictions/sel_baseline_test_predictions.pkl", "rb"))
bbp = pickle.load(open(R + "/full_variable/predictions/bn_test_predictions.pkl", "rb"))["bn_test_probs"]; ab = pickle.load(open(R + "/full_variable/predictions/ablation_predictions.pkl", "rb")); v = bbp.notna().to_numpy()
ST = {"Application": ("selection", np.asarray(sb["y"]).astype(int), {"BN": np.asarray(sb["p"], float), "LR": np.asarray(bp["p_lr"], float), "RF": np.asarray(bp["p_rf"], float)}),
      "After acceptance": ("monitoring", y_te[v], {"BN": bbp.to_numpy(float)[v], "LR": np.asarray(ab["LR_all"], float)[v], "RF": np.asarray(ab["RF_all"], float)[v]})}
v4 = pd.read_csv(ADD + "/ledger_additions_v4.csv", dtype=str, keep_default_na=False).set_index("ledger_id")
allr = []
for stage, (key, y, P) in ST.items():
    T = {m: {r: float(comb[(comb.contribution == key) & (comb.model == m) & (comb.r == r)].tau.iloc[0]) for r in RS} for m in P}
    pt = savings_point(y, P, T, RS); bt = saving_boot(y, P, T, RS); S = summarise(pt, bt, ["BN", "LR", "RF"], RS); S.insert(0, "stage", stage); S["n_rows"] = len(y); allr.append(S)
    d = comb[comb.contribution == key]
    for (m, r), (sv, fl, c) in pt.items():
        row = d[(d.model == m) & (d.r == r)].iloc[0]; assert abs(sv - row.saving) < 1e-15 and c == row.cost and fl == row.flagged
    p.check(f"{stage}: point savings (60 cells) equal sel_decision_value_combined.csv", True)
S = pd.concat(allr, ignore_index=True)
def get(stage, kind, model, comp, r): return S[(S.stage == stage) & (S.kind == kind) & (S.model == model) & (S.comparator == comp) & (S.r == r)].iloc[0]
# gate against v4 ledger rows: parse displays
def parse(disp):
    d = disp.replace(MINUS, "-").replace(" pts", "").replace("%", "").replace("[", "").replace("]", "").replace(",", ""); a = d.split(); return [float(x) for x in a]
for lid, row in v4.iterrows():
    n = int(lid.split("-")[2])
    if not (10 <= n <= 40): continue
    q = row.quantity; stage = "Application" if "Application" in q else "After acceptance"
    r = int(q.split("r=")[1].split()[0].rstrip(",")) if "r=" in q else 14
    if q.startswith("Cost saving vs the cheaper"):
        model = q.split("interval, ")[1].split(",")[0]; x = get(stage, "saving", model, "", r)
    elif q.startswith("Cost saving difference"):
        a, b = q.split("difference ")[1].split(" with")[0].split(" minus "); x = get(stage, "difference", a, b, r)
    else: continue
    pv, lo, hi = parse(row.display)
    dec = len(row.display.split("[")[0].split(".")[1].split("%")[0].split(" ")[0]) if "." in row.display else 1
    ok = abs(round(x.point, dec) - pv) < 1e-9 and abs(round(x.lo, dec) - lo) < 1e-9 and abs(round(x.hi, dec) - hi) < 1e-9
    p.gate(f"{lid} ({q[:70]}...): value and interval reproduced ({row.display})", ok, f"{x.point:.4f} [{x.lo:.4f}, {x.hi:.4f}]")
S.to_csv(ADD + "/cost_saving_by_r_with_intervals.csv", index=False)
md = ["## C1 data: cost saving and saving differences with 95% intervals, every integer r", "",
      "`cost_saving_by_r_with_intervals.csv` (stage, kind, model, comparator, r, point, lo, hi, n_rows): saving (percent) for BN, LR, RF and saving differences BN minus LR and BN minus RF (percentage points) for r = 1..20 at both stages, with the J2 procedure (test rows resampled, 2,000 resamples, `default_rng(42)` multinomial counts, predictions and thresholds fixed, thresholds = stored `tau` column, paired across models and r, cheaper trivial policy recomputed in every resample). The J2 run kept intervals only for selected r, so all 20 were recomputed.", "",
      "**Gate: PASSED** (all 31 J2 ledger rows N-ADD-010 to 040 that concern savings and differences, i.e. including N-ADD-022 to 029 and 030 to 034, reproduce to the printed precision; point estimates equal `sel_decision_value_combined.csv` exactly).", "", checks_md(p), ""]
p.write("\n".join(md))
