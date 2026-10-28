"""J19 aggregation: gates (existing-comparator rows equal the stored row-bootstrap summaries), tables, summary counts. Writes j19_*.csv and j19/j19_summary.pkl."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
from addcommon import Part
p = Part("j19"); J19 = ADD + "/j19"
STG = {"application": ("Application", ["Age", "Debt Ratio"], R + "/selection/tables/sel_rq3_row_bootstrap_summary.csv"), "after": ("After acceptance", ["Credit Utilisation Rate", "30-59 Days Past Due", "60-89 Days Past Due", "90+ Days Late"], ADD + "/mon_rq3_row_bootstrap_summary.csv")}
TYPES = {"std": "existing (imputed values, trained on complete data)", "A": "A: matched-training with indicators", "B": "B: subset-retrained (reduced model on masked rows)", "Aown": "A, degradation vs its own unmasked-test AUC"}
rows = []; dec = []; auc_rows = []; notes = {}
for st, (lab, vars_, refcsv) in STG.items():
    ref = pd.read_csv(refcsv); ref["level"] = ref.level.round(2); worst = 0.0; nchk = 0
    for var in vars_:
        f = f"{J19}/res_{st}_{var.replace(' ', '_').replace('+', 'plus')}.json"
        if not os.path.exists(f): print("MISSING", f); continue
        d = json.load(open(f)); notes[(st, var)] = d["notes"]
        for key, m in d["masking"].items():
            for nm, r in m["res"].items():
                typ, mod = nm.split("_")[1], nm.split("_")[0]
                typ = typ if typ in ("A", "B", "Aown") else "std"
                rows.append(dict(stage=lab, variable=var, level=m["level"], mechanism=m["mechanism"], comparator_type=typ, model=mod, point=r["point"], boot_mean=r["boot_mean"], lo=r["lo"], hi=r["hi"], reading=reading(r["lo"], r["hi"]),
                                 deg_BN=r["deg_ref"], deg_comp=r["deg_c"], realised=m["realised_mean"], excluded_max=m["excluded_max"]))
            for nm, lst in m["auc"]:
                a_ = np.array([(a, a0) for s, a, a0 in lst]); auc_rows.append(dict(stage=lab, variable=var, level=m["level"], mechanism=m["mechanism"], score=nm, mean_masked_auc=float(a_[:, 0].mean()), mean_baseline_auc=float(a_[:, 1].mean())))
            if m["level"] and any(ref.variable.eq(var)):
                for nm, c in (("LR_std", "BN minus LR"), ("RF_std", "BN minus RF")):
                    q = ref[(ref.variable == var) & (ref.level == m["level"]) & (ref.mechanism == m["mechanism"]) & (ref.contrast == c)].iloc[0]; r = m["res"][nm]
                    worst = max(worst, abs(r["point"] - q.point_estimate), abs(r["lo"] - q.ci_lo), abs(r["hi"] - q.ci_hi)); nchk += 1
        for key, dd in d["decision"].items():
            for k, e in dd["saving"].items():
                nm, rr = k.rsplit("_r", 1); dec.append(dict(stage=lab, variable=var, mechanism=dd["mechanism"], level=dd["level"], score=nm, r=int(rr), **e))
    p.gate(f"{lab}: the existing comparators (stored LR/RF, refit imputer) reproduce the stored row-bootstrap summary ({nchk} contrasts: point, lower, upper; max abs diff {worst:.2e}, tolerance 1e-9)", nchk > 0 and worst < 1e-9)
MK = pd.DataFrame(rows); MK.to_csv(ADD + "/j19_masking_contrasts.csv", index=False)
DEC = pd.DataFrame(dec); DEC.to_csv(ADD + "/j19_decision_level.csv", index=False)
AUCS = pd.DataFrame(auc_rows); AUCS.to_csv(ADD + "/j19_masked_auc.csv", index=False)
# summary counts
sc = []
for lab in MK.stage.unique():
    for typ in ("std", "A", "B", "Aown"):
        for mod in ("LR", "RF"):
            x = MK[(MK.stage == lab) & (MK.comparator_type == typ) & (MK.model == mod)]
            sc.append(dict(stage=lab, comparator_type=typ, model=mod, n_scenarios=len(x), bn_degrades_less=int((x.lo > 0).sum()), tie=int(((x.lo <= 0) & (x.hi >= 0)).sum()), bn_degrades_more=int((x.hi < 0).sum()),
                           median_point=float(x.point.median()), min_point=float(x.point.min()), max_point=float(x.point.max())))
SC = pd.DataFrame(sc); SC.to_csv(ADD + "/j19_summary_counts.csv", index=False)
print(SC.to_string()); print(MK[MK.comparator_type != "std"].groupby(["stage", "comparator_type", "model"]).point.describe().round(4).to_string())
pickle.dump(dict(MK=MK, DEC=DEC, AUCS=AUCS, SC=SC, notes=notes, checks=p.checks), open(J19 + "/j19_summary.pkl", "wb"))
