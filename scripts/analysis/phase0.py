"""Phase 0 gate for J17-J19: reproduce the reference values of the brief from stored predictions / tables / the J12 seed-42 gate files."""
from addcommon import *
import pickle, json
from sklearn.metrics import roc_auc_score
sys.path.insert(0, R + "/shared/src")
import decision_value as dv
fails = []
def chk(n, ok, d=""):
    print(("PASS " if ok else "FAIL ") + n + (" | " + d if d else "")); 
    if not ok: fails.append(n)
J = {s: json.load(open(f"{ADD}/j12/{s}_seed42_gate.json")) for s in ("application", "after")}
P = {s: pickle.load(open(f"{ADD}/j12/{s}_seed42_preds.pkl", "rb")) for s in J}
ref_auc = {"application": (0.6627, 0.6904, 0.6926), "after": (0.8519, 0.8618, 0.8603)}
ref_sav = {"application": {5: (0.0, 0.0, 1.1), 10: (9.9, 16.1, 15.7)}, "after": {5: (35.2, 35.0, 35.1), 10: (47.7, 49.0, 49.1)}}
for s in J:
    p = P[s]; v = p["valid"]; y = p["yte"]
    auc = (roc_auc_score(y[v], p["pbn"][v]), roc_auc_score(y, p["plr"]), roc_auc_score(y, p["prf"]))
    chk(f"{s}: AUC BN/LR/RF {[round(a,4) for a in auc]} = {ref_auc[s]}", tuple(round(a, 4) for a in auc) == ref_auc[s])
    for r in (5, 10):
        tau = {m: J[s]["tau"][m][str(r)] for m in ("BN", "LR", "RF")}
        yv = y[v]; tr = dv.trivial(yv, r)
        got = tuple(round(100 * dv.saving(dv.cost(yv, p[k][v] >= tau[m], r), tr["cheaper"]), 1) for m, k in (("BN", "pbn"), ("LR", "plr"), ("RF", "prf")))
        chk(f"{s}: saving r={r} BN/LR/RF {got} = {ref_sav[s][r]}", got == ref_sav[s][r])
    chk(f"{s}: stored J12 seed-42 gate checks all pass ({len(J[s]['checks'])})", all(c["ok"] for c in J[s]["checks"]))
ref_sexp = {"Credit Utilisation Rate (bin)": 0.0280, "30-59 Days Past Due (bin)": 0.0245, "90+ Days Late (bin)": 0.0223, "60-89 Days Past Due (bin)": 0.0111, "Age (bin)": 0.0270, "Debt Ratio (bin)": 0.0122}
for s in J:
    for k, v in J[s]["sexp"].items(): chk(f"{s} mean S_exp {k} {v['mean']:.4f} = {ref_sexp[k]}", round(v["mean"], 4) == ref_sexp[k])
fr = pd.read_csv(ADD + "/flip_rate_by_r.csv"); fr = fr[fr.stage.str.startswith("After")]
ref = {5: {"Credit Utilisation Rate": 4.62, "30-59 Days Past Due": 4.63, "90+ Days Late": 3.10, "60-89 Days Past Due": 1.40}, 10: {"Credit Utilisation Rate": 3.76, "30-59 Days Past Due": 13.58, "90+ Days Late": 15.39, "60-89 Days Past Due": 3.72}}
for r in ref:
    for k, v in ref[r].items():
        g = 100 * fr[(fr.r == r) & (fr.variable == k)].flip_rate.iloc[0]; chk(f"After flip rate r={r} {k} {g:.2f} = {v}", round(g, 2) == v)
lad = pd.read_csv(R + "/selection/tables/sel_info_ladder_table.csv"); print(lad[["rung", "auc"]].to_string())
print("ladder", [round(x, 4) for x in lad.auc]); chk("ladder values 0.6582/0.6903/0.7934/0.8359/0.8618", [round(x, 4) for x in lad.auc] == [0.6582, 0.6903, 0.7934, 0.8359, 0.8618] or set(round(x,4) for x in lad.auc) >= {0.6582, 0.6903, 0.7934, 0.8359, 0.8618})
print("FAILS:", fails); sys.exit(1 if fails else 0)
