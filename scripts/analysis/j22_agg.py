"""J22 aggregation: gates (seed 42 equals J17), per-seed table, criterion. Writes j22_split_seed_results.csv and j22/j22_summary.pkl"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from addcommon import *
import pickle
p = Part("j22"); rows = []; J = {}
for v in ("after_V1", "after_V2"):
    for s in (42, 1, 2, 3, 4, 5):
        f = f"{ADD}/j22/{v}_seed{s}.json"
        if os.path.exists(f): J[(v, s)] = json.load(open(f))
for v in ("after_V1", "after_V2"):
    g = json.load(open(f"{ADD}/j17/{v}_fit.json")); d = J[(v, 42)]
    ok = all(abs(d["auc"][m] - g["auc"][m]) < 1e-9 for m in ("BN", "LR", "RF")) and set(d["blanket"]["blanket"]) == set(g["blanket"]["blanket"]) and d["structure"]["edges"] == g["structure"]["edges"] and all(abs(d["saving"][k]["point"] - g["saving"][k]["point"]) < 1e-9 for k in g["saving"])
    p.gate(f"{v}: seed 42 through the J22 code reproduces the J17 fit (structure, blanket, AUCs, savings; AUC BN {d['auc']['BN']:.4f})", ok)
    for k in ("auc_BN_minus_LR", "auc_BN_minus_RF"): p.gate(f"{v}: seed 42 {k} equals J17", abs(d[k]["observed"] - g[k]["observed"]) < 1e-9 and abs(d[k]["lo"] - g[k]["lo"]) < 1e-9)
short = lambda b: ", ".join(sorted(x.replace(" (bin)", "") for x in b if not x.startswith("M_")))
for (v, s), d in sorted(J.items()):
    if s == 42: continue
    row = dict(variant=v, seed=s, blanket=short(d["blanket"]["blanket"]), n_edges=d["structure"]["n_edges"], lr_solver=d["lr_solver"], n_bn_defined=d["n_bn_defined"], auc_BN=d["auc"]["BN"], auc_LR=d["auc"]["LR"], auc_RF=d["auc"]["RF"])
    for m in ("LR", "RF"): row.update({f"auc_BN_minus_{m}": d[f"auc_BN_minus_{m}"]["observed"], f"auc_BN_minus_{m}_lo": d[f"auc_BN_minus_{m}"]["lo"], f"auc_BN_minus_{m}_hi": d[f"auc_BN_minus_{m}"]["hi"]})
    for r in (5, 10):
        for m in ("BN", "LR", "RF"): e = d["saving"][f"{m}_r{r}"]; row.update({f"saving_{m}_r{r}": e["point"], f"saving_{m}_r{r}_lo": e["lo"], f"saving_{m}_r{r}_hi": e["hi"]})
        for m in ("LR", "RF"): e = d["saving_diff"][f"BN_minus_{m}_r{r}"]; row.update({f"savdiff_BN_minus_{m}_r{r}": e["point"], f"savdiff_BN_minus_{m}_r{r}_lo": e["lo"], f"savdiff_BN_minus_{m}_r{r}_hi": e["hi"]})
    rows.append(row)
R_ = pd.DataFrame(rows); R_.to_csv(ADD + "/j22_split_seed_results.csv", index=False)
v1 = R_[R_.variant == "after_V1"]; v2 = R_[R_.variant == "after_V2"]
inc0 = lambda d, m: (d[f"savdiff_BN_minus_{m}_r5_lo"] <= 0) & (d[f"savdiff_BN_minus_{m}_r5_hi"] >= 0)
n_v1 = int((inc0(v1, "LR") & inc0(v1, "RF")).sum()); n_v2 = int((v2["savdiff_BN_minus_LR_r5_hi"] < 0).sum())
verdict = "HELPFUL" if (n_v1 >= 4 and n_v2 >= 4) else "NOT HELPFUL (see counts)"
S = dict(n_v1_level=n_v1, n_v2_below=n_v2, verdict=verdict, n_v1_seeds=len(v1), n_v2_seeds=len(v2)); print(S)
print(R_[["variant", "seed", "blanket", "auc_BN", "auc_LR", "savdiff_BN_minus_LR_r5", "savdiff_BN_minus_LR_r5_lo", "savdiff_BN_minus_LR_r5_hi", "savdiff_BN_minus_RF_r5", "savdiff_BN_minus_RF_r5_lo", "savdiff_BN_minus_RF_r5_hi"]].round(3).to_string())
pickle.dump(dict(S=S, R=R_, checks=p.checks), open(ADD + "/j22/j22_summary.pkl", "wb"))
