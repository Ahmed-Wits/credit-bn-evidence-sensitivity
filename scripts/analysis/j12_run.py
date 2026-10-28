"""python j12_run.py STAGE SEED [gate]: run one seed of J12 and save j12/STAGE_seedN.json (+ predictions pickle). With 'gate' and seed 42 the result is compared with the stored seed-42 results."""
import sys, os, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_robust import *
stage, seed = sys.argv[1], int(sys.argv[2]); gate = len(sys.argv) > 3 and sys.argv[3] == "gate"
OUTD = ADD + "/j12"
out = run_seed(stage, seed, log=lambda s: print(s, flush=True), n_jobs=int(os.environ.get("RF_JOBS", "2")))
preds = out.pop("preds"); pickle.dump(preds, open(f"{OUTD}/{stage}_seed{seed}_preds.pkl", "wb"))
checks = []
def chk(name, ok, detail=""): checks.append(dict(item=name, ok=bool(ok), detail=detail)); print(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""), flush=True)
if gate:
    assert seed == 42
    L = lambda f: pickle.load(open(R + "/" + f, "rb"))
    tr, te = get_split(42); sp = L("Data/processed/split_indices.pkl")
    chk("split seed 42 reproduces the stored split_indices (train and test index)", np.array_equal(tr, np.asarray(sp["train_idx"])) and np.array_equal(te, np.asarray(sp["test_idx"])))
    pa = L("Data/processed/preprocessing_artifacts.pkl")["bin_edges"]; ed = get_edges(tr)
    chk("train-only bin edges equal the stored bin_edges (5 variables)", all(np.allclose(ed[c], pa[c], rtol=0, atol=1e-12) for c in EQ5))
    sel = set(map(tuple, out["structure"]["edges"]))
    if stage == "application":
        stored = {tuple(e) for e in json.load(open(R + "/selection/tables/sel_dag_9node_edges.json"))["edges"]} - set(IND_EDGES)
        chk("learned structure equals the stored 13-edge Application structure (seed 55, BIC -788,831.4158)", sel == stored and out["structure"]["best_seed"] == 55 and abs(out["structure"]["best_bic"] + 788831.4158) < 1e-3, f"seed {out['structure']['best_seed']} BIC {out['structure']['best_bic']:.4f}")
        sbp = L("selection/predictions/sel_bn_test_predictions.pkl"); bp = L("selection/predictions/sel_baseline_test_predictions.pkl")
        chk("EM: 9 iterations, final train LL -998,922.732607 (stored)", out["em"]["iterations"] == 9 and abs(out["em"]["train_ll"] + 998922.732607) < 1e-4, f"{out['em']['iterations']} it, {out['em']['train_ll']:.6f}")
        chk("BN test posteriors equal the stored sel_bn_test_predictions p (max abs diff %.2e, tol 1e-9)" % np.abs(preds["pbn"] - np.asarray(sbp["p"], float)).max(), np.abs(preds["pbn"] - np.asarray(sbp["p"], float)).max() <= 1e-9)
        chk("LR test probabilities equal the stored p_lr (max abs diff %.2e, tol 1e-6)" % np.abs(preds["plr"] - np.asarray(bp["p_lr"], float)).max(), np.abs(preds["plr"] - np.asarray(bp["p_lr"], float)).max() <= 1e-6)
        chk("RF test probabilities equal the stored p_rf (max abs diff %.2e, tol 1e-9)" % np.abs(preds["prf"] - np.asarray(bp["p_rf"], float)).max(), np.abs(preds["prf"] - np.asarray(bp["p_rf"], float)).max() <= 1e-9)
        chk("AUC BN / LR / RF = 0.6627 / 0.6904 / 0.6926", tuple(round(out["auc"][m], 4) for m in ("BN", "LR", "RF")) == (0.6627, 0.6904, 0.6926), str({m: round(out["auc"][m], 4) for m in ("BN", "LR", "RF")}))
        led = {("BN", 5): 0.0, ("LR", 5): 0.0, ("RF", 5): 1.1, ("BN", 10): 9.9, ("LR", 10): 16.1, ("RF", 10): 15.7}
        chk("Table 7 Application savings at r = 5 and 10 (BN 0.0/9.9, LR 0.0/16.1, RF 1.1/15.7)", all(round(out["saving"][f"{m}_r{r}"]["point"], 1) == v for (m, r), v in led.items()), str({k: round(v["point"], 2) for k, v in out["saving"].items()}))
        chk("blanket = {Age, Debt Ratio}", out["blanket"]["blanket"] == ["Age (bin)", "Debt Ratio (bin)"])
        chk("mean S_exp Age 0.02697, Debt Ratio 0.01216 (ledger N-SEL-016, 017)", round(out["sexp"]["Age (bin)"]["mean"], 5) == 0.02697 and round(out["sexp"]["Debt Ratio (bin)"]["mean"], 5) == 0.01216, str({k: round(v["mean"], 5) for k, v in out["sexp"].items()}))
    else:
        d79 = L("full_variable/models/bn_final_model_iteration79.pkl"); stored = {tuple(e) for e in d79["model_edges"]} - set(IND_EDGES)
        chk("learned structure equals the stored 22-edge After-acceptance structure (seed 53, BIC -1,026,115.7308)", sel == stored and out["structure"]["best_seed"] == 53 and abs(out["structure"]["best_bic"] + 1026115.7308) < 1e-2, f"seed {out['structure']['best_seed']} BIC {out['structure']['best_bic']:.4f}; {len(sel)} edges")
        bb = L("full_variable/predictions/bn_test_predictions.pkl")["bn_test_probs"]; ab = L("full_variable/predictions/ablation_predictions.pkl")
        v = bb.notna().to_numpy(); chk("BN defined on 29,998 test rows, same rows as stored", np.array_equal(preds["valid"], v))
        dd = np.abs(preds["pbn"][v] - bb.to_numpy(float)[v]).max(); chk("iteration-100 BN test posteriors equal the stored iteration-79 posteriors (max abs diff %.2e, tol 1e-9; J11: the BN predictions do not depend on the iteration)" % dd, dd <= 1e-9)
        dl = np.abs(preds["plr"] - np.asarray(ab["LR_all"], float)).max(); dr = np.abs(preds["prf"] - np.asarray(ab["RF_all"], float)).max()
        chk("LR test probabilities equal the stored LR_all (max abs diff %.2e, tol 1e-6)" % dl, dl <= 1e-6); chk("RF test probabilities equal the stored RF_all (max abs diff %.2e, tol 1e-6)" % dr, dr <= 1e-6)
        chk("AUC BN / LR / RF = 0.8519 / 0.8618 / 0.8603", tuple(round(out["auc"][m], 4) for m in ("BN", "LR", "RF")) == (0.8519, 0.8618, 0.8603), str({m: round(out["auc"][m], 4) for m in ("BN", "LR", "RF")}))
        led = {("BN", 5): 35.2, ("LR", 5): 35.0, ("RF", 5): 35.1, ("BN", 10): 47.7, ("LR", 10): 49.0, ("RF", 10): 49.1}
        chk("Table 7 After-acceptance savings at r = 5 and 10 (35.2/35.0/35.1 and 47.7/49.0/49.1)", all(round(out["saving"][f"{m}_r{r}"]["point"], 1) == v_ for (m, r), v_ in led.items()), str({k: round(v_["point"], 2) for k, v_ in out["saving"].items()}))
        chk("blanket = {Credit Utilisation Rate, 90+ Days Late, 60-89 Days Past Due, 30-59 Days Past Due}", set(out["blanket"]["blanket"]) == {"Credit Utilisation Rate (bin)", "90+ Days Late (bin)", "60-89 Days Past Due (bin)", "30-59 Days Past Due (bin)"})
        chk("mean S_exp 0.0280 / 0.0245 / 0.0223 / 0.0111 (CUR, 30-59, 90+, 60-89)", [round(out["sexp"][k]["mean"], 4) for k in ("Credit Utilisation Rate (bin)", "30-59 Days Past Due (bin)", "90+ Days Late (bin)", "60-89 Days Past Due (bin)")] == [0.028, 0.0245, 0.0223, 0.0111])
out["checks"] = checks; json.dump(out, open(f"{OUTD}/{stage}_seed{seed}{'_gate' if gate else ''}.json", "w"), indent=1, default=float)
print("saved", f"{stage}_seed{seed}", "| gate fails:", sum(not c["ok"] for c in checks))
