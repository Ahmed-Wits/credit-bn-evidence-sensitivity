"""J8 step 1: re-run the After-acceptance masking grid (240 seed runs) keeping per-run test predictions. Inference only: the locked BN, LR and RF are not refitted
(the primary IterativeImputer is refitted on a masked train copy, as in the original grid, 07_rq3_grid cell index 5). Reproduction gate: every run's AUC and
delta AUC (BN, LR_primary, RF_primary) must equal mon_rq3_per_seed_v2.csv within 1e-6, else STOP before anything is saved or any interval is computed."""
import os, sys, time, json, pickle, hashlib, importlib.util, warnings
assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED must be 0"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = os.environ["CREDIT_BN_ROOT"]; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"
GATE = 1e-6
sys.path.insert(0, R + "/shared/src")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = json.load(open(R + "/selection/tables/sel_rq3_module_v2.json"))
for rel, h in {"shared/src/masking.py": "c6094c2b232e", "shared/src/rq3.py": rec["rq3_v2_sha12"], "shared/src/checks.py": "ef542907c037"}.items():
    assert sha(R + "/" + rel).startswith(h), "hash mismatch for %s: STOP" % rel
def load(name, rel):
    s = importlib.util.spec_from_file_location(name, R + "/" + rel); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rq3 = load("rq3", "shared/src/rq3.py")
ctx = rq3.load_context("monitoring"); rq3.validate_0pct(ctx, dict(BN=0.8519, LR=0.8618, RF=0.8603, n_bn=29998))
y, p0 = ctx["y"], ctx["p0"]
ps = pd.read_csv(R + "/monitoring_rq3/mon_rq3_per_seed_v2.csv"); ps["level"] = ps["level"].round(2)
ref = ps.set_index(["variable", "level", "mechanism", "seed", "variant"])
OUT = ADD + "/intermediate/mon_rq3_masked_predictions.pkl"; assert not os.path.exists(OUT)
saved = {}; worst = 0.0; t_all = time.time(); nrun = 0
LEVELS, SEEDS, MECHS = (0.20, 0.40, 0.50), range(42, 52), ("mcar", "dependent")
for var in rq3.MASK_VARS["monitoring"]:
    for mech in MECHS:
        for lev in LEVELS:
            t_c = time.time()
            for seed in SEEDS:
                r = rq3.run_config(ctx, var, lev, mech, seed, secondary="median", bp_check=(seed == 42), return_preds=True, bin_mode="wrap")
                v = r["valid"]; key = (var, round(lev, 2), mech, seed)
                for vname in ("BN", "LR_primary", "RF_primary"):
                    mdl = vname.split("_")[0]; p = r["preds"][vname]
                    a = roc_auc_score(y[v], p[v]); d = a - roc_auc_score(y[v], p0[mdl][v]); rr = ref.loc[key + (vname,)]
                    for nm, val in (("auc", a), ("delta_auc", d)):
                        diff = abs(val - float(rr[nm])); worst = max(worst, diff)
                        if diff > GATE: print("GATE FAILED: %s %s %s diff %.3e vs mon_rq3_per_seed_v2.csv: STOP" % (key, vname, nm, diff), flush=True); sys.exit(2)
                    assert int(rr["n_valid"]) == int(v.sum()), "n_valid mismatch %s" % (key,)
                saved[key] = dict(BN=r["preds"]["BN"].copy(), LR=r["preds"]["LR_primary"].copy(), RF=r["preds"]["RF_primary"].copy(), valid=v.copy(), realised=r["realised"]); nrun += 1
            print("  %-24s %-9s %.2f | 10 seeds reproduced | %.0fs | running worst diff %.2e | %d runs, %.1f min" % (var, mech, lev, time.time() - t_c, worst, nrun, (time.time() - t_all) / 60), flush=True)
sm = pd.read_csv(R + "/monitoring_rq3/mon_rq3_summary_v2.csv"); sm["level"] = sm["level"].round(2); worst_s = 0.0
for (var, lev, mech), _ in sm.groupby(["variable", "level", "mechanism"]):
    for vname, mdl in (("BN", "BN"), ("LR_primary", "LR"), ("RF_primary", "RF")):
        ds = [roc_auc_score(y[saved[(var, lev, mech, s)]["valid"]], saved[(var, lev, mech, s)][mdl][saved[(var, lev, mech, s)]["valid"]]) - roc_auc_score(y[saved[(var, lev, mech, s)]["valid"]], p0[mdl][saved[(var, lev, mech, s)]["valid"]]) for s in SEEDS]
        ref_m = float(sm[(sm.variable == var) & (sm.level == lev) & (sm.mechanism == mech) & (sm.variant == vname)].mean_delta_auc.iloc[0]); worst_s = max(worst_s, abs(np.mean(ds) - ref_m))
        assert abs(np.mean(ds) - ref_m) <= GATE, "summary mismatch %s %s %s %s" % (var, lev, mech, vname)
print("\nREPRODUCTION GATE PASSED: %d seed runs; per-seed AUC and delta AUC max abs diff %.3e (gate %.0e); summary mean delta AUC max abs diff %.3e; %.1f min" % (nrun, worst, GATE, worst_s, (time.time() - t_all) / 60), flush=True)
blob = dict(y=y, row_labels=np.asarray(ctx["test_idx"]), p0={k: np.asarray(v) for k, v in p0.items()}, runs=saved,
            repro=dict(max_per_seed_diff=worst, max_summary_diff=worst_s, gate=GATE, runs=nrun, minutes=(time.time() - t_all) / 60),
            note="LARGE INTERMEDIATE, After-acceptance RQ3 grid re-run with predictions kept (analysis_additions J8). Primary imputer only.")
with open(OUT, "xb") as f: pickle.dump(blob, f, protocol=4)
print("saved", OUT, os.path.getsize(OUT) // 1_000_000, "MB", flush=True)
