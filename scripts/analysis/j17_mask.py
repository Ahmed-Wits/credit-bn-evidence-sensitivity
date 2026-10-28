"""python j17_mask.py VARIANT RAWVAR : masking experiment (article design, primary imputer only) for one blanket variable of one After-acceptance variant.
BN drops the variable and marginalises; LR and RF (trained on complete data, not refit) get values from the IterativeImputer refit to a training copy masked in the same way.
Levels 20, 40, 50%; MCAR and dependent; 10 mask seeds (42-51); shared 2,000 multinomial bootstrap resamples (default_rng(42)); results in j17/mask_VARIANT_VAR.json"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from j_lib import *
name, var = sys.argv[1], sys.argv[2]; t_all = time.time()
fit, info, _ = load_fit(name, ADD + "/j17"); stage = fit["stage"]; cfg = variant_cfg(stage, fit["drop"]); g = get_g()
base = g["build_design"](cfg["raw"]); Xtr, Xte = base["Xtr"].values, base["Xte"].values; ytr = fit["ytr"]
lr, _s = lr_fit(cfg["lr"], Xtr, ytr); rf = RandomForestClassifier(n_jobs=int(os.environ.get("RF_JOBS", "2")), **cfg["rf"]).fit(Xtr, ytr)
assert np.abs(lr.predict_proba(Xte)[:, 1] - fit["plr"]).max() < 1e-9 and np.abs(rf.predict_proba(Xte)[:, 1] - fit["prf"]).max() < 1e-9, "refit LR/RF differ from the stored fit: STOP"
ctx = variant_ctx(fit, lr, rf, name, base=base); y = ctx["y"]; acc = MaskAcc(y); out = {}
base_p = {"BN": ctx["p0"]["BN"], "LR": ctx["p0"]["LR"], "RF": ctx["p0"]["RF"]}
for mech in MECHS:
    for lev in LEVELS:
        key = (var, lev, mech); real = []; excl = []
        for seed in SEEDS:
            r = rq3.run_config(ctx, var, lev, mech, seed, return_preds=True, bp_check=False); P = r["preds"]; v = r["valid"]; real.append(r["realised"]); excl.append(r["excluded"]["total"])
            acc.add(key, seed, v, {"BN": P["BN"], "LR": P["LR_primary"], "RF": P["RF_primary"]}, base_p)
        res, deg = acc.finish(key, ["BN", "LR", "RF"])
        out["|".join(map(str, key))] = dict(variable=var, level=lev, mechanism=mech, res=res, deg=deg, realised_mean=float(np.mean(real)), excluded_max=int(max(excl)), auc=[(m, [(s, a, a0) for s, a, a0 in acc.auc[(key, m)]]) for m in ("BN", "LR", "RF")])
        print(f"[{name} {var}] {mech} {lev:.1f} done  deg BN {deg['BN']:.5f} LR {deg['LR']:.5f} RF {deg['RF']:.5f}  D_LR {res['LR']['point']:+.5f} [{res['LR']['lo']:+.5f},{res['LR']['hi']:+.5f}]  ({(time.time()-t_all)/60:.1f} min)", flush=True)
        json.dump(out, open(f"{ADD}/j17/mask_{name}_{var.replace(' ', '_').replace('+','plus')}.json", "w"), indent=1, default=float)
print("saved", name, var, f"{(time.time()-t_all)/60:.1f} min")
