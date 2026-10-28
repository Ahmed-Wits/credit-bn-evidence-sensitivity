"""J8 step 2: paired test-row bootstrap for the After-acceptance masking experiment, same procedure as the Application stage (Phase 8 Part B2):
one shared 2000 x 30000 multinomial count matrix (default_rng(42)), counts zeroed on rows invalid in that seed run (the 0% baseline of the run uses the same
restricted counts), D(m, b) = mean over the 10 mask seeds of [masked AUC - 0% AUC], contrast C(b) = D(BN, b) - D(model, b); positive = the BN degrades less."""
import os, sys, time, json, pickle, hashlib
assert os.environ.get("PYTHONHASHSEED") == "0"
import numpy as np, pandas as pd
R = os.environ["CREDIT_BN_ROOT"]; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"
sys.path.insert(0, R + "/shared/src"); import fast_auc as fa
assert hashlib.sha256(open(R + "/shared/src/fast_auc.py", "rb").read()).hexdigest()[:12] == "66200c06261d", "fast_auc changed"
blob = pickle.load(open(ADD + "/intermediate/mon_rq3_masked_predictions.pkl", "rb")); y, p0, runs = blob["y"], blob["p0"], blob["runs"]; n = len(y)
sm = pd.read_csv(R + "/monitoring_rq3/mon_rq3_summary_v2.csv"); sm["level"] = sm["level"].round(2)
B = 2000; rng = np.random.default_rng(42); W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.int16)
MODELS = ("BN", "LR", "RF"); fix = lambda p: np.nan_to_num(np.asarray(p, float), nan=0.0)
P0 = {m: fa.prepare(fix(p0[m]), y) for m in MODELS}
def batch(prep, v):
    out = np.empty(B)
    for a in range(0, B, 500): out[a:a + 500] = fa.auc_weighted(prep, W[a:a + 500].astype(float) * v, chunk=500)
    return out
keys = sorted({(k[0], k[1], k[2]) for k in runs}); assert len(keys) == 24
rows = []; t_all = time.time(); worst = 0.0
for (var, lev, mech) in keys:
    Dn = {m: np.zeros(B) for m in MODELS}; Dp = {m: 0.0 for m in MODELS}; less = {"LR": 0, "RF": 0}
    for seed in range(42, 52):
        r = runs[(var, lev, mech, seed)]; v = r["valid"].astype(float); dseed = {}
        for m in MODELS:
            pm = fa.prepare(fix(r[m]), y)
            a_b = batch(pm, v); b_b = batch(P0[m], v); Dn[m] += a_b - b_b
            dseed[m] = float(fa.auc_weighted(pm, v)[0] - fa.auc_weighted(P0[m], v)[0]); Dp[m] += dseed[m]
        for o in ("LR", "RF"): less[o] += int(dseed["BN"] - dseed[o] > 0)
    for m in MODELS: Dn[m] /= 10; Dp[m] /= 10
    for other, vn in (("LR", "LR_primary"), ("RF", "RF_primary")):
        ref = lambda nm: float(sm[(sm.variable == var) & (sm.level == lev) & (sm.mechanism == mech) & (sm.variant == nm)].mean_delta_auc.iloc[0])
        ref_c = ref("BN") - ref(vn); pt = Dp["BN"] - Dp[other]; worst = max(worst, abs(pt - ref_c))
        assert abs(pt - ref_c) <= 1e-6, "POINT ESTIMATE MISMATCH %s %s %s BN-%s: %.3e vs %.3e: STOP" % (var, lev, mech, other, pt, ref_c)
        C = Dn["BN"] - Dn[other]; lo, hi = np.percentile(C, [2.5, 97.5]); reading = "BN degrades less" if lo > 0 else ("BN degrades more" if hi < 0 else "tie")
        rows.append(dict(variable=var, level=lev, mechanism=mech, contrast="BN minus " + other, point_estimate=pt, summary_file_value=ref_c, boot_mean=float(C.mean()), ci_lo=lo, ci_hi=hi,
                         share_gt0=float((C > 0).mean()), reading=reading, n_resamples=B, n_seeds=10, n_seeds_bn_less=less[other]))
    print("  %-24s %-9s %.2f done | %.1f min elapsed" % (var, mech, lev, (time.time() - t_all) / 60), flush=True)
df = pd.DataFrame(rows); print("POINT ESTIMATES match mon_rq3_summary_v2.csv: max diff %.2e" % worst); print(df.round(4).to_string(index=False))
df.to_csv(ADD + "/mon_rq3_row_bootstrap_summary.csv", index=False)
json.dump(dict(resamples=B, seed=42, design="one shared 2000 x 30000 multinomial count matrix (default_rng(42)) for every seed run, model and configuration; counts zeroed on rows invalid in that seed run; the 0% baseline of the run uses the same restricted counts",
               statistic="D(m,b) = mean over the 10 mask seeds of [masked AUC - 0% AUC]; contrast C(b) = D(BN,b) - D(model,b); positive = the BN degrades less", point_estimate_max_diff_vs_summary=worst,
               reproduction=blob["repro"], readings={r: int(c) for r, c in df.reading.value_counts().items()}, runtime_min=(time.time() - t_all) / 60), open(ADD + "/mon_rq3_row_bootstrap_notes.json", "w"), indent=1)
print("readings:", df.reading.value_counts().to_dict(), "| %.1f min" % ((time.time() - t_all) / 60))
