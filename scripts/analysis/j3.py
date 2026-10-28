"""J3: the dependent-masking rule, its per-bin probabilities (Application) and a reproduction of the stored realised fractions."""
from addcommon import *
import importlib.util, json
spec = importlib.util.spec_from_file_location("MK", R + "/shared/src/masking.py"); MK = importlib.util.module_from_spec(spec); spec.loader.exec_module(MK)
p = Part("j3")
SRCF = "shared/src/masking.py"
t = pd.read_pickle(R + "/Data/processed/sel_test.pkl"); sn = json.load(open(R + "/Data/processed/sel_state_names.json"))
n = len(t); assert n == 30000
def codes(col):
    c = pd.Index(sn[col]).get_indexer(t[col].astype(str)); assert (c >= 0).all() and c.max() <= 4; return c
COND = {"Age": ("Debt Ratio (bin)", codes("Debt Ratio (bin)")), "Debt Ratio": ("Age (bin)", codes("Age (bin)"))}
for v_, (cn, cc) in COND.items():
    p.check(f"conditioning variable for masked {v_} is {cn.replace(' (bin)', '')} (masking.cond_variable)", MK.cond_variable(v_) == cn)
LV = [0.2, 0.4, 0.5]; SEEDS = MK.SEEDS
def scale(u, level, cc):
    """The bisection of masking.make_mask, returning s (identical code path)."""
    w = cc.astype(float) + 1.0; base = w / w.mean(); frac = lambda s: float((u < np.minimum(1.0, s * base)).mean())
    lo, hi = 0.0, 1.0
    while frac(hi) < level and hi < 1e6: hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if frac(mid) < level: lo = mid
        else: hi = mid
    s = hi if abs(frac(hi) - level) <= abs(frac(lo) - level) else lo
    return s, base
stored = pd.read_csv(R + "/selection/tables/sel_rq3_per_seed_v2.csv"); stored = stored[stored.model == "BN"].copy(); stored["level"] = stored.level.round(2)
res = {}; ok_real = True; ok_mask = True; n_cmp = 0
for v_, (cn, cc) in COND.items():
    for lev in LV:
        ps = []; ss = []
        for sd in SEEDS:
            u = np.random.RandomState(sd).random_sample(n)
            s, base = scale(u, lev, cc); pk = np.minimum(1.0, s * base)   # per-row probability
            m_dep, real_dep = MK.make_mask(n, lev, "dependent", sd, cc)
            ok_mask &= bool(np.array_equal(m_dep, u < pk))
            m_mc, real_mc = MK.make_mask(n, lev, "mcar", sd)
            ok_mask &= bool(np.array_equal(m_mc, u < lev))
            for mech, real in (("dependent", real_dep), ("mcar", real_mc)):
                r = stored[(stored.variable == v_) & (stored.level == lev) & (stored.mechanism == mech) & (stored.seed == sd)].realised
                assert len(r) == 1; ok_real &= abs(float(r.iloc[0]) - real) < 1e-12; n_cmp += 1
            bins = np.array([pk[cc == k][0] for k in range(5)]); ps.append(bins); ss.append(s)
        res[(v_, lev)] = dict(p=np.array(ps), s=np.array(ss))
p.check(f"my reimplementation of the bisection reproduces the masks of masking.make_mask for all {len(COND)*len(LV)*len(SEEDS)} (variable, level, seed) cells, both mechanisms", ok_mask)
p.check(f"the realised hidden fraction of every one of the {n_cmp} (variable, level, mechanism, seed) masks equals the value stored in sel_rq3_per_seed_v2.csv (1e-12): conditioning codes and seeds are the ones used by the grid", ok_real)
# overall hidden share equals the level (dependent): weighted mean of per-bin p
for (v_, lev), d in res.items():
    cc = COND[v_][1]; cnt = np.bincount(cc, minlength=5) / n
    sh = float((d["p"].mean(axis=0) * cnt).sum())
    p.check(f"{v_} masked, m={int(lev*100)}%: sum over bins of P(bin) x p_bin = {sh:.5f} equals m (tol 0.005; realised fraction per seed within 0.005 asserted in masking.py)", abs(sh - lev) <= 0.005)
# rows
CIM = ""
for v_, (cn, cc) in COND.items():
    labels = sn[cn]
    for lev in LV:
        d = res[(v_, lev)]; mean_p = d["p"].mean(axis=0); mn = d["p"].min(axis=0); mx = d["p"].max(axis=0); cnt = np.bincount(cc, minlength=5)
        for k in range(5):
            p.row("rq3-selection", f"Dependent masking: probability that a test row is hidden, {v_} masked, conditioning bin {k+1} of {cn.replace(' (bin)', '')} ({labels[k]}), masking level {int(lev*100)}%",
                  float(mean_p[k]), f"{mean_p[k]:.4f}", "probability", SRCF, f"masking.make_mask(mechanism='dependent') | masked={v_} | conditioning={cn} | bin index {k} | level={lev} | mean over seeds 42-51",
                  derived="Y", notes=f"p = min(1, s x (k+1) / mean(k+1)) with k the 0-based conditioning bin and s the bisected scale (mean s = {d['s'].mean():.4f}, range {d['s'].min():.4f} to {d['s'].max():.4f} over the 10 seeds); value = mean over seeds 42 to 51, range {mn[k]:.4f} to {mx[k]:.4f}; {cnt[k]:,} of 30,000 test rows are in this bin; rule from shared/src/masking.py, used by 07_rq3_grid cells index 4 (selection grid)",
                  display_rule="p4", n=30000, variable=v_, mechanism="dependent", level=f"{int(lev*100)}%", caveat="C6")
for v_ in COND:
    for lev in LV:
        d = res[(v_, lev)]
        p.row("rq3-selection", f"Dependent masking: scale factor s, {v_} masked, masking level {int(lev*100)}% (mean over seeds)", float(d["s"].mean()), f"{d['s'].mean():.4f}", "scale factor", SRCF,
              f"masking.make_mask | bisection of s | masked={v_} | level={lev}", derived="Y", notes=f"range over seeds 42-51: {d['s'].min():.4f} to {d['s'].max():.4f}; s is bisected so the realised masked share of (u < min(1, s w/mean(w))) is within 0.005 of the level (in practice exactly the level)", display_rule="s4", n=30000, variable=v_, mechanism="dependent", level=f"{int(lev*100)}%")
# report
def tbl(v_):
    cn, cc = COND[v_]; labels = sn[cn]
    L = [f"**{v_} masked (conditioning variable: {cn.replace(' (bin)', '')})**", "", "| Bin (k = 0..4) | Label | test rows | m = 20% | m = 40% | m = 50% |", "|---|---|---|---|---|---|"]
    cnt = np.bincount(cc, minlength=5)
    for k in range(5):
        L.append("| %d | %s | %s | %s |" % (k, labels[k], f"{cnt[k]:,}", " | ".join(f"{res[(v_, lv)]['p'].mean(axis=0)[k]:.4f}" for lv in LV)))
    L.append("| s (mean) | | | " + " | ".join(f"{res[(v_, lv)]['s'].mean():.4f}" for lv in LV) + " |")
    return L
md = ["## J3. Dependent-masking rule", "",
      "**Source.** `shared/src/masking.py` (`make_mask`, `cond_variable`); conditioning codes from `shared/src/rq3.py` (`cond_codes`, line 89; `run_config`, line 202; train-set mask for the imputer, `get_refit_imputer`, line 130, seed 42); the grid was run in `selection/notebooks/07_rq3_grid.ipynb`, cell index 3 (rq3.py v2), cell index 4 (selection grid, 12 configurations x 10 seeds) and cell index 5 (After-acceptance grid, 24 configurations x 10 seeds). The first grid in cell index 1 (`sel_rq3_per_seed.csv`) is superseded.", "",
      "**MCAR.** For a test set of n rows and a seed, `u = numpy.random.RandomState(seed).random_sample(n)` (one uniform per row, in row order of the test set) and row i is hidden iff `u_i < m` (m = 0.20, 0.40, 0.50). Each row is hidden independently with probability m; the realised share varies by seed (for example 0.1938 to 0.2017 at m = 0.2).", "",
      "**Dependent (MAR on an always-observed bin).** The same uniforms `u` (same seed, so the MCAR and dependent masks share the u sequence) are used. Let c_i in {0,...,4} be the 0-based bin of the conditioning variable of row i (equal-frequency five bins of the stored binning; the conditioning variable is always observed), w_i = c_i + 1 and wbar = mean over rows of w_i. Row i is hidden iff", "",
      "    u_i < p_i,   p_i = min(1, s x w_i / wbar)", "",
      "so the probability of being hidden rises linearly with the bin index (bin 4 is 5 times as likely to be hidden as bin 0 before capping). The scale s is found by bisection (`lo = 0`, `hi` doubled from 1 until the share reaches m, then 100 halvings) so that the realised share mean(u < p) is as close as possible to m; the code asserts |realised - m| <= 0.005 (in practice the realised share equals m to within 1/n). The overall hidden share is therefore m by construction, and within a bin it is p_k (up to sampling of u). No other normalisation is used.", "",
      "**Conditioning variable.** Debt Ratio bin for every masked variable, except Age bin when Debt Ratio is masked (`masking.cond_variable`). So Age is masked conditional on Debt Ratio, Debt Ratio conditional on Age, and each of the four After-acceptance variables conditional on Debt Ratio.", "",
      "**Applied to.** The test set only (30,000 rows); the same mask is shared by the BN, LR and RF for a (variable, level, mechanism, seed) cell; seeds 42 to 51. For the LR/RF primary imputer a separate train-set mask (seed 42, same rule, conditioning codes from the train rows) is used to refit the IterativeImputer.", "",
      "**Per-bin hiding probabilities p_k, Application (mean over the 10 seeds; seed-to-seed range is in the ledger notes).**", "", *tbl("Age"), "", *tbl("Debt Ratio"), "",
      "These probabilities are below 1 for every bin, so the cap `min(1, .)` is not active at m <= 50%: the highest is bin 4 at m = 50%.", "", checks_md(p), "",
      "**Deviations / notes.** The bins (and therefore the conditioning codes) are taken from `Data/processed/sel_test.pkl` with the order of `sel_state_names.json` instead of re-running `rq3.cond_codes` (which needs the full model context); the check that the realised fractions of all 120 stored dependent and 120 MCAR masks are reproduced exactly shows the codes are the ones used. Per-bin probabilities for the After-acceptance variables were not requested and are not tabulated (same rule, conditioning on Debt Ratio bin).", ""]
p.write("\n".join(md))
