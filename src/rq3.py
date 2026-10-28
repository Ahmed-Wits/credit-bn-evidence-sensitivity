"""RQ3 pipeline: ONE code path parameterised by contribution ('selection' or 'monitoring').
load_context -> run_config(ctx, masked_var, level, mechanism, seed). Masks apply to the TEST set only; models are trained on complete data."""
import os
assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED != 0: wrong kernel"
import json, pickle, time, importlib.util
import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp

R = os.environ["CREDIT_BN_ROOT"]
PROC = R + "/Data/processed/"
_L = lambda p: pickle.load(open(p, "rb"))
RQ3_VERSION = "v2: bin_mode wrap|clip; Number of Dependents imputation logging; run_clip_lr_rf"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, R + "/" + rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


MK = _load("masking_saved", "shared/src/masking.py")
SPEC_A = ['Age', 'Debt Ratio', 'Monthly Income', 'Open Credit Lines', 'Real Estate Loans', 'Number of Dependents']
MASK_VARS = {"selection": ['Age', 'Debt Ratio'], "monitoring": ['Credit Utilisation Rate', '30-59 Days Past Due', '90+ Days Late', '60-89 Days Past Due']}
PILOT_VAR = {"selection": "Debt Ratio", "monitoring": "60-89 Days Past Due"}
FB = ['Credit Utilisation Rate (bin)', 'Age (bin)', 'Debt Ratio (bin)', 'Open Credit Lines (bin)', '30-59 Days Past Due (bin)',
      '90+ Days Late (bin)', '60-89 Days Past Due (bin)', 'Real Estate Loans (bin)', 'Monthly Income (bin)', 'Number of Dependents (bin)']
SEL_EV = ['Age (bin)', 'Debt Ratio (bin)', 'Monthly Income (bin)', 'Number of Dependents (bin)', 'Real Estate Loans (bin)', 'Open Credit Lines (bin)']


def _states_frame(F, cols, S):
    out = {}
    for c in cols:
        s = F[c]
        if set(s.dropna().unique()) <= set(S[c]):
            out[c] = s
        else:
            m = s.map(lambda x: str(x) if pd.notnull(x) else np.nan)
            assert set(m.dropna().unique()) <= set(S[c]), "label mismatch in %s" % c
            out[c] = m
    return pd.DataFrame(out)


def _xin(model, X):
    return X if hasattr(model, "feature_names_in_") else X.values


def load_context(contribution):
    assert contribution in ("selection", "monitoring")
    from pgmpy.models import DiscreteBayesianNetwork
    sp = _L(PROC + "split_indices.pkl")
    g = {"os": os, "np": np, "pd": pd, "L": _L, "PROC": PROC, "ROOT": R}
    g["train_idx"], g["test_idx"] = np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"])
    g["y_train"], g["y_test"] = np.asarray(sp["y_train"]), np.asarray(sp["y_test"])
    code = "".join(json.load(open(R + "/selection/notebooks/00_precheck.ipynb"))["cells"][2]["source"])
    exec(compile(code[:code.index("# ---- validation")], "build_design_src", "exec"), g)
    raw_cols = SPEC_A if contribution == "selection" else list(g["IMPUTE_ORDER"])
    base = g["build_design"](raw_cols)
    y = g["y_test"].astype(int)
    if contribution == "selection":
        lr_gs = _L(R + "/selection/models/sel_lr_grid.pkl"); rf_gs = _L(R + "/selection/models/sel_rf_grid.pkl")
        models = {"LR": lr_gs.best_estimator_, "RF": rf_gs.best_estimator_}
        bp = _L(R + "/selection/predictions/sel_baseline_test_predictions.pkl"); bnp = _L(R + "/selection/predictions/sel_bn_test_predictions.pkl")
        p0 = {"BN": np.asarray(bnp["p"], float), "LR": np.asarray(bp["p_lr"], float), "RF": np.asarray(bp["p_rf"], float)}
        tau = {"BN": 1.0 / 11.0, "LR": float(bp["tau"]["LR"]), "RF": float(bp["tau"]["RF"])}
        b = _L(R + "/selection/models/sel_bn_final.pkl"); edges, cpds = b["model_edges"], b["cpds"]
        F = pd.read_pickle(PROC + "sel_test.pkl"); ev_cols = SEL_EV + ["M_Income", "M_Dependents"]
    else:
        models = {"LR": _L(R + "/full_variable/models/lr_final.pkl"), "RF": _L(R + "/full_variable/models/rf_final_restricted.pkl")}
        abl = _L(R + "/full_variable/predictions/ablation_predictions.pkl"); bnp = _L(R + "/full_variable/predictions/bn_test_predictions.pkl")
        p0 = {"BN": np.asarray(bnp["bn_test_probs"], float), "LR": np.asarray(abl["LR_all"], float), "RF": np.asarray(abl["RF_all"], float)}
        tau = {"BN": 0.17, "LR": 0.69, "RF": 0.67}
        d = _L(R + "/full_variable/models/bn_final_model_iteration79.pkl"); edges, cpds = d["model_edges"], d["cpds"]
        F = _L(PROC + "df_test.pkl"); ev_cols = FB + ["M_Income", "M_Dependents"]
        assert bnp["bn_test_probs"].index.equals(F.index), "BN prediction index differs from df_test: STOP"
    assert list(F.index) == list(g["test_idx"]), "test frame order differs from split: STOP"
    assert (F["Default"].to_numpy().astype(int) == y).all()
    bn = DiscreteBayesianNetwork([tuple(e) for e in edges]); bn.add_cpds(*cpds); assert bn.check_model()
    S = {n: list(bn.get_cpds(n).state_names[n]) for n in bn.nodes()}
    Fs = _states_frame(F, ev_cols, S)
    codes = np.column_stack([pd.Index(S[c]).get_indexer(Fs[c]) for c in ev_cols])
    for j, c in enumerate(ev_cols):
        assert ((codes[:, j] < 0) == Fs[c].isna().to_numpy()).all(), "unseen state in %s" % c
    return dict(name=contribution, g=g, dfc=g["dfc"], raw_cols=raw_cols, base=base, y=y, train_idx=g["train_idx"], test_idx=g["test_idx"],
                models=models, p0=p0, tau=tau, bn=bn, S=S, ev_cols=ev_cols, codes=codes, imp_cache={})


def cond_codes(ctx, masked_raw, split):
    cv = MK.cond_variable(masked_raw).replace(" (bin)", "")
    idx = ctx["test_idx"] if split == "test" else ctx["train_idx"]
    s = ctx["g"]["cut_edges"](ctx["dfc"].loc[idx, cv], ctx["g"]["EDGES"][cv])
    c = np.asarray(s.cat.codes).astype(int)
    assert c.min() >= 0 and c.max() <= 4, "conditioning codes out of range"
    return c


def bn_posteriors(ctx, masked_raw, mask, check_bp=True):
    from pgmpy.inference import VariableElimination, BeliefPropagation
    bn, S, ev_cols = ctx["bn"], ctx["S"], ctx["ev_cols"]
    c2 = ctx["codes"].copy()
    if masked_raw is not None:
        c2[mask, ev_cols.index(masked_raw + " (bin)")] = -1
    u, inv = np.unique(c2, axis=0, return_inverse=True); inv = np.asarray(inv).ravel()
    ev_of = lambda r: {c: S[c][i] for c, i in zip(ev_cols, r) if i >= 0}
    eng = VariableElimination(bn); pu = np.full(len(u), np.nan); t0 = time.time()
    for k, r in enumerate(u):
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                p = float(eng.query(variables=["Default"], evidence=ev_of(r), show_progress=False).get_value(Default=1))
        except Exception:
            p = np.nan
        pu[k] = p if (np.isfinite(p) and 0 <= p <= 1) else np.nan
    info = dict(n_patterns=int(len(u)), t_ve=time.time() - t0, bp_maxdiff=None, bp_n=0, t_bp=0.0)
    if check_bp:
        valid = np.where(np.isfinite(pu))[0]
        pick = valid if len(valid) <= 500 else np.random.RandomState(42).choice(valid, 500, replace=False)
        bpe = BeliefPropagation(bn); t0 = time.time(); dmax = 0.0
        for k in pick:
            pb = float(bpe.query(variables=["Default"], evidence=ev_of(u[k]), show_progress=False).get_value(Default=1))
            dmax = max(dmax, abs(pb - pu[k]))
        info.update(bp_maxdiff=dmax, bp_n=int(len(pick)), t_bp=time.time() - t0)
    return pu[inv], info


def _imp_cols(ctx):
    return [c for c in ctx["g"]["IMPUTE_ORDER"] if c in ctx["raw_cols"]]


def get_refit_imputer(ctx, masked_raw, level, mechanism):
    key = (masked_raw, float(level), mechanism)
    if key in ctx["imp_cache"]:
        imp, t, real = ctx["imp_cache"][key]; return imp, t, real, True
    imp_cols = _imp_cols(ctx)
    tr = ctx["dfc"].loc[ctx["train_idx"], imp_cols].astype(float)
    if "Monthly Income" in imp_cols:
        tr["Monthly Income"] = np.log1p(tr["Monthly Income"])
    cc = None if mechanism == "mcar" else cond_codes(ctx, masked_raw, "train")
    tm, real = MK.make_mask(len(tr), level, mechanism, 42, cc)
    tr.loc[tm, masked_raw] = np.nan
    t0 = time.time(); imp = IterativeImputer(random_state=42).fit(tr); t = time.time() - t0
    if "Number of Dependents" in imp_cols:
        dv = imp.transform(tr)[:, imp_cols.index("Number of Dependents")]
        ctx.setdefault("train_stats", {})[key] = dict(train_dep_le_m1=int((dv <= -1).sum()), train_dep_le_m5=int((dv <= -5).sum()), train_dep_min=float(dv.min()))
    ctx["imp_cache"][key] = (imp, t, real)
    return imp, t, real, False


def _bin_natural(series, bin_mode="wrap", max_category=3):
    """Locked bin_natural_category, crash-free. 'wrap' (primary): labels[int(x)]; Python negative indexing for int(x) in -4..-1
    (identical to the locked function wherever that did not raise), int(x) % 4 for int(x) <= -5. 'clip' (sensitivity only): x clipped at 0 first."""
    if bin_mode not in ("wrap", "clip"):
        raise ValueError(bin_mode)
    labels = [str(i) for i in range(max_category)] + [f"{max_category}+"]
    def map_val(x):
        if pd.isnull(x):
            return np.nan
        if bin_mode == "clip":
            x = max(x, 0.0)
        if x < max_category:
            i = int(x)
            if i < -len(labels):
                i = i % len(labels)
            return labels[i]
        return labels[-1]
    return pd.Categorical(series.map(map_val), categories=labels, ordered=True)


def masked_matrix(ctx, masked_raw, test_mask, mode, imp=None, bin_mode="wrap", stats=None):
    g, base = ctx["g"], ctx["base"]
    imp_cols = _imp_cols(ctx); des_cols = [c for c in g["DESIGN_ORDER"] if c in ctx["raw_cols"]]
    tr = ctx["dfc"].loc[ctx["train_idx"], imp_cols].astype(float); te = ctx["dfc"].loc[ctx["test_idx"], imp_cols].astype(float)
    if "Monthly Income" in imp_cols:
        tr["Monthly Income"] = np.log1p(tr["Monthly Income"]); te["Monthly Income"] = np.log1p(te["Monthly Income"])
    if mode == "refit":
        te.loc[test_mask, masked_raw] = np.nan; use = imp
    elif mode in ("median", "mean"):
        te.loc[test_mask, masked_raw] = tr[masked_raw].median() if mode == "median" else tr[masked_raw].mean(); use = base["imputer"]
    else:
        raise ValueError(mode)
    te_i = pd.DataFrame(use.transform(te), columns=imp_cols, index=te.index)
    if "Monthly Income" in imp_cols:
        te_i["Monthly Income"] = np.expm1(te_i["Monthly Income"]).clip(lower=0)
    if stats is not None and "Number of Dependents" in te_i.columns:
        dv = te_i["Number of Dependents"].to_numpy()
        stats.update(test_dep_le_m1=int((dv <= -1).sum()), test_dep_le_m5=int((dv <= -5).sum()), test_dep_min=float(dv.min()))
    s_te = pd.DataFrame(index=te_i.index)
    for c in des_cols:
        s_te[c + " (bin)"] = g["cut_edges"](te_i[c], g["EDGES"][c]) if c in g["EDGE_VARS"] else _bin_natural(te_i[c], bin_mode, 3)
    s_te = s_te.astype(str)
    return pd.DataFrame(base["encoder"].transform(s_te), columns=base["Xte"].columns, index=te_i.index)


def _row(ctx, model, imp, p, valid):
    y = ctx["y"][valid]; pv = p[valid]; p0v = ctx["p0"][model][valid]; t = ctx["tau"][model]
    a, a0 = roc_auc_score(y, pv), roc_auc_score(y, p0v); d = np.abs(pv - p0v)
    return dict(model=model, imputation=imp, auc=a, gini=2 * a - 1, ks=float(ks_2samp(pv[y == 1], pv[y == 0]).statistic),
                ap=float(average_precision_score(y, pv)), auc_0pct_same_rows=a0, delta_auc=a - a0, mean_abs_dp=float(d.mean()),
                share_dp_gt_0_01=float((d > 0.01).mean()), flip_share=float(((pv >= t) != (p0v >= t)).mean()), tau=t, n_valid=int(valid.sum()))


def run_config(ctx, masked_raw, level, mechanism, seed, secondary="median", bp_check=True, stop_frac=0.001, return_preds=False, bin_mode="wrap"):
    y = ctx["y"]; n = len(y); p0 = ctx["p0"]; models = ctx["models"]; T = {}
    cc = None if mechanism == "mcar" else cond_codes(ctx, masked_raw, "test")
    mask, realised = MK.make_mask(n, level, mechanism, seed, cc)
    pbn, info = bn_posteriors(ctx, masked_raw, mask, check_bp=bp_check)
    T["bn_ve"], T["bn_bp"] = info["t_ve"], info["t_bp"]
    bad0, bad1 = ~np.isfinite(p0["BN"]), ~np.isfinite(pbn); valid = ~(bad0 | bad1)
    excl = dict(baseline=int(bad0.sum()), new_in_config=int((bad1 & ~bad0).sum()), total=int((~valid).sum()))
    if excl["total"] / n > stop_frac:
        raise RuntimeError("STOP: %d rows (%.3f%%) excluded in %s %s %.2f %s seed %d (limit %.1f%%)" % (excl["total"], 100.0 * excl["total"] / n, ctx["name"], masked_raw, level, mechanism, seed, 100 * stop_frac))
    imp, t_fit, _, cached = get_refit_imputer(ctx, masked_raw, level, mechanism)
    T["imputer_refit"] = 0.0 if cached else t_fit; T["imputer_cached"] = cached
    preds = {"BN": pbn}; rows = [_row(ctx, "BN", "evidence removed", pbn, valid)]
    for lab, mode in (("primary", "refit"), (secondary, secondary)):
        st = {}
        t0 = time.time(); X = masked_matrix(ctx, masked_raw, mask, mode, imp=imp, bin_mode=bin_mode, stats=st); T[lab + "_transform"] = time.time() - t0
        if lab == "primary":
            st.update(ctx.get("train_stats", {}).get((masked_raw, float(level), mechanism), {}))
        elif "Number of Dependents" in ctx["base"]["imp_tr"].columns:
            dv = ctx["base"]["imp_tr"]["Number of Dependents"].to_numpy()
            st.update(train_dep_le_m1=int((dv <= -1).sum()), train_dep_le_m5=int((dv <= -5).sum()), train_dep_min=float(dv.min()))
        for k in ("LR", "RF"):
            t0 = time.time(); p = models[k].predict_proba(_xin(models[k], X))[:, 1]; T[k.lower() + "_" + lab] = time.time() - t0
            preds[k + "_" + lab] = p
            r_ = _row(ctx, k, "refit IterativeImputer (primary)" if lab == "primary" else "train %s (secondary)" % secondary, p, valid); r_.update(st); rows.append(r_)
    out = dict(contribution=ctx["name"], masked_var=masked_raw, level=level, mechanism=mechanism, seed=seed, realised=realised,
               n_patterns=info["n_patterns"], bp_maxdiff=info["bp_maxdiff"], bp_n=info["bp_n"], excluded=excl, timing=T, rows=rows)
    if return_preds:
        out.update(preds=preds, mask=mask, valid=valid)
    return out


def validate_0pct(ctx, expected):
    y, p0, models, base = ctx["y"], ctx["p0"], ctx["models"], ctx["base"]; out = {}
    if ctx["name"] == "monitoring":
        Xl = _L(PROC + "X_test_df.pkl")
        assert np.array_equal(base["Xte"].values, Xl.values) and list(base["Xte"].columns) == list(Xl.columns), "rebuilt design != locked X_test_df: STOP"
        print("  rebuilt monitoring design equals the locked X_test_df exactly")
    for k in ("LR", "RF"):
        p = models[k].predict_proba(_xin(models[k], base["Xte"]))[:, 1]
        d = float(np.max(np.abs(p - p0[k]))); a = roc_auc_score(y, p)
        print("  %s object on the rebuilt design: AUC %.4f (expected %.4f) | max |p - saved| %.3e" % (k, a, expected[k], d))
        assert d <= 1e-12 and round(a, 4) == expected[k], "%s does not reproduce the saved predictions: STOP" % k
        out[k] = dict(auc=a, max_abs_diff=d)
    r = run_config(ctx, MASK_VARS[ctx["name"]][0], 0.0, "mcar", 42, return_preds=True)
    pb = r["preds"]["BN"]
    assert np.array_equal(np.isfinite(pb), np.isfinite(p0["BN"])), "BN NaN positions differ from saved: STOP"
    ok = np.isfinite(pb); d = float(np.max(np.abs(pb[ok] - p0["BN"][ok]))); a = roc_auc_score(y[ok], pb[ok])
    print("  BN at 0%%: %d rows valid | AUC %.4f (expected %.4f) | max |p - saved| %.3e | unique patterns %d | BP check on %d patterns max diff %.3e" % (
        ok.sum(), a, expected["BN"], d, r["n_patterns"], r["bp_n"], r["bp_maxdiff"]))
    assert d <= 1e-10 and round(a, 4) == expected["BN"] and int(ok.sum()) == expected["n_bn"] and r["bp_maxdiff"] <= 1e-10, "BN validation failed: STOP"
    out["BN"] = dict(auc=a, max_abs_diff=d, n_valid=int(ok.sum()), n_patterns=r["n_patterns"], bp_maxdiff=r["bp_maxdiff"], bp_n=r["bp_n"])
    for k in ("LR", "RF"):
        for lab in ("primary", "median"):
            dd = float(np.max(np.abs(r["preds"][k + "_" + lab] - p0[k])))
            print("  level-0 masked pipeline, %s %s: max |p - saved| %.3e" % (k, lab, dd))
            assert dd <= 1e-9, "level-0 pipeline does not reproduce baseline: STOP"
            out[k + "_level0_" + lab] = dd
    assert all(abs(x["delta_auc"]) < 1e-9 and x["flip_share"] == 0 for x in r["rows"]), "level-0 delta/flip not zero: STOP"
    print("  level 0: delta AUC 0 and flip share 0 for every row of the table; excluded rows: %s" % r["excluded"])
    out["excluded_at_0pct"] = r["excluded"]; out["tau"] = ctx["tau"]
    return out



def run_clip_lr_rf(ctx, masked_raw, level, mechanism, seed):
    """SENSITIVITY ONLY: LR/RF with the primary refit imputer, imputed counts clipped at 0 before binning (bin_mode='clip').
    The BN is not rerun: the valid-row set is the baseline's (caller asserts the saved seed record shows no new BN exclusions).
    delta AUC is measured against the same saved 0% (wrap) predictions as the main grid."""
    y = ctx["y"]; n = len(y); p0 = ctx["p0"]
    cc = None if mechanism == "mcar" else cond_codes(ctx, masked_raw, "test")
    mask, realised = MK.make_mask(n, level, mechanism, seed, cc)
    valid = np.isfinite(p0["BN"])
    imp, _, _, _ = get_refit_imputer(ctx, masked_raw, level, mechanism)
    st = {}
    X = masked_matrix(ctx, masked_raw, mask, "refit", imp=imp, bin_mode="clip", stats=st)
    out = {}
    for k in ("LR", "RF"):
        p = ctx["models"][k].predict_proba(_xin(ctx["models"][k], X))[:, 1]
        out[k] = _row(ctx, k, "refit IterativeImputer (primary), clip", p, valid)
    return dict(realised=realised, rows=out, n_valid=int(valid.sum()))
