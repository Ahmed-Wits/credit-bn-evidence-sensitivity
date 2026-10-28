"""Sensitivity and value-of-information loops ported from the locked kernel code (no logic changes).
Notebook globals of the locked code are explicit arguments here. Additions that do not change locked outputs:
analyze_blanket(skip_unobserved=...) and an extra trailing column retracted_prob; boot_means groups the locked bootstrap loop."""
import os
assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED != 0: wrong kernel"
import time
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination
from sklearn.metrics import roc_auc_score


def make_posterior_fn(model):
    eng = VariableElimination(model)
    cache_m = {}
    def f(evidence):
        key = tuple(sorted(evidence.items(), key=lambda kv: kv[0]))
        if key in cache_m:
            return cache_m[key]
        with np.errstate(invalid='ignore', divide='ignore'):
            try:
                p = eng.query(variables=['Default'], evidence=evidence, show_progress=False).get_value(Default=1)
            except Exception:
                p = np.nan
        if not np.isfinite(p) or not (0 <= p <= 1):
            p = np.nan
        cache_m[key] = p
        return p
    return f


def analyze_blanket(model, rows, evid_all, mb, STATES, tau, skip_unobserved=False, verbose=True):
    post = make_posterior_fn(model)
    eng = VariableElimination(model)
    wcache = {}
    def pred_dist(var, ev_rest):
        key = (var, tuple(sorted(ev_rest.items(), key=lambda kv: kv[0])))
        if key in wcache:
            return wcache[key]
        with np.errstate(invalid='ignore', divide='ignore'):
            try:
                f = eng.query(variables=[var], evidence=ev_rest, show_progress=False)
                vals = np.asarray(f.values, dtype=float)
                w = dict(zip(f.state_names[var], vals)) if np.all(np.isfinite(vals)) else None
            except Exception:
                w = None
        wcache[key] = w
        return w
    recs, t0 = [], time.time()
    for i, idx in enumerate(rows):
        ev = evid_all[idx]
        base = post(ev)
        if np.isnan(base):
            continue
        for var in mb:
            if skip_unobserved and var not in ev:
                continue
            xo = ev[var]
            e_rest = {k: v for k, v in ev.items() if k != var}
            shifts, skipped = {}, 0
            for s in STATES[var]:
                if s == xo:
                    continue
                p2 = post({**ev, var: s})
                if np.isnan(p2):
                    skipped += 1
                else:
                    shifts[s] = abs(p2 - base)
            S_max = max(shifts.values()) if shifts else np.nan
            w = pred_dist(var, e_rest)
            if w is None:
                S_exp, w_obs = np.nan, np.nan
            else:
                S_exp = sum(ws * shifts[s] for s, ws in w.items() if s != xo and ws > 0 and s in shifts)
                w_obs = w.get(xo, np.nan)
            p_r = post(e_rest)
            delta = np.nan if np.isnan(p_r) else abs(p_r - base)
            flip = np.nan if np.isnan(p_r) else int((base >= tau) != (p_r >= tau))
            recs.append((idx, var, S_max, S_exp, delta, flip, base, skipped, w_obs, p_r))
        if verbose and (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(rows)} rows, {time.time()-t0:.0f}s")
    return pd.DataFrame(recs, columns=['row_index', 'variable', 'S_max', 'S_exp', 'delta', 'flip',
                                       'baseline_prob', 'skipped', 'w_obs', 'retracted_prob'])


def partB_check(post, rows, evid_all, m_var, val_var, STATES, tau):
    rows_idx = [i for i in rows if evid_all[i][m_var] == 0]
    worst, flips, skipped = 0.0, 0, 0
    for idx in rows_idx:
        ev = evid_all[idx]; base = post(ev)
        for s in STATES[val_var]:
            p2 = post({**ev, m_var: 1, val_var: s})
            if np.isnan(p2):
                skipped += 1
                continue
            worst = max(worst, abs(p2 - base))
            flips += int((base >= tau) != (p2 >= tau))
    return len(rows_idx), worst, flips, skipped


def paired_bootstrap(y, pa, pb, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    obs = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        diffs.append(roc_auc_score(yb, pa[idx]) - roc_auc_score(yb, pb[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = max(2 * min((diffs > 0).mean(), (diffs < 0).mean()), 1.0 / len(diffs))
    return obs, lo, hi, p, len(diffs)


def boot_means(mats, B=2000, seed=42):
    """Row-bootstrap of column means; every matrix in mats uses the same resamples (as in the locked loop)."""
    rng = np.random.RandomState(seed)
    n = len(mats[0])
    out = [np.empty((B, m.shape[1])) for m in mats]
    for i in range(B):
        ix = rng.randint(0, n, n)
        for o, m in zip(out, mats):
            o[i] = m[ix].mean(axis=0)
    return out
