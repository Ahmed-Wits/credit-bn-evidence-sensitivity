"""Cost-saving helpers shared by C1, J11, J12, J13 (same definitions and bootstrap as J2 / Fig 4)."""
import sys
import numpy as np, pandas as pd
from addcommon import R, boot_weights, pct_ci
sys.path.insert(0, R + "/shared/src")
import decision_value as dv

def savings_point(y, P, T, rs):
    """P: {model: p}, T: {model: {r: tau}} -> {(model, r): (saving, flagged, cost)}"""
    out = {}
    for r in rs:
        tr = dv.trivial(y, r)
        for m, p in P.items():
            fl = np.asarray(p) >= T[m][r]; c = dv.cost(y, fl, r); out[(m, r)] = (dv.saving(c, tr["cheaper"]), int(fl.sum()), c)
    return out

def saving_boot(y, P, T, rs, W=None):
    """Paired test-row bootstrap of the saving for every model and r (shared resample counts W: B x n). Returns {model: array(B, len(rs))}."""
    n = len(y); W = boot_weights(n) if W is None else W
    D = W @ y.astype(np.float32); B = W.shape[0]; out = {}
    for m, p in P.items():
        s = np.empty((B, len(rs)))
        for j, r in enumerate(rs):
            fl = np.asarray(p) >= T[m][r]
            fn = W @ ((y == 1) & ~fl).astype(np.float32); fp = W @ ((y == 0) & fl).astype(np.float32)
            s[:, j] = 1.0 - (r * fn + fp) / np.minimum(r * D, n - D)
        out[m] = s
    return out

def summarise(point, boot, models, rs, pairs=(("BN", "LR"), ("BN", "RF"))):
    """Rows: saving (percent) with 95% interval per model and r, and saving differences (percentage points) for the pairs."""
    rows = []
    for m in models:
        for j, r in enumerate(rs):
            lo, hi = pct_ci(100 * boot[m][:, j]); rows.append(dict(kind="saving", model=m, comparator="", r=r, point=100 * point[(m, r)][0], lo=lo, hi=hi))
    for a, b in pairs:
        for j, r in enumerate(rs):
            lo, hi = pct_ci(100 * (boot[a][:, j] - boot[b][:, j])); rows.append(dict(kind="difference", model=a, comparator=b, r=r, point=100 * (point[(a, r)][0] - point[(b, r)][0]), lo=lo, hi=hi))
    return pd.DataFrame(rows)

from sklearn.metrics import roc_auc_score
def paired_boot_full(y, pa, pb, n_boot=2000, seed=42):
    """Same loop as sensitivity.paired_bootstrap (RandomState(seed), randint); returns the observed difference and the resampled differences."""
    rng = np.random.RandomState(seed); n = len(y); diffs = []
    for _ in range(n_boot):
        ix = rng.randint(0, n, n); yb = y[ix]
        if yb.min() == yb.max(): continue
        diffs.append(roc_auc_score(yb, pa[ix]) - roc_auc_score(yb, pb[ix]))
    return roc_auc_score(y, pa) - roc_auc_score(y, pb), np.array(diffs)
