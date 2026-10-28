"""Decision-value helpers (descriptive). cost(y, flag, r) = r*FN + FP. flag = p >= tau.
Trivial policies: accept_all (flag nobody) = r*D; reject_all (flag everyone) = N - D. Primary baseline = the cheaper of the two."""
import numpy as np
import pandas as pd

GRID = np.round(np.arange(1, 100) / 100.0, 2)


def cost(y, flag, r):
    y = np.asarray(y).astype(int); flag = np.asarray(flag).astype(bool)
    return int(r * ((~flag) & (y == 1)).sum() + (flag & (y == 0)).sum())


def confusion(y, flag):
    y = np.asarray(y).astype(int); flag = np.asarray(flag).astype(bool)
    return int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())


def trivial(y, r):
    y = np.asarray(y).astype(int); D = int(y.sum()); N = len(y)
    a, b = r * D, N - D
    return dict(accept_all=int(a), reject_all=int(b), cheaper=int(min(a, b)), cheaper_policy="accept_all" if a <= b else "reject_all")


def saving(cost_model, cheaper):
    return 1.0 - cost_model / cheaper


def oof_tau(oof, y_train, r, tie="highest"):
    """Cost-minimising tau on the 0.01 grid from OUT-OF-FOLD train predictions. tie: 'highest' (spec) or 'lowest'."""
    c = np.array([cost(y_train, np.asarray(oof) >= t, r) for t in GRID])
    best = GRID[c == c.min()]
    return float(best.max() if tie == "highest" else best.min()), int(c.min()), (float(best.min()), float(best.max()))


def curve(y, preds, taus_by_model, rs):
    """preds: {model: p}; taus_by_model: {model: {r: tau}}. Returns one row per (model, r)."""
    rows = []
    for r in rs:
        tr = trivial(y, r)
        for m, p in preds.items():
            t = taus_by_model[m][r]; fl = np.asarray(p) >= t; c = cost(y, fl, r); TP, FP, FN, TN = confusion(y, fl)
            rows.append(dict(r=r, model=m, tau=t, flagged=int(fl.sum()), TP=TP, FP=FP, FN=FN, TN=TN, cost=c, accept_all=tr["accept_all"], reject_all=tr["reject_all"],
                             cheaper_policy=tr["cheaper_policy"], cost_cheaper_trivial=tr["cheaper"], saving=saving(c, tr["cheaper"])))
    return pd.DataFrame(rows)
