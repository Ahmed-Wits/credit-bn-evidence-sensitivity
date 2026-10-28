"""Weighted AUC for many bootstrap resamples from one sort.
Resamples are multinomial row counts used as weights. Ties use the 0.5 convention (AUC = P(s+ > s-) + 0.5 P(s+ = s-)),
so scores with few distinct values (e.g. a BN with 25 values) are correct. prepare() once per score vector; auc_weighted() per batch of weights."""
import numpy as np


def prepare(scores, y):
    s = np.asarray(scores, float); y = np.asarray(y).astype(int)
    order = np.argsort(s, kind="mergesort"); ss = s[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    return dict(order=order, y=y[order].astype(float), starts=starts)


def auc_weighted(prep, W, chunk=200):
    W = np.atleast_2d(np.asarray(W, float)); out = np.empty(W.shape[0]); o, yy, st = prep["order"], prep["y"], prep["starts"]
    for a in range(0, W.shape[0], chunk):
        Wo = W[a:a + chunk][:, o]
        Wp = np.add.reduceat(Wo * yy, st, axis=1); Wn = np.add.reduceat(Wo * (1.0 - yy), st, axis=1)
        cumneg = np.cumsum(Wn, axis=1) - Wn
        num = (Wp * cumneg).sum(1) + 0.5 * (Wp * Wn).sum(1)
        out[a:a + chunk] = num / (Wp.sum(1) * Wn.sum(1))
    return out
