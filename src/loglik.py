"""Log-likelihood of a discrete BN on data with missing values (chain-rule CPT lookup).
Equivalent to the locked compute_log_likelihood (joint marginal of the observed variables, floor 1e-12),
but vectorised over rows grouped by missing-pattern. Missing nodes are summed out by enumeration."""
import os
assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED != 0: wrong kernel"
import itertools
import numpy as np
import pandas as pd

EPS = 1e-12            # final floor, applied to rows with missing variables only
FACTOR_FLOOR = 1e-12   # per-CPT-factor floor (matches locked compute_joint_prob_chain_rule)


def _tables(model):
    nodes = list(model.nodes())
    T = {}
    for n in nodes:
        cpd = model.get_cpds(n)
        vars_ = list(cpd.variables)  # [node] + parents
        arr = np.asarray(cpd.values, dtype=float)  # axes follow vars_
        T[n] = (vars_, arr)
    sn = {}
    for n in nodes:
        cpd = model.get_cpds(n)
        for v in cpd.variables:
            sn.setdefault(v, list(cpd.state_names[v]))
    return nodes, T, sn


def compute_log_likelihood(model, data, return_detail=False):
    nodes, T, sn = _tables(model)
    N = len(data)
    codes = {}
    for n in nodes:
        if n in data.columns:
            c = pd.Index(sn[n]).get_indexer(data[n])
            bad = (c < 0) & data[n].notna().to_numpy()
            if bad.any():
                raise ValueError(f"unseen state in {n}: {data[n][bad].unique()[:5]}")
        else:
            c = np.full(N, -1)
        codes[n] = np.asarray(c)
    M = np.column_stack([codes[n] < 0 for n in nodes])
    pats, inv = np.unique(M, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    ll = np.zeros(N)
    skipped = np.zeros(N, dtype=bool)
    floored = np.zeros(N, dtype=bool)
    for g in range(len(pats)):
        rows = np.where(inv == g)[0]
        miss = [n for n, m in zip(nodes, pats[g]) if m]
        if len(miss) == len(nodes):
            skipped[rows] = True
            continue
        total = np.zeros(len(rows))
        for combo in itertools.product(*[range(len(sn[n])) for n in miss]):
            fixed = dict(zip(miss, combo))
            prod = np.ones(len(rows))
            for n in nodes:
                vars_, arr = T[n]
                idx = tuple(fixed[v] if v in fixed else codes[v][rows] for v in vars_)
                prod = prod * np.maximum(arr[idx], FACTOR_FLOOR)
            total += prod
        ff = EPS if miss else 0.0
        floored[rows] = total < ff
        ll[rows] = np.log(np.maximum(total, max(ff, 1e-300)))
    out = float(ll[~skipped].sum())
    if return_detail:
        return out, dict(per_row=ll, skipped=skipped, floored=floored, n_patterns=len(pats))
    return out
