"""Joint-query helpers for J23 and J24: one VE query per distinct reduced-evidence pattern gives w_x = P(X_j=x|e_-j), p_x = P(Y=1|e_-j, X_j=x) and p0 = P(Y=1|e_-j).
Equivalent to the separate queries of J18 (checked by the J23/J24 gates)."""
import numpy as np
from pgmpy.inference import VariableElimination

def ev_dict(S, ev_cols, row):
    return {c: S[c][i] for c, i in zip(ev_cols, row) if i >= 0}

def var_tables(eng, S, ev_cols, codes, j, extra_unobs=()):
    """returns inv (row -> pattern), p0 (n_pat,), W (n_pat, K), PX (n_pat, K) for variable index j (evidence column) with other evidence as in codes"""
    c2 = codes.copy(); c2[:, [j] + list(extra_unobs)] = -1
    u, inv = np.unique(c2, axis=0, return_inverse=True); inv = np.asarray(inv).ravel()
    v = ev_cols[j]; K = len(S[v]); P0 = np.full(len(u), np.nan); W = np.zeros((len(u), K)); PX = np.full((len(u), K), np.nan)
    for k, row in enumerate(u):
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                f = eng.query(variables=["Default", v], evidence=ev_dict(S, ev_cols, row), show_progress=False)
            vals = np.asarray(f.values, float)
            if f.variables[0] != "Default": vals = vals.T
            order = [f.state_names[v].index(s) for s in S[v]]; vals = vals[:, order]
            i1 = f.state_names["Default"].index(1); tot = vals.sum(axis=0)
            W[k] = np.where(np.isfinite(tot), tot, 0.0); P0[k] = vals[i1].sum()
            with np.errstate(invalid="ignore", divide="ignore"): PX[k] = np.where(tot > 0, vals[i1] / tot, np.nan)
            for si in np.where(~np.isfinite(PX[k]))[0]:            # a state whose joint weight underflows to 0: use the conditional query (as J18)
                try:
                    with np.errstate(invalid="ignore", divide="ignore"): PX[k][si] = eng.query(variables=["Default"], evidence={**ev_dict(S, ev_cols, row), v: S[v][si]}, show_progress=False).get_value(Default=1)
                except Exception:
                    pass
        except Exception:
            pass
    return inv, P0, W, PX

def joint_tables(eng, S, ev_cols, codes, cand_idx):
    """base evidence = rows with all candidate columns unobserved; returns inv, p0 (n_pat,), W list of flattened joint weights, PX flattened P(Y=1 | base, x) for the joint states x"""
    c2 = codes.copy(); c2[:, cand_idx] = -1; u, inv = np.unique(c2, axis=0, return_inverse=True); inv = np.asarray(inv).ravel()
    vs = [ev_cols[j] for j in cand_idx]; sizes = [len(S[v]) for v in vs]; n = int(np.prod(sizes)); P0 = np.full(len(u), np.nan); W = np.zeros((len(u), n)); PX = np.full((len(u), n), np.nan)
    for k, row in enumerate(u):
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                f = eng.query(variables=["Default"] + vs, evidence=ev_dict(S, ev_cols, row), show_progress=False)
            vals = np.asarray(f.values, float); axes = [f.variables.index(v) for v in ["Default"] + vs]; vals = np.transpose(vals, axes)
            for a, v in enumerate(vs):
                order = [f.state_names[v].index(s) for s in S[v]]; vals = np.take(vals, order, axis=a + 1)
            i1 = f.state_names["Default"].index(1); v1 = vals[i1].reshape(-1); tot = vals.sum(axis=0).reshape(-1)
            W[k] = np.where(np.isfinite(tot), tot, 0.0); P0[k] = v1.sum()
            with np.errstate(invalid="ignore", divide="ignore"): PX[k] = np.where(tot > 0, v1 / tot, np.nan)
        except Exception:
            pass
    return inv, P0, W, PX

def voi(p0, w, px, r):
    cb = np.minimum(r * p0, 1 - p0); ca = np.where(w > 0, w * np.minimum(r * px, 1 - px), 0.0).sum(axis=1); return np.maximum(cb - ca, 0.0)

def ecp(p0, w, px, r):
    """expected change probability = sum_x w_x 1[flag(p_x) != flag(p0)] ; SDP = sum_x w_x 1[flag(p_x) = flag(p0)]"""
    tau = 1 / (1 + r); f0 = (p0 >= tau)[:, None]; ch = ((np.nan_to_num(px, nan=0.0) >= tau) != f0) & (w > 0)
    e = np.where(ch, w, 0.0).sum(axis=1); sdp = np.where((~ch) & (w > 0), w, 0.0).sum(axis=1); return e, sdp
