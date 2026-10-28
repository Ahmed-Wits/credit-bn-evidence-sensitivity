"""Test-set masking for RQ3 (paired across models; one mask per contribution/variable/level/mechanism/seed).
make_mask(n_rows, level, mechanism, seed, cond_codes=None) -> (mask, realised_fraction); True = masked.
'mcar': each row masked independently with p = level.
'dependent': p_i = min(1, s * w_i / mean(w)), w_i = cond_codes_i + 1 (cond_codes in 0..4); one uniform u_i per row from
RandomState(seed); s is bisected so the REALISED masked fraction of (u < p) is within 0.005 of level (in practice ~1/n).
Conditioning variable rule (record in notes): Debt Ratio bin for every masked variable, EXCEPT Age bin when the masked
variable is Debt Ratio (selection)."""
import os
assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED != 0: wrong kernel"
import numpy as np

SEEDS = list(range(42, 52))          # 10 seeds
SEEDS_FALLBACK = list(range(42, 47)) # 5-seed fallback
LEVELS_TOL = 0.005


def cond_variable(masked_var):
    """Name of the always-observed conditioning variable for dependent missingness."""
    return "Age (bin)" if masked_var.replace(" (bin)", "") == "Debt Ratio" else "Debt Ratio (bin)"


def bin_codes(series, states):
    """0-based bin index of each row in the given ordered states (labels as strings). Raises on unseen/NaN values."""
    import pandas as pd
    c = pd.Index([str(s) for s in states]).get_indexer(series.astype(str))
    if (c < 0).any():
        raise ValueError("value not in states")
    return c


def make_mask(n_rows, level, mechanism, seed, cond_codes=None):
    if not (0.0 <= level <= 1.0):
        raise ValueError("level must be in [0, 1]")
    rng = np.random.RandomState(seed)
    u = rng.random_sample(n_rows)
    if mechanism == "mcar":
        m = u < level
        return m, float(m.mean())
    if mechanism != "dependent":
        raise ValueError("mechanism must be 'mcar' or 'dependent'")
    if cond_codes is None:
        raise ValueError("cond_codes required for 'dependent'")
    cc = np.asarray(cond_codes)
    if len(cc) != n_rows or cc.min() < 0 or cc.max() > 4:
        raise ValueError("cond_codes must be integers 0..4, one per row")
    w = cc.astype(float) + 1.0
    base = w / w.mean()
    frac = lambda s: float((u < np.minimum(1.0, s * base)).mean())
    lo, hi = 0.0, 1.0
    while frac(hi) < level and hi < 1e6:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if frac(mid) < level:
            lo = mid
        else:
            hi = mid
    s = hi if abs(frac(hi) - level) <= abs(frac(lo) - level) else lo
    m = u < np.minimum(1.0, s * base)
    r = float(m.mean())
    assert abs(r - level) <= LEVELS_TOL, "realised fraction %.4f not within %.3f of %.3f" % (r, LEVELS_TOL, level)
    return m, r
