"""Convergence checks for sklearn estimators with n_iter_ (e.g. LogisticRegression).
assert_converged: n_iter_ < max_iter and no ConvergenceWarning. fit_converged: fit under a warning catcher, then assert."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning


def assert_converged(model, max_iter=None, caught=None, name="model"):
    mi = model.max_iter if max_iter is None else max_iter
    ni = int(np.max(model.n_iter_))
    conv_w = [w for w in (caught or []) if issubclass(w.category, ConvergenceWarning)]
    assert ni < mi, "%s did not converge: n_iter_ %d reached max_iter %d" % (name, ni, mi)
    assert not conv_w, "%s raised %d ConvergenceWarning(s)" % (name, len(conv_w))
    return ni


def fit_converged(model, X, y, name="model", **fit_kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y, **fit_kw)
    assert_converged(model, model.max_iter, caught, name)
    return model
