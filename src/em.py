"""Custom EM for discrete Bayesian networks with missing values (VariableElimination E-step).
compute_row_weights and compute_new_cpds are ported verbatim from the locked kernel code.
run_em adds: state_names passed in, per-iteration callback, train-LL stopping rule, optional row aggregation."""
import os
assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED != 0: wrong kernel"
import itertools, time
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination


def compute_row_weights(model, row, missing_vars, infer_engine):
    """
    For a single row, return a list of (full_assignment_dict, weight) tuples.
    - If missing_vars is empty: hard assignment, weight 1.0
    - If missing_vars non-empty: soft weights from VariableElimination posterior
    """
    observed = {var: row[var] for var in row.index if var not in missing_vars and pd.notnull(row[var])}

    if not missing_vars:
        # Complete row: hard weight
        return [(observed, 1.0)]

    # Query joint posterior over missing variables given observed evidence
    result = infer_engine.query(variables=list(missing_vars), evidence=observed, show_progress=False)

    # result is a DiscreteFactor over missing_vars; enumerate all state combinations
    weighted_assignments = []
    state_names = result.state_names
    all_states = [state_names[var] for var in result.variables]

    for combo in itertools.product(*all_states):
        assignment = dict(zip(result.variables, combo))
        prob = result.get_value(**assignment)
        if prob > 0:
            full_assignment = {**observed, **assignment}
            weighted_assignments.append((full_assignment, prob))

    return weighted_assignments


def compute_new_cpds(model, all_weighted_assignments, state_names):
    new_cpds = []

    for node in model.nodes():
        parents = list(model.get_parents(node))
        node_states = state_names[node]
        n_node_states = len(node_states)

        if not parents:
            counts = np.zeros(n_node_states)
            for assignment, weight in all_weighted_assignments:
                val = assignment[node]
                counts[node_states.index(val)] += weight

            total = counts.sum()
            probs = counts / total if total > 0 else np.ones(n_node_states) / n_node_states
            cpd = TabularCPD(node, n_node_states, probs.reshape(-1, 1),
                              state_names={node: node_states})
        else:
            parent_states = [state_names[p] for p in parents]
            parent_combos = list(itertools.product(*parent_states))
            n_parent_combos = len(parent_combos)

            counts = np.zeros((n_node_states, n_parent_combos))
            combo_index = {combo: i for i, combo in enumerate(parent_combos)}

            for assignment, weight in all_weighted_assignments:
                node_val = assignment[node]
                parent_vals = tuple(assignment[p] for p in parents)
                row_i = node_states.index(node_val)
                col_i = combo_index[parent_vals]
                counts[row_i, col_i] += weight

            col_sums = counts.sum(axis=0)
            col_sums_safe = np.where(col_sums > 0, col_sums, 1)
            probs = counts / col_sums_safe
            zero_cols = np.where(col_sums == 0)[0]
            if len(zero_cols) > 0:
                probs[:, zero_cols] = 1.0 / n_node_states

            cpd = TabularCPD(node, n_node_states, probs,
                              evidence=parents,
                              evidence_card=[len(state_names[p]) for p in parents],
                              state_names={**{node: node_states}, **{p: state_names[p] for p in parents}})

        new_cpds.append(cpd)

    return new_cpds


def _mle_estimator():
    from pgmpy.parameter_estimator import DiscreteMLE
    return DiscreteMLE


def init_model(edges, data, state_names):
    """Initialise from complete cases by plain MLE (as in the locked run)."""
    cols = [c for c in data.columns if c in state_names]
    complete = data[cols].dropna()
    model = DiscreteBayesianNetwork([tuple(e) for e in edges])
    for n in state_names:
        if n not in model.nodes():
            model.add_node(n)
    model.fit(complete, estimator=_mle_estimator()(state_names=state_names))
    return model, len(complete)


def _row_table(d, cols, state_names, aggregate):
    codes = np.column_stack([pd.Index(state_names[c]).get_indexer(d[c]) for c in cols])
    for j, c in enumerate(cols):
        bad = (codes[:, j] < 0) & d[c].notna().to_numpy()
        if bad.any():
            raise ValueError("state not in state_names for %s: %s" % (c, d[c][bad].unique()[:5]))
    if aggregate:
        u, cnt = np.unique(codes, axis=0, return_counts=True)
    else:
        u, cnt = codes, np.ones(len(codes), dtype=np.int64)
    rows = []
    for r in u:
        rows.append(pd.Series({c: (state_names[c][k] if k >= 0 else np.nan) for c, k in zip(cols, r)}, dtype=object))
    return rows, cnt


def run_em(edges, data, state_names, ll_fn, max_iter=200, ll_tol=0.01, patience=3,
           aggregate=True, on_iteration=None, nonmono_tol=1e-6, verbose=True):
    """ll_fn(model) -> train log-likelihood (float). on_iteration(it, model, train_ll, info) called after each M-step.
    Stops when train-LL gain < ll_tol for `patience` consecutive iterations, or at max_iter.
    Raises RuntimeError if train LL decreases by more than nonmono_tol."""
    cols = [c for c in data.columns if c in state_names]
    d = data[cols]
    model, n_cc = init_model(edges, d, state_names)
    rows, cnt = _row_table(d, cols, state_names, aggregate)
    if verbose:
        print("init from %d complete cases of %d rows | E-step units: %d (aggregate=%s)" % (n_cc, len(d), len(rows), aggregate))
    hist, prev, flat, reason = [], None, 0, "max_iter"
    for it in range(1, max_iter + 1):
        t0 = time.time()
        infer = VariableElimination(model)
        allw, bad_rows = [], 0
        for r, c in zip(rows, cnt):
            mv = [v for v in cols if pd.isnull(r[v])]
            w = compute_row_weights(model, r, mv, infer)
            if abs(sum(x[1] for x in w) - 1.0) > 1e-9:
                bad_rows += int(c)
            allw.extend((a, wt * c) for a, wt in w)
        new_cpds = compute_new_cpds(model, allw, state_names)
        model.cpds = []
        model.add_cpds(*new_cpds)
        ll = float(ll_fn(model))
        gain = None if prev is None else ll - prev
        if gain is not None and gain < -nonmono_tol:
            raise RuntimeError("train LL decreased at iteration %d: %.9f -> %.9f" % (it, prev, ll))
        info = dict(iteration=it, train_ll=ll, gain=gain, seconds=time.time() - t0, bad_weight_rows=bad_rows)
        hist.append(info)
        if on_iteration is not None:
            on_iteration(it, model, ll, info)
        if verbose and (it <= 3 or it % 10 == 0):
            print("it %3d | train LL %.6f | gain %s | %.1fs | rows with weight-sum != 1: %d" % (
                it, ll, "n/a" if gain is None else "%.6f" % gain, info["seconds"], bad_rows))
        if gain is not None:
            flat = flat + 1 if gain < ll_tol else 0
            if flat >= patience:
                reason = "gain < %g for %d consecutive iterations" % (ll_tol, patience)
                break
        prev = ll
    return model, hist, reason
