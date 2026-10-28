"""S7: deterministic rerun of the 100-iteration hand-written EM for the 13-node After-acceptance BN.
Same structure (edges of the stored iteration-79 model), maximum-likelihood initialisation on the complete training rows,
no prior, no early stop, 100 iterations. Uses the saved em.py / loglik.py (ported from the locked kernel code).
Records train and test log-likelihood at every iteration. Writes s7_em_history.csv next to this script."""
import sys, os, pickle, time, json, warnings
os.environ.setdefault("PYTHONHASHSEED", "0")
assert os.environ["PYTHONHASHSEED"] == "0"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = os.environ["CREDIT_BN_ROOT"]
sys.path.insert(0, R + "/shared/src")
import loglik as ll, em
OUTCSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1] if len(sys.argv) > 1 else "s7_em_history.csv")
d = pickle.load(open(R + "/full_variable/models/bn_final_model_iteration79.pkl", "rb"))
sn = d["state_names"]; cols = list(sn.keys()); edges = [tuple(e) for e in d["model_edges"]]
tr = pd.read_pickle(R + "/Data/processed/df_train.pkl")[cols]; te = pd.read_pickle(R + "/Data/processed/df_test.pkl")[cols]
rows = []; cpd79 = {}
def on_it(it, model, tll, info):
    tell = float(ll.compute_log_likelihood(model, te))
    rows.append(dict(iteration=it, train_ll=tll, test_ll=tell, seconds=info["seconds"]))
    if it == 79:
        for c in d["cpds"]:
            new = model.get_cpds(c.variable)
            # align axes by variable/state names
            a = np.asarray(c.values, float); 
            if list(new.variables) != list(c.variables):
                new = new.copy(); new.reorder_parents([v for v in c.variables[1:]])
            cpd79[c.variable] = float(np.max(np.abs(np.asarray(new.values, float) - a)))
model, hist, reason = em.run_em(edges, tr, sn, lambda m: ll.compute_log_likelihood(m, tr), max_iter=100, ll_tol=-1e18, patience=10 ** 9,
                                aggregate=True, on_iteration=on_it, nonmono_tol=1e18, verbose=True)
df = pd.DataFrame(rows); df.to_csv(OUTCSV, index=False)
json.dump(dict(max_abs_cpd_diff_at_79=cpd79), open(OUTCSV.replace(".csv", "_cpd79.json"), "w"), indent=1)
pk = int(df.test_ll.idxmax()) + 1
print("peak test LL at iteration", pk, "| it79 train %.6f test %.6f" % (df.train_ll[78], df.test_ll[78]))
print("stored  train %.6f test %.6f" % (d["train_ll_at_peak"], d["test_ll_at_peak"]))
