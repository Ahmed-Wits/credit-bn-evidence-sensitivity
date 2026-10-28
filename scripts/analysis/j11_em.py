"""J11 step 1: deterministic 100-iteration EM (same as figures_final/code/s7_em_rerun.py) keeping the parameters at iterations 79 and 100.
GATE: iteration-79 CPDs equal the stored CPDs to 1e-12 and the train/test log-likelihoods equal the stored values; else stop."""
import sys, os, pickle, time, json, warnings
assert os.environ.get("PYTHONHASHSEED") == "0"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
R = os.environ["CREDIT_BN_ROOT"]; ADD = R + "/Submissions/4. Article/WRITEUP/analysis_additions"
sys.path.insert(0, R + "/shared/src")
import loglik as ll, em
d = pickle.load(open(R + "/full_variable/models/bn_final_model_iteration79.pkl", "rb"))
sn = d["state_names"]; cols = list(sn.keys()); edges = [tuple(e) for e in d["model_edges"]]
tr = pd.read_pickle(R + "/Data/processed/df_train.pkl")[cols]; te = pd.read_pickle(R + "/Data/processed/df_test.pkl")[cols]
keep = {}; hist = []
def snap(model):
    return {c: dict(variables=list(model.get_cpds(c).variables), values=np.asarray(model.get_cpds(c).values, float).copy(), state_names={k: list(v) for k, v in model.get_cpds(c).state_names.items()})
            for c in model.nodes()}
def on_it(it, model, tll, info):
    hist.append(dict(iteration=it, train_ll=tll, test_ll=float(ll.compute_log_likelihood(model, te))))
    if it in (79, 100): keep[it] = snap(model)
    if it == 79:
        worst = 0.0
        for c in d["cpds"]:
            new = model.get_cpds(c.variable)
            if list(new.variables) != list(c.variables): new = new.copy(); new.reorder_parents(list(c.variables[1:]))
            worst = max(worst, float(np.max(np.abs(np.asarray(new.values, float) - np.asarray(c.values, float)))))
        keep["gate79"] = dict(max_cpd_diff=worst, train_diff=abs(tll - d["train_ll_at_peak"]), test_diff=abs(hist[-1]["test_ll"] - d["test_ll_at_peak"]))
        print("GATE iteration 79: max |CPD diff| %.3e | train LL diff %.3e | test LL diff %.3e" % (worst, keep["gate79"]["train_diff"], keep["gate79"]["test_diff"]), flush=True)
        if worst > 1e-12 or keep["gate79"]["train_diff"] > 1e-6 or keep["gate79"]["test_diff"] > 1e-6:
            print("GATE FAILED: stop"); os._exit(2)
t0 = time.time()
model, h, reason = em.run_em(edges, tr, sn, lambda m: ll.compute_log_likelihood(m, tr), max_iter=100, ll_tol=-1e18, patience=10 ** 9, aggregate=True, on_iteration=on_it, nonmono_tol=1e18, verbose=True)
H = pd.DataFrame(hist); keep["history"] = H; keep["edges"] = edges; keep["state_names"] = sn; keep["minutes"] = (time.time() - t0) / 60
pickle.dump(keep, open(ADD + "/intermediate/em_iter79_iter100_params.pkl", "wb"), protocol=4)
print("done %.1f min | peak test LL iteration %d" % (keep["minutes"], int(H.test_ll.idxmax()) + 1), flush=True)
