"""J12 aggregation: per-seed and mean/min/max tables, ledger rows, report part."""
from addcommon import *
p = Part("j12")
SEEDS = [1, 2, 3, 4, 5]; STG = {"application": ("Application", "selection"), "after": ("After acceptance", "monitoring")}
J = {}; G = {}
for st in STG:
    for s in SEEDS: J[(st, s)] = json.load(open(f"{ADD}/j12/{st}_seed{s}.json"))
    G[st] = json.load(open(f"{ADD}/j12/{st}_seed42_gate.json"))
    for c in G[st]["checks"]: p.gate(f"{STG[st][0]}, seed-42 reference run through the new code: " + c["item"], c["ok"], c["detail"])
for st in STG:
    for s in SEEDS:
        p.check(f"{STG[st][0]} seed {s}: stratified 80/20 split, train default rate {J[(st,s)]['default_rate_train']:.4f} / test {J[(st,s)]['default_rate_test']:.4f}", abs(J[(st, s)]["default_rate_train"] - 0.0668) < 0.001 and J[(st, s)]["n_train"] == 119999)
REF_BL = {"application": {"Age (bin)", "Debt Ratio (bin)"}, "after": {"Credit Utilisation Rate (bin)", "90+ Days Late (bin)", "60-89 Days Past Due (bin)", "30-59 Days Past Due (bin)"}}
short = lambda n: n.replace(" (bin)", "")
def srcf(st, s): return f"Submissions/4. Article/WRITEUP/analysis_additions/j12/{st}_seed{s}.json"
CIM = "95% percentile interval, 2000 test-row bootstrap resamples (seed 42), paired across models; predictions and thresholds fixed (J2 procedure)"
CIA = "95% percentile interval, 2000 paired test-row bootstrap resamples (RandomState(42), randint), as Table 6"
EQ = {}
def agg(vals):
    a = np.array(vals); return float(a.mean()), float(a.min()), float(a.max())
def compute_A(st):
    A = {}
    for m in ("BN", "LR", "RF"): A[f"AUC {m}"] = agg([J[(st, s)]["auc"][m] for s in SEEDS])
    for m in ("LR", "RF"): A[f"AUC difference BN minus {m}"] = agg([J[(st, s)][f"auc_BN_minus_{m}"]["observed"] for s in SEEDS])
    for r in (5, 10):
        for m in ("BN", "LR", "RF"): A[f"saving {m} r={r} (%)"] = agg([J[(st, s)]["saving"][f"{m}_r{r}"]["point"] for s in SEEDS])
        for m in ("LR", "RF"): A[f"saving difference BN minus {m} r={r} (points)"] = agg([J[(st, s)]["saving_diff"][f"BN_minus_{m}_r{r}"]["point"] for s in SEEDS])
    return A
AGGCSV = "Submissions/4. Article/WRITEUP/analysis_additions/j12_aggregate.csv"
pd.DataFrame([dict(stage=STG[st][0], quantity=k, mean=v[0], minimum=v[1], maximum=v[2]) for st in STG for k, v in compute_A(st).items()]).to_csv(R + "/" + AGGCSV, index=False)
md = ["## J12. Split-seed robustness (split seeds 1 to 5)", "",
      "**Method.** For each split seed the whole pipeline was redone with the new split: stratified 80/20 split (`train_test_split`, `random_state` = seed), train-only quantile bin edges, **structure search relearned** (15 random-restart hill climbs, BIC, complete training rows only, `shared/src/structure.py`), BN parameters by EM (Application: the stored stopping rule, tolerance 0.01, patience 3; After acceptance: a fixed 100 iterations, the **iteration-100** model, no test-based selection), LR and RF with the Table S-Settings parameters (no retuning; design matrix and imputer refit on the new training rows), thresholds as in the cost curves (BN 1/(1+r); LR and RF the cost-minimising value on 5-fold out-of-fold training predictions, ties to the highest; After-acceptance LR OOF with the converged lbfgs refit as in the original cost curves). Test evaluation: AUC; paired bootstrap AUC differences (2,000 resamples, RandomState(42), After acceptance on the rows where the BN is defined); savings at r = 5 and 10 with the J2 bootstrap; Markov blanket of Default; mean S_exp of the blanket variables. Code: `code/split_robust.py`, `code/j12_run.py`, `code/j12_agg.py`; per-seed results in `j12/`.", "",
      "**Gate (PASSED).** Running split seed 42 through the same code reproduces the stored seed-42 results exactly: split, bin edges, structure (Application 13 edges, seed 55, BIC -788,831.4158; After acceptance 22 edges, seed 53, BIC -1,026,115.7308), Application EM (9 iterations, train LL -998,922.732607), BN, LR and RF test probabilities (max abs diff 0 / 0 / 0 at Application; 6e-15 / 0 / 3e-16 After acceptance, because the iteration-100 BN equals the stored iteration-79 BN, J11), AUCs (0.6627 / 0.6904 / 0.6926 and 0.8519 / 0.8618 / 0.8603), the Table 7 savings at r = 5 and 10 at both stages, the seed-42 blankets and the mean S_exp values. Time per seed: Application about 3 to 4 minutes, After acceptance about 13 minutes (EM about 7 minutes); the runs were executed in parallel.", ""]
for st, (lab, led_stage) in STG.items():
    md += [f"### {lab}: per seed", "", "| Seed | edges | BIC | Markov blanket of Default | = seed-42 blanket? | AUC BN | AUC LR | AUC RF | BN minus LR [95% interval] | BN minus RF [95% interval] |", "|---|---|---|---|---|---|---|---|---|---|"]
    stored = REF_BL[st]; n_eq = 0
    for s in SEEDS:
        d = J[(st, s)]; bl = set(d["blanket"]["blanket"]) - {"M_Income", "M_Dependents"}; eq = bl == stored; n_eq += eq; EQ[(st, s)] = eq
        a = d["auc"]; l = d["auc_BN_minus_LR"]; r = d["auc_BN_minus_RF"]
        md.append(f"| {s} | {d['structure']['n_edges']} | {d['structure']['best_bic']:.1f} | {', '.join(short(x) for x in sorted(bl))} | {'yes' if eq else 'NO'} | {a['BN']:.4f} | {a['LR']:.4f} | {a['RF']:.4f} | {sgn(l['observed'],4)} [{sgn(l['lo'],4)}, {sgn(l['hi'],4)}] | {sgn(r['observed'],4)} [{sgn(r['lo'],4)}, {sgn(r['hi'],4)}] |")
    md += ["", f"Seed-42 blanket (reference): {', '.join(short(x) for x in sorted(stored))}. **{n_eq} of the 5 split seeds give the same blanket.**", ""]
    md += ["| Seed | r | BN | LR | RF | BN minus LR [95% interval] | BN minus RF [95% interval] |", "|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        d = J[(st, s)]
        for r in (5, 10):
            sv = d["saving"]; sd = d["saving_diff"]
            md.append(f"| {s} | {r} | " + " | ".join(f"{sv[f'{m}_r{r}']['point']:.1f}% [{sv[f'{m}_r{r}']['lo']:.1f}, {sv[f'{m}_r{r}']['hi']:.1f}]" for m in ("BN", "LR", "RF")) + " | " + " | ".join(f"{sgn(sd[f'BN_minus_{m}_r{r}']['point'],2,True)} [{sgn(sd[f'BN_minus_{m}_r{r}']['lo'],2)}, {sgn(sd[f'BN_minus_{m}_r{r}']['hi'],2)}]" for m in ("LR", "RF")) + " |")
    md += [""]
    A = compute_A(st)
    md += ["**Mean, minimum and maximum over the 5 split seeds**", "", "| Quantity | mean | minimum | maximum |", "|---|---|---|---|"]
    for k, (mn, lo, hi) in A.items():
        dd = 4 if k.startswith("AUC") else 2
        md.append(f"| {k} | {mn:.{dd}f} | {lo:.{dd}f} | {hi:.{dd}f} |")
    nex_auc = {m: sum(J[(st, s)][f"auc_BN_minus_{m}"]["hi"] < 0 or J[(st, s)][f"auc_BN_minus_{m}"]["lo"] > 0 for s in SEEDS) for m in ("LR", "RF")}
    nex_sav = {(m, r): sum(J[(st, s)]["saving_diff"][f"BN_minus_{m}_r{r}"]["hi"] < 0 or J[(st, s)]["saving_diff"][f"BN_minus_{m}_r{r}"]["lo"] > 0 for s in SEEDS) for m in ("LR", "RF") for r in (5, 10)}
    md += ["", f"Seeds (of 5) whose 95% interval excludes 0: AUC difference BN minus LR {nex_auc['LR']}, BN minus RF {nex_auc['RF']}; saving difference at r = 5: BN minus LR {nex_sav[('LR',5)]}, BN minus RF {nex_sav[('RF',5)]}; at r = 10: BN minus LR {nex_sav[('LR',10)]}, BN minus RF {nex_sav[('RF',10)]}.", ""]
    sx = {}
    for s in SEEDS:
        for v, dd in J[(st, s)]["sexp"].items(): sx.setdefault(v, {})[s] = dd["mean"]
    md += ["**Mean S_exp of the blanket variables (seeds where the variable is in the blanket)**", "", "| Variable | seeds in blanket | mean S_exp per seed | mean over those seeds | min | max |", "|---|---|---|---|---|---|"]
    for v, dct in sorted(sx.items(), key=lambda kv: -len(kv[1])):
        vals = list(dct.values()); md.append(f"| {short(v)} | {len(dct)} | {', '.join(f's{k}: {x:.4f}' for k, x in dct.items())} | {np.mean(vals):.4f} | {min(vals):.4f} | {max(vals):.4f} |")
    md += [""]
    for s in SEEDS:
        d = J[(st, s)]; f_ = srcf(st, s); lv_ = f"seed {s}"; note0 = f"split seed {s} (not 42): split, bin edges, structure (relearned on complete training rows), BN parameters ({'EM, stored stopping rule' if st=='application' else 'EM 100 iterations, iteration-100 model'}), LR and RF (Table S-Settings, no retuning) and OOF thresholds all redone; code/j12_run.py"
        for m in ("BN", "LR", "RF"):
            p.row(led_stage, f"Split-seed robustness, {lab}: test AUC, {m}, split seed {s}", d["auc"][m], f"{d['auc'][m]:.4f}", "AUC", f_, f"auc.{m}", derived="Y", notes=note0 + ("; LR and RF on all 30,000 test rows, BN on the rows with a defined posterior" if st == "after" else ""), display_rule="auc4", n=d["n_bn_defined"] if m == "BN" else d["n_test"], model=m, level=lv_, caveat="C5")
        for m in ("LR", "RF"):
            x = d[f"auc_BN_minus_{m}"]; p.row(led_stage, f"Split-seed robustness, {lab}: test AUC difference BN minus {m}, split seed {s}", x["observed"], f"{sgn(x['observed'],4)} [{sgn(x['lo'],4)}, {sgn(x['hi'],4)}]", "AUC difference", f_, f"auc_BN_minus_{m}", derived="Y", ci_lo=x["lo"], ci_hi=x["hi"], ci_method=CIA,
                  notes=note0 + f"; observed difference on the common rows (bootstrap mean {x['boot_mean']:+.6f})", display_rule="d4", n=d["n_bn_defined"], model="BN", comparator=m, level=lv_, tested=("yes-tie" if x["lo"] <= 0 <= x["hi"] else "yes-significant"))
        for r in (5, 10):
            for m in ("BN", "LR", "RF"):
                x = d["saving"][f"{m}_r{r}"]; p.row(led_stage, f"Split-seed robustness, {lab}: cost saving, {m}, r={r}, split seed {s}", x["point"] / 100, f"{x['point']:.1f}% [{x['lo']:.1f}%, {x['hi']:.1f}%]", "%", f_, f"saving.{m}_r{r}", derived="Y", ci_lo=x["lo"] / 100, ci_hi=x["hi"] / 100, ci_method=CIM,
                      notes=note0 + f"; thresholds BN 1/(1+r), LR {d['tau']['LR'][str(r)]:.2f}, RF {d['tau']['RF'][str(r)]:.2f}", display_rule="pct1", n=d["n_bn_defined"], model=m, level=f"r={r}, {lv_}", caveat="C3;C5")
            for m in ("LR", "RF"):
                x = d["saving_diff"][f"BN_minus_{m}_r{r}"]; p.row(led_stage, f"Split-seed robustness, {lab}: cost saving difference BN minus {m}, r={r}, split seed {s} (percentage points)", x["point"], f"{sgn(x['point'],2,True)} [{sgn(x['lo'],2)}, {sgn(x['hi'],2)}]", "percentage points", f_, f"saving_diff.BN_minus_{m}_r{r}", derived="Y", ci_lo=x["lo"], ci_hi=x["hi"], ci_method=CIM,
                      notes=note0, display_rule="txt", n=d["n_bn_defined"], model="BN", comparator=m, level=f"r={r}, {lv_}", tested=("yes-tie" if x["lo"] <= 0 <= x["hi"] else "yes-significant"), caveat="C3;C5")
        bl = sorted(short(x) for x in set(d["blanket"]["blanket"]) - {"M_Income", "M_Dependents"})
        p.row(led_stage, f"Split-seed robustness, {lab}: Markov blanket of Default found, split seed {s} (equal to the seed-42 blanket: {'yes' if EQ[(st, s)] else 'no'})", ", ".join(bl), ", ".join(bl), "variables", f_, "blanket.blanket", derived="Y",
              notes=note0 + f"; structure {d['structure']['n_edges']} learned edges, BIC {d['structure']['best_bic']:.2f}, best restart seed {d['structure']['best_seed']}; parents {', '.join(short(x) for x in d['blanket']['parents'])}; children {', '.join(short(x) for x in d['blanket']['children'])}; seed-42 blanket {', '.join(sorted(short(x) for x in REF_BL[st]))}", display_rule="txt", n=d["n_train"], model="BN", level=lv_, caveat="C8")
        for v_, dd in d["sexp"].items():
            p.row(led_stage, f"Split-seed robustness, {lab}: mean S_exp, {short(v_)}, split seed {s}", dd["mean"], f"{dd['mean']:.4f}", "probability", f_, f"sexp.{v_}.mean", derived="Y", notes=note0 + f"; over {dd['n']:,} test rows where the variable is observed and the posterior is defined; the variable is in this seed's blanket", display_rule="s4", n=dd["n"], model="BN", variable=short(v_), level=lv_, caveat="C7")
    for k, (mn, lo, hi) in A.items():
        dd = 4 if k.startswith("AUC") else 2
        for nm, val in (("mean", mn), ("minimum", lo), ("maximum", hi)):
            p.row(led_stage, f"Split-seed robustness, {lab}: {k}, {nm} over split seeds 1 to 5", val / (100 if "(%)" in k else 1), (f"{val:.4f}" if dd == 4 else f"{val:.2f}"), ("AUC" if k.startswith("AUC") and "difference" not in k else ("AUC difference" if k.startswith("AUC") else ("%" if "(%)" in k else "percentage points"))), AGGCSV, f"aggregate over {lab} seeds 1-5 | {k} | {nm}", derived="Y", notes="aggregate of the five split-seed runs (seed 42 excluded); point estimates, intervals are per seed", display_rule="txt", n=5, model="BN", caveat="C3;C5")
    p.row(led_stage, f"Split-seed robustness, {lab}: number of the 5 split seeds with the same Markov blanket as seed 42", n_eq, str(n_eq), "seeds", AGGCSV, "aggregate | blanket equal to seed 42", derived="Y", notes=f"seed-42 blanket {', '.join(sorted(short(x) for x in REF_BL[st]))}", display_rule="int", n=5, model="BN", caveat="C8")
    md += ["Blankets found: " + "; ".join(f"seed {s}: {{{', '.join(sorted(short(x) for x in set(J[(st, s)]['blanket']['blanket']) - {'M_Income', 'M_Dependents'}))}}}" for s in SEEDS), ""]
md += ["### Imputed Number of Dependents (design matrix of LR and RF)", "", "In the LR/RF design the IterativeImputer can return negative Number of Dependents for rows with missing Dependents; the locked binning function (`labels[int(x)]`) indexes the label list with that negative number. For seed 42 the values stay between -1 and -4 and the locked behaviour is reproduced exactly; for other seeds a value of -5 or lower occurs (counts of imputed values <= -5 are in each result file, `imputed_dependents_wrap`) and the function failed with an IndexError. The rq3 v2 'wrap' rule (`int(x) % 4`, identical to the locked function wherever that is defined) was used, as in the RQ3 grid; it affects at most one row per split.", "", checks_md(p), ""]
p.write("\n".join(md))
for st in STG: print(st, "blankets equal to seed 42:", sum(EQ[(st, s)] for s in SEEDS), "of 5")
