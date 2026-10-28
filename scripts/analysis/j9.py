"""J9: which published numbers use the locked After-acceptance thresholds (BN 0.17, LR 0.69, RF 0.67) and which use 1/(1+r) with out-of-fold LR/RF thresholds. Read-only."""
from addcommon import *
import pickle
sys.path.insert(0, R + "/shared/src")
import decision_value as dv
p = Part("j9")
L = lambda f: pickle.load(open(R + "/" + f, "rb"))
comb = pd.read_csv(R + "/selection/tables/sel_decision_value_combined.csv")
# ---------- Table 5 / Section 4.2.2: flip rates After acceptance
pr = L("full_variable/per_row/vo_partA_retraction_per_row.pkl") if False else pd.read_pickle(R + "/full_variable/per_row/vo_partA_retraction_per_row.pkl")
tab = pd.read_csv(R + "/full_variable/tables/rq2_ext_partA_table.csv").set_index("variable")
IDF = {"Credit Utilisation Rate": "N-MON-017", "30-59 Days Past Due": "N-MON-019", "90+ Days Late": "N-MON-021", "60-89 Days Past Due": "N-MON-023"}
flip = {}
for v, lid in IDF.items():
    g = pr[pr.variable == v + " (bin)"]; g = g[g.baseline_prob.notna() & g.retracted_prob.notna()]
    row = []
    for tau in (0.17, 1 / 6, 1 / 11):
        fr = float(((g.baseline_prob >= tau) != (g.retracted_prob >= tau)).mean()); row.append(fr)
    flip[v] = (row, len(g))
    p.check(f"Table 5 {v}: flip rate at tau = 0.17 ({100*row[0]:.2f}%) equals the stored Part A rate and ledger {lid} ({LED.loc[lid,'display'].split(' ')[0]})", abs(row[0] - tab.loc[v + " (bin)", "flip_rate"]) < 1e-12 and f"{100*row[0]:.2f}%" == LED.loc[lid, "display"].split(" ")[0], f"n = {len(g):,}")
# ---------- Table 8: stage agreement thresholds
sa = pd.read_csv(R + "/selection/tables/sel_stage_agreement.csv"); a10 = sa[sa.r == 10]
p.check("Table 8 (r = 10) uses tau = 1/11 at BOTH stages: sel_stage_agreement.csv column tau = 0.0909 and the code flags `ps >= 1/(1+r)` and `pm >= 1/(1+r)` (08_story_extras cell index 4)", bool(np.allclose(a10.tau, 1 / 11)))
d_ = a10.set_index("group").n_defaulters
p.check("Table 8 numbers: flagged at application = %d + %d = %d (article 1,023), missed at application and flagged after acceptance %d (593), neither %d (387), all %d (2,003)" % (d_["yes-yes"], d_["yes-no"], d_["yes-yes"] + d_["yes-no"], d_["no-yes"], d_["no-no"], d_.sum()),
        (d_["yes-yes"] + d_["yes-no"], d_["no-yes"], d_["no-no"], d_.sum()) == (1023, 593, 387, 2003))
sb = L("selection/predictions/sel_bn_test_predictions.pkl"); bb = L("full_variable/predictions/bn_test_predictions.pkl")["bn_test_probs"]; v = bb.notna().to_numpy(); idx = bb.index[v]
ps = pd.Series(np.asarray(sb["p"], float), index=pd.Index(sb["index"])).loc[idx]; pm = bb.loc[idx]; y = pd.Series(np.asarray(sb["y"]).astype(int), index=pd.Index(sb["index"])).loc[idx]
def split(ta, tm):
    fa, fm = ps >= ta, pm >= tm
    return int(y[fa].sum()), int(y[~fa & fm].sum()), int(y[~fa & ~fm].sum())
sens_rows = [("1/11 (as in the article)", "1/11", split(1 / 11, 1 / 11)), ("Application 1/11, After acceptance locked 0.17", "1/11 / 0.17", split(1 / 11, 0.17)), ("Application 1/11, After acceptance 1/6", "1/11 / 1/6", split(1 / 11, 1 / 6))]
p.check("recomputed Table 8 split at 1/11 equals the article", sens_rows[0][2] == (1023, 593, 387), str(sens_rows[0][2]))
# ---------- Table 7 / Fig 4: thresholds
T = {}
for st, lab in (("selection", "Application"), ("monitoring", "After acceptance")):
    for m in ("BN", "LR", "RF"):
        T[(st, m)] = {r: float(comb[(comb.contribution == st) & (comb.model == m) & (comb.r == r)].tau.iloc[0]) for r in (5, 10)}
locked = {"BN": 0.17, "LR": 0.69, "RF": 0.67}
p.check("Fig 4 / Table 7 BN thresholds are 1/(1+r) at both stages (r = 5, 10)", all(abs(T[(s, "BN")][r] - 1 / (1 + r)) < 1e-12 for s in ("selection", "monitoring") for r in (5, 10)))
p.check("none of the Fig 4 / Table 7 After-acceptance thresholds at r = 5 and 10 is a locked value (BN 0.17, LR 0.69, RF 0.67)", all(abs(T[("monitoring", m)][r] - locked[m]) > 1e-9 for m in ("BN", "LR", "RF") for r in (5, 10)),
        "After acceptance taus r=5: " + ", ".join(f"{m} {T[('monitoring', m)][5]:.4g}" for m in ("BN", "LR", "RF")) + "; r=10: " + ", ".join(f"{m} {T[('monitoring', m)][10]:.4g}" for m in ("BN", "LR", "RF")))
# what Table 7 would be at the locked thresholds (r=5, After acceptance, 29,998 rows)
ym = y.to_numpy(); Pm = {"BN": pm.to_numpy(float), "LR": np.asarray(L("full_variable/predictions/ablation_predictions.pkl")["LR_all"], float)[v], "RF": np.asarray(L("full_variable/predictions/ablation_predictions.pkl")["RF_all"], float)[v]}
tr = dv.trivial(ym, 5); lk = {}
for m in ("BN", "LR", "RF"):
    c_lock = dv.cost(ym, Pm[m] >= locked[m], 5); c_tab = dv.cost(ym, Pm[m] >= T[("monitoring", m)][5], 5)
    lk[m] = (100 * dv.saving(c_tab, tr["cheaper"]), 100 * dv.saving(c_lock, tr["cheaper"]))
# the stored Table 7 numbers
t7 = {"BN": 35.2, "LR": 35.0, "RF": 35.1}
p.check("Table 7 After acceptance r = 5 (35.2 / 35.0 / 35.1) equals the savings at the Fig 4 thresholds (the locked thresholds would give a different LR and RF saving; the BN gives the same saving at 0.17 and at 1/6 at r = 5)", all(round(lk[m][0], 1) == t7[m] for m in t7), "; ".join(f"{m}: Fig 4 thr {lk[m][0]:.2f}% vs locked thr {lk[m][1]:.2f}%" for m in lk))
md = ["## J9. Which published numbers use the locked After-acceptance thresholds", "",
      "**Locked thresholds.** BN 0.17, LR 0.69, RF 0.67: minimise 5 x FN + FP on the **test** set (0.01 grid; BN on the 29,998 rows where it is defined); see J4(a). **1/(1+r) with out-of-fold LR/RF thresholds:** BN fixed a priori at 1/(1+r); LR and RF the cost-minimising value on out-of-fold **training** predictions at each r (J4(b)).", "",
      "| Result (article location) | Threshold used | Threshold chosen on the test set? |", "|---|---|---|",
      "| Application flip rates, Table 3 and Section 4.1.2 (r = 5, 10, 15, 20); the 1/6 line in Fig 3 | BN 1/(1+r) (1/6, 1/11, 1/16, 1/21) | No, fixed in advance |",
      "| **After-acceptance flip rates, Table 5 and Section 4.2.2** (4.58%, 4.63%, 3.09%, 1.40%) | **Locked BN threshold 0.17** (reproduced: stored Part A rates at tau = 0.17) | **Yes** (swept on the test set at 5:1) |",
      "| Section 3.6 sentence 'the after-acceptance flip rates used the BN's locked threshold, 0.17' and Section 5.1.2 ('locked threshold of 0.17 was close to tau_5 = 1/6') | Locked BN 0.17 | Yes |",
      "| Table 7 and Fig 4, Application (r = 1 to 20; 5 and 10 in Table 7) | BN 1/(1+r); LR and RF out-of-fold-training thresholds per r (highest of tied minima) | No (training data); BN fixed |",
      "| Table 7 and Fig 4, **After acceptance** | BN 1/(1+r); LR and RF out-of-fold-training thresholds per r. **The locked 0.17 / 0.69 / 0.67 are NOT used** (asserted: the table equals the savings at the Fig 4 thresholds; at the locked LR and RF thresholds the r = 5 savings would be 35.37% and 35.36%) | Thresholds: no. But the After-acceptance BN is the EM iteration-79 model, chosen by test log-likelihood, and the OOF predictions use an imputer fitted on all training rows (features only) |",
      "| Section 4.2.4 (BN leads/trails: +0.83 at r=3, -1.92 at r=2, -1.81 at r=11) and Section 4.3.1 (Application: LR first saves at r=6, RF at r=4, BN at r=8; shortfalls 7.49 at r=8 and 6.24 at r=10) | Same as Fig 4 (BN 1/(1+r); LR/RF OOF thresholds) | No |",
      "| **Table 8 and Section 4.3.2** (1,023 / 593 / 387 at r = 10) | **BN 1/11 at BOTH stages** (`ps >= 1/(1+r)`, `pm >= 1/(1+r)`); the locked 0.17 is not used; the Application BN flags use 1/11 | No, fixed in advance |",
      "| Fig 5 (Application), Fig S3, Fig S4, Tables 4, 6, 9, S1, S2 (AUC, paired AUC differences, masking AUC loss) | None: AUC is threshold-free (the per-seed files also hold a `flip_share` at r = 10 / locked thresholds, but it is not in the article) | n/a (After acceptance uses the iteration-79 BN, chosen on test log-likelihood) |",
      "| S_exp, Table 2/5 sensitivities, Fig 2 and S5/S6 | None (S_exp is a probability shift) | n/a |", "",
      "### Evidence for the three key rows", "",
      "**Table 5 (flip rates at the locked 0.17) and what the other thresholds would give** (the stored per-row baseline and retracted probabilities, `vo_partA_retraction_per_row.pkl`, 29,998 rows per variable):", "",
      "| Variable | flip rate at 0.17 (Table 5) | at 1/6 (r = 5) | at 1/11 (r = 10) |", "|---|---|---|---|"] + \
     [f"| {v} | {100*r[0]:.2f}% | {100*r[1]:.2f}% | {100*r[2]:.2f}% |" for v, (r, n) in flip.items()] + \
     ["", "**Table 8: sensitivity to the After-acceptance BN threshold** (the article uses 1/11 at both stages; defaulters flagged at application / missed at application but flagged after acceptance / neither, of 2,003):", "",
      "| Application threshold / After-acceptance threshold | flagged at application | flagged only after acceptance | neither |", "|---|---|---|---|"] + \
     [f"| {a} | {s[0]:,} | {s[1]:,} | {s[2]:,} |" for a, _, s in sens_rows] + \
     ["", "**Table 7, After acceptance, r = 5: saving at the Fig 4 thresholds against the locked thresholds** (29,998 rows):", "", "| Model | Fig 4 / Table 7 threshold | saving there | locked threshold | saving there |", "|---|---|---|---|---|"] + \
     [f"| {m} | {T[('monitoring', m)][5]:.4g} | {lk[m][0]:.2f}% | {locked[m]} | {lk[m][1]:.2f}% |" for m in ("BN", "LR", "RF")] + \
     ["", checks_md(p), "",
      "### Points to note", "",
      "1. **Table 8 versus the text.** Section 4.3.2 says 'the after-acceptance threshold was chosen on the test set'. Table 8 does not use the locked 0.17: it uses 1/11 at both stages, fixed in advance. The only test-set choice behind Table 8 is the After-acceptance BN itself (EM iteration 79, chosen by test log-likelihood). The sentence is accurate for Table 5, not for Table 8; consider rewording (for example 'the after-acceptance BN was the iteration chosen on the test set').",
      "2. **Table 7 and Fig 4 are free of the locked thresholds**, which is the stronger reading of Section 3.6; only Table 5 / Section 4.2.2 (flip rates) and the statement in Section 5.1.2 depend on the locked BN threshold 0.17. The locked LR and RF thresholds (0.69, 0.67) do not appear in any article number or supplement table (they appear in the ledger as N-MON-004 to 006 and in the locked-model figures that are not used).",
      "3. **The flip rates at r = 5 and r = 10 would differ from Table 5** (table above), so the choice of 0.17 affects the published rates; at 1/6 (the a-priori threshold for r = 5) the rates are close to the 0.17 rates.", ""]
p.write("\n".join(md))
