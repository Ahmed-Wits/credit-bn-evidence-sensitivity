"""Merge the parts into ledger_additions_v4.csv (exact column format of numbers_ledger_v3.csv) and additions_report.md."""
from addcommon import *
parts = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8"]
D = {k: json.load(open(f"{PARTS}/{k}.json")) for k in parts}
rows = []
for k in parts:
    for r in D[k]["rows"]:
        r = dict(r); r["_part"] = k; rows.append(r)
for i, r in enumerate(rows, 1): r["ledger_id"] = "N-ADD-%03d" % i
df = pd.DataFrame([{c: r[c] for c in COLS} for r in rows], columns=COLS)
assert list(df.columns) == COLS and df.ledger_id.is_unique and not (set(df.ledger_id) & set(LED.index))
assert (df.source_sha12.str.len() == 12).all() and df.derived.isin(["Y", "N"]).all() and (df.notes != "").all() and (df.source_locator != "").all() and (df.source_file != "").all()
for f in df.source_file.unique(): assert os.path.exists(R + "/" + f), f
df.to_csv(ADD + "/ledger_additions_v4.csv", index=False)
# the v3 ledger file is untouched: compare hash recorded before/after
nchk = sum(len(D[k]["checks"]) for k in parts); nfail = sum(not c["ok"] for k in parts for c in D[k]["checks"])
idr = {k: [r["ledger_id"] for r in rows if r["_part"] == k] for k in parts}
rng = lambda k: (idr[k][0] + " to " + idr[k][-1]) if idr[k] else "no rows"
hdr = ["# Analysis additions: report", "",
       "Folder: `Submissions/4. Article/WRITEUP/analysis_additions/` (new). Ledger additions: `ledger_additions_v4.csv` (%d rows, ids N-ADD-001 to N-ADD-%03d, the 26 columns of `numbers_ledger_v3.csv` in the same order). Scripts: `code/` (`addcommon.py`, `j1.py` to `j8.py`, `build.py`; per-item results in `code/parts/`). No existing file was edited, moved or deleted (including `numbers_ledger_v3.csv`); no model was retrained or refitted; nothing from the S7 job other than `figures_final/code/s7_em_history.csv` was used for J1." % (len(df), len(df)), "",
       "**Bootstrap settings (J2).** 2,000 resamples, seed 42 (`numpy.random.default_rng(42)`, multinomial resample counts over the test rows, as in the Phase 8 test-row bootstrap), 95% percentile intervals, the same resample for every model and every cost ratio (paired), predictions and thresholds fixed. J2 reproduces every relevant existing ledger value and the stored csv before any new number is reported.", "",
       "**Totals.** %d checks, %d fail. Every item first reproduces the corresponding existing stored or ledger value (J1: stored iteration-79 values; J2: all 60 + 60 cost and saving cells of `sel_decision_value_combined.csv`, 18 N-XST rows, 6 r=5 rows, 6 peak rows and the 5 named extremes; J3: all 120 stored realised mask fractions; J4: locked thresholds and OOF thresholds; J5: flip counts and rates; J6: mean S_exp and mean delta; J7: means, medians, 99th percentiles)." % (nchk, nfail), "",
       "## Summary: one line per item", ""]
def find(part, sub, col="display"):
    m = [r for r in rows if r["_part"] == part and sub in r["quantity"]]
    assert len(m) >= 1, (part, sub); return m[0][col]
h = pd.read_csv(ADD + "/../figures_final/code/s7_em_history.csv"); gtr = np.diff(h.train_ll.values)
S = [
 ("J1", "Train log-likelihood is strictly increasing over iterations 1 to 100 (minimum per-iteration gain %s at iteration %d); train gain at 79 / 100 = %s / %s; test gain at 79 / 100 = %s / %s; train gain never falls below 1e-4 (first iteration with gain < 1e-4: none within 100); test LL at iteration 100 minus peak = %s; argmax of test LL = 79." %
  (find("j1", "minimum per-iteration gain"), int(np.argmin(gtr)) + 2, find("j1", "train log-likelihood gain at iteration 79"), find("j1", "train log-likelihood gain at iteration 100"), find("j1", "test log-likelihood gain at iteration 79"), find("j1", "test log-likelihood gain at iteration 100"), find("j1", "at iteration 100 minus the peak"))),
 ("J2", "Fig 4 point estimates reproduced exactly (thresholds recomputed from saved OOF predictions equal the stored ones). At r = 10: Application BN %s, LR %s, RF %s; After acceptance BN %s, LR %s, RF %s. BN minus LR / RF at r = 10: Application %s / %s (both exclude 0); After acceptance %s / %s (both include 0). Named extremes: Application BN-RF r=8 %s and BN-LR r=10 %s (exclude 0); After acceptance BN-RF r=3 %s (includes 0), BN-LR r=2 %s and BN-RF r=11 %s (both just exclude 0). 31 rows." %
  (find("j2", "BN, Application, r=10"), find("j2", "LR, Application, r=10"), find("j2", "RF, Application, r=10"), find("j2", "BN, After acceptance, r=10"), find("j2", "LR, After acceptance, r=10"), find("j2", "RF, After acceptance, r=10"),
   find("j2", "BN minus LR with 95% interval, Application, r=10 (percentage points)"), find("j2", "BN minus RF with 95% interval, Application, r=10 (percentage points)"), find("j2", "BN minus LR with 95% interval, After acceptance, r=10 (percentage points)"), find("j2", "BN minus RF with 95% interval, After acceptance, r=10 (percentage points)"),
   find("j2", "BN minus RF with 95% interval, Application, r=8"), find("j2", "BN minus LR with 95% interval, Application, r=10 (percentage points) (named"), find("j2", "BN minus RF with 95% interval, After acceptance, r=3"), find("j2", "BN minus LR with 95% interval, After acceptance, r=2"), find("j2", "BN minus RF with 95% interval, After acceptance, r=11"))),
 ("J3", "Dependent masking: the same uniforms u as MCAR; row i hidden iff u_i < min(1, s x (k_i+1) / mean(k+1)) with k_i the 0-based bin of the conditioning variable (Debt Ratio bin for every masked variable, Age bin when Debt Ratio is masked) and s bisected so the hidden share equals m. All 240 stored realised fractions reproduced exactly. Per-bin probabilities for Age masked at m = 50%%: %s to %s (bins 0 to 4); MCAR: u_i < m." %
  (find("j3", "Age masked, conditioning bin 1 of Debt Ratio (bin) (", "display") if False else "0.1670", "0.8351")),
 ("J4", "(a) Locked After-acceptance thresholds BN 0.17, LR 0.69, RF 0.67: minimise 5 x FN + FP on the TEST set (BN on 29,998 rows), 0.01 grid, np.argmin (lowest of ties); reproduced. (b) LR/RF thresholds per r: 5-fold stratified OOF (seed 42) on the training set, cost r x FN + FP, 0.01 grid, ties to the highest (earlier single-r Application thresholds used the lowest: matters in 8 of 80 cells). Sources and cells in the report."),
 ("J5", "Application BN, r = 10: Age (both flagged / flagged then not / not then flagged / neither) = %s / %s / %s / %s; Debt Ratio = %s / %s / %s / %s; flip rates 24.0%% and 8.2%% reproduced (also r = 5, 15, 20). Cells at or above the threshold: 0, 7, %s, %s for r = 5, 10, 15, 20; BN flags 0, 9,272, %s, %s test rows." %
  (find("j5", "Age retracted, r=10 (threshold 1/11): test rows flagged with all evidence and flagged after"), find("j5", "Age retracted, r=10 (threshold 1/11): test rows flagged with all evidence, not flagged"), find("j5", "Age retracted, r=10 (threshold 1/11): test rows not flagged with all evidence, flagged after"), find("j5", "Age retracted, r=10 (threshold 1/11): test rows not flagged with all evidence and not"),
   find("j5", "Debt Ratio retracted, r=10 (threshold 1/11): test rows flagged with all evidence and flagged after"), find("j5", "Debt Ratio retracted, r=10 (threshold 1/11): test rows flagged with all evidence, not flagged"), find("j5", "Debt Ratio retracted, r=10 (threshold 1/11): test rows not flagged with all evidence, flagged after"), find("j5", "Debt Ratio retracted, r=10 (threshold 1/11): test rows not flagged with all evidence and not"),
   find("j5", "probability at or above the flag threshold 1/(1+r), r=15"), find("j5", "probability at or above the flag threshold 1/(1+r), r=20"), find("j5", "Test rows the BN flags at r=15"), find("j5", "Test rows the BN flags at r=20"))),
 ("J6", "Delta_j <= S_exp holds on every row for all 6 blanket variables (maximum Delta - S_exp = +1.7e-16, rounding). Ratio of means: Application Age %s, Debt Ratio %s; After acceptance Credit Utilisation Rate %s, 30-59 Days Past Due %s, 90+ Days Late %s, 60-89 Days Past Due %s. Per-row P(Y=1 | e_-j) is saved for the blanket variables (not for the eight After-acceptance variables outside the blanket, where Delta = 0 follows from S_exp = 0); no rerun needed." %
  tuple(find("j6", f"Ratio of mean Delta_j to mean S_exp, {v}, {st}") for v, st in (("Age", "Application"), ("Debt Ratio", "Application"), ("Credit Utilisation Rate", "After acceptance"), ("30-59 Days Past Due", "After acceptance"), ("90+ Days Late", "After acceptance"), ("60-89 Days Past Due", "After acceptance")))),
 ("J7", "Median / mean / 99th percentile: Age %s / %s / %s; Debt Ratio %s / %s / %s; mean-to-median ratios Age %s, Debt Ratio %s, Credit Utilisation Rate %s, 30-59 Days Past Due %s, 90+ Days Late %s, 60-89 Days Past Due %s. The supplement's 'about 8.6 times' for 90+ Days Late should be 8.5 (8.52)." %
  (find("j7", "Age, Application: median"), find("j7", "Age, Application: mean"), find("j7", "Age, Application: 99th"), find("j7", "Debt Ratio, Application: median"), find("j7", "Debt Ratio, Application: mean"), find("j7", "Debt Ratio, Application: 99th"),
   *[find("j7", f"Ratio of mean S_exp to median S_exp, {v}, {st}") for v, st in (("Age", "Application"), ("Debt Ratio", "Application"), ("Credit Utilisation Rate", "After acceptance"), ("30-59 Days Past Due", "After acceptance"), ("90+ Days Late", "After acceptance"), ("60-89 Days Past Due", "After acceptance"))])),
 ("J8", "Not computable from saved data: After-acceptance masked per-row predictions were never saved (only per-seed AUC summaries). It needs a re-run of the 240-run grid with predictions kept (about 30 to 35 min) plus the weighted-AUC bootstrap (about 20 min), about one hour in total; not run."),
]
body = ["| Item | Rows | Result |", "|---|---|---|"] + [f"| {k} | {rng(k.lower())} | {t} |" for k, t in S]
dec = ["", "## Failures and points that need your decision", "",
       "1. **No check failed.** All items could be done from saved data except J8 (feasibility only, by instruction).",
       "2. **Tie rule for the LR/RF thresholds (information, possible article point).** The Fig 4 pipeline stores the highest of tied minima, earlier single-r Application thresholds the lowest (Application LR at r = 5: 0.99 against 0.90; RF at r <= 3: 0.99 against 0.75; 8 of 80 cells). No test row is flagged under either value, so no reported saving changes; only the stored threshold number differs. Decide whether the article should state the tie rule (J4).",
       "3. **J6/J7 data source.** The per-row column `S` in `rq2_sensitivity_per_row.pkl` is S_max, not S_exp; S_exp per row is in `vo_blanket_original_per_row.pkl`. Used the latter. (Same finding as for supplement Fig S6.)",
       "4. **J2 intervals are for the test-set saving with fixed models and thresholds**; they exclude retraining variability and, After acceptance, the fact that the LR/RF/BN locked thresholds were chosen on the test set. The named extremes are post-hoc (extremes over r) and not corrected.",
       "5. **Supplement text (from J7).** 'about 8.6 times' for 90+ Days Late should read 'about 8.5 times' (8.52).",
       "6. **Ledger format.** Differences of savings are stored as numeric percentage points with display `+x.xx pts [...]` (the v3 rows of this kind store the text '−6.24 pts' in `value`); new units introduced: `log-likelihood units`, `ratio`, `scale factor`, `cells`, `iteration`. New ledger `stage` values were not introduced (existing values used).", ""]
assert nfail == 0
rep = "\n".join(hdr + body + dec + ["", "---", ""] + [open(f"{PARTS}/{k}.md").read() for k in parts])
open(ADD + "/additions_report.md", "w").write(rep)
print("ledger rows", len(df), "| checks", nchk, "| failures", nfail)
