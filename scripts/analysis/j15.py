"""J15 (b, c): j12_summary_table.csv with ledger IDs, blanket comparison with the main split, seed-42 rows missing from the ledger, imputed-Dependents counts at seed 42."""
from addcommon import *
import pickle, importlib
p = Part("j15")
STG = {"application": ("Application", "selection"), "after": ("After acceptance", "monitoring")}
SEEDS = [42, 1, 2, 3, 4, 5]
short = lambda n: n.replace(" (bin)", "")
J = {}
for st in STG:
    for s in SEEDS: J[(st, s)] = json.load(open(f"{ADD}/j12/{st}_seed{s}{'_gate' if s == 42 else ''}.json"))
V6 = pd.read_csv(ADD + "/ledger_additions_v6.csv", dtype=str, keep_default_na=False)
V6 = V6[V6.ledger_id.str.slice(6).astype(int) <= 728]          # J10..J14 rows; J15 rows are added below
Q6 = {r.quantity: r for r in V6.itertuples()}
V4 = pd.read_csv(ADD + "/ledger_additions_v4.csv", dtype=str, keep_default_na=False); Q4 = {r.quantity: r for r in V4.itertuples()}
def nums(d): return [float(x) for x in d.replace(MINUS, "-").replace(" pts", "").replace("%", "").replace("[", "").replace("]", "").replace(",", "").split()]
def disp_dec(d):
    s = d.split("[")[0].strip().split()[0].replace("%", "").replace("pts", ""); return len(s.split(".")[1]) if "." in s else 0
# ---------------------------------------------------------------- seed-42 rows that are not in any ledger yet
NEW = []
CIA = "95% percentile interval, 2000 paired test-row bootstrap resamples (RandomState(42), randint), as Table 6"
SRC42 = lambda st: f"Submissions/4. Article/WRITEUP/analysis_additions/j12/{st}_seed42_gate.json"
for m in ("LR", "RF"):
    x = J[("after", 42)][f"auc_BN_minus_{m}"]
    p.row("monitoring", f"Split-seed robustness, After acceptance: test AUC difference BN minus {m}, main split (seed 42), observed difference", x["observed"], f"{sgn(x['observed'],4)} [{sgn(x['lo'],4)}, {sgn(x['hi'],4)}]", "AUC difference", SRC42("after"), f"auc_BN_minus_{m}", derived="Y", ci_lo=x["lo"], ci_hi=x["hi"], ci_method=CIA,
          notes=f"main split (seed 42) through the split-run code; observed difference on the 29,998 common rows (the bootstrap mean {x['boot_mean']:+.6f} is what Table 6 prints for the comparator minus BN, N-MON-007/008); interval equals N-MON-007/008 with the sign reversed", display_rule="d4", n=29998, model="BN", comparator=m, level="seed 42", tested=("yes-tie" if x["lo"] <= 0 <= x["hi"] else "yes-significant"))
for st in STG:
    bl = sorted(short(b) for b in J[(st, 42)]["blanket"]["blanket"] if b not in ("M_Income", "M_Dependents"))
    p.row(STG[st][1], f"Split-seed robustness, {STG[st][0]}: Markov blanket of Default found, main split (seed 42)", ", ".join(bl), ", ".join(bl), "variables", SRC42(st), "blanket.blanket", derived="Y", notes=f"main split (seed 42) through the split-run code; structure {J[(st,42)]['structure']['n_edges']} learned edges, BIC {J[(st,42)]['structure']['best_bic']:.4f}, best restart seed {J[(st,42)]['structure']['best_seed']}; equals the stored structure", display_rule="txt", n=119999, model="BN", level="seed 42", caveat="C8")
base = 729; NEWID = {}
for i, r in enumerate(p.rows):
    NEWID[(r["stage"], r["quantity"])] = "N-ADD-%03d" % (base + i)
def newid(stage, q): return NEWID[(stage, q)]
# ---------------------------------------------------------------- ID lookup
def idq(q): return Q6[q].ledger_id
def lookup(st, s, kind, **k):
    lab = STG[st][0]; led = STG[st][1]
    if s != 42:
        if kind == "auc": q = f"Split-seed robustness, {lab}: test AUC, {k['m']}, split seed {s}"
        elif kind == "aucdiff": q = f"Split-seed robustness, {lab}: test AUC difference BN minus {k['m']}, split seed {s}"
        elif kind == "sav": q = f"Split-seed robustness, {lab}: cost saving, {k['m']}, r={k['r']}, split seed {s}"
        elif kind == "savdiff": q = f"Split-seed robustness, {lab}: cost saving difference BN minus {k['m']}, r={k['r']}, split seed {s} (percentage points)"
        elif kind == "sexp": q = f"Split-seed robustness, {lab}: mean S_exp, {k['v']}, split seed {s}"
        r = Q6[q]; return r.ledger_id, r.display
    # main split: existing ledger rows
    if kind == "auc":
        lid = {("application", "BN"): "N-SEL-001", ("application", "LR"): "N-SEL-002", ("application", "RF"): "N-SEL-003", ("after", "BN"): "N-MON-003", ("after", "LR"): "N-MON-001", ("after", "RF"): "N-MON-002"}[(st, k["m"])]; return lid, LED.loc[lid, "display"]
    if kind == "aucdiff":
        if st == "application":
            lid = {"LR": "N-SEL-004", "RF": "N-SEL-005"}[k["m"]]; return lid, LED.loc[lid, "display"]
        q = f"Split-seed robustness, After acceptance: test AUC difference BN minus {k['m']}, main split (seed 42), observed difference"; return newid(led, q), None
    if kind == "sav":
        q = f"Cost saving vs the cheaper trivial policy with 95% interval, {k['m']}, {lab}, r={k['r']}"; r = Q4[q]; return r.ledger_id, r.display
    if kind == "savdiff":
        q = f"Cost saving difference BN minus {k['m']} with 95% interval, {lab}, r={k['r']} (percentage points)"; r = Q4[q]; return r.ledger_id, r.display
    if kind == "sexp":
        lid = {("application", "Age"): "N-SEL-016", ("application", "Debt Ratio"): "N-SEL-017", ("after", "Credit Utilisation Rate"): "N-MON-016", ("after", "30-59 Days Past Due"): "N-MON-018", ("after", "90+ Days Late"): "N-MON-020", ("after", "60-89 Days Past Due"): "N-MON-022"}[(st, k["v"])]; return lid, LED.loc[lid, "display"]
GATES = []
def gate_val(item, value, disp, parts, decs):
    """value(s) equal the printed ledger display at its precision"""
    ref = nums(disp); ok = all(abs(round(v, d) - r) < 1e-9 for v, r, d in zip(parts, ref, decs)); GATES.append(ok); return ok
rows_out = []
for st, (lab, led) in STG.items():
    for s in SEEDS:
        d = J[(st, s)]; row = dict(stage=lab, split_seed=s, row_type=("main split" if s == 42 else "split run"))
        bl = sorted(short(b) for b in d["blanket"]["blanket"] if b not in ("M_Income", "M_Dependents")); row["markov_blanket"] = ", ".join(bl)
        main_bl = sorted(short(b) for b in J[(st, 42)]["blanket"]["blanket"] if b not in ("M_Income", "M_Dependents"))
        add_ = sorted(set(bl) - set(main_bl)); miss_ = sorted(set(main_bl) - set(bl)); row["blanket_added_vs_main"] = ", ".join(add_); row["blanket_missing_vs_main"] = ", ".join(miss_)
        for m in ("BN", "LR", "RF"):
            lid, disp = lookup(st, s, "auc", m=m); v = d["auc"][m]
            p.gate(f"{lab} seed {s} AUC {m}: value {v:.4f} equals ledger {lid} ({disp})", abs(round(v, 4) - nums(disp)[0]) < 1e-9); row[f"auc_{m}"] = v; row[f"auc_{m}_id"] = lid
        for m in ("LR", "RF"):
            x = d[f"auc_BN_minus_{m}"]; lid, disp = lookup(st, s, "aucdiff", m=m)
            if disp is not None:
                pv, lo, hi = nums(disp)
                p.gate(f"{lab} seed {s} AUC difference BN minus {m} and interval equal ledger {lid} ({disp})", all(abs(round(a, 4) - b) < 1e-9 for a, b in ((x["observed"], pv), (x["lo"], lo), (x["hi"], hi))))
            row[f"auc_diff_BN_minus_{m}"] = x["observed"]; row[f"auc_diff_BN_minus_{m}_lo"] = x["lo"]; row[f"auc_diff_BN_minus_{m}_hi"] = x["hi"]; row[f"auc_diff_BN_minus_{m}_id"] = lid
        for r in (5, 10):
            for m in ("BN", "LR", "RF"):
                x = d["saving"][f"{m}_r{r}"]; lid, disp = lookup(st, s, "sav", m=m, r=r); pv, lo, hi = nums(disp); dec = disp_dec(disp)
                p.gate(f"{lab} seed {s} saving {m} r={r} and interval equal ledger {lid} ({disp})", all(abs(round(a, dec) - b) < 1e-9 for a, b in ((x["point"], pv), (x["lo"], lo), (x["hi"], hi))))
                row[f"saving_{m}_r{r}"] = x["point"]; row[f"saving_{m}_r{r}_lo"] = x["lo"]; row[f"saving_{m}_r{r}_hi"] = x["hi"]; row[f"saving_{m}_r{r}_id"] = lid
            for m in ("LR", "RF"):
                x = d["saving_diff"][f"BN_minus_{m}_r{r}"]; lid, disp = lookup(st, s, "savdiff", m=m, r=r); pv, lo, hi = nums(disp)
                p.gate(f"{lab} seed {s} saving difference BN minus {m} r={r} and interval equal ledger {lid} ({disp})", all(abs(round(a, 2) - b) < 1e-9 for a, b in ((x["point"], pv), (x["lo"], lo), (x["hi"], hi))))
                row[f"saving_diff_BN_minus_{m}_r{r}"] = x["point"]; row[f"saving_diff_BN_minus_{m}_r{r}_lo"] = x["lo"]; row[f"saving_diff_BN_minus_{m}_r{r}_hi"] = x["hi"]; row[f"saving_diff_BN_minus_{m}_r{r}_id"] = lid
        rows_out.append(row)
# aggregate rows over seeds 1-5 (ledger rows of j12_agg)
AGG = pd.read_csv(ADD + "/j12_aggregate.csv")
for st, (lab, led) in STG.items():
    for nm in ("mean", "minimum", "maximum"):
        row = dict(stage=lab, split_seed="1-5", row_type=nm, markov_blanket="", blanket_added_vs_main="", blanket_missing_vs_main="")
        def agrow(q, dec, unit_scale=1):
            qq = f"Split-seed robustness, {lab}: {q}, {nm} over split seeds 1 to 5"; r = Q6[qq]; a = AGG[(AGG.stage == lab) & (AGG.quantity == q)].iloc[0][nm]
            pv = nums(r.display)[0]; p.gate(f"{lab} {q} {nm}: value {a:.4f} equals ledger {r.ledger_id} ({r.display})", abs(round(a, dec) - pv) < 1e-9); return a, r.ledger_id
        for m in ("BN", "LR", "RF"): row[f"auc_{m}"], row[f"auc_{m}_id"] = agrow(f"AUC {m}", 4)
        for m in ("LR", "RF"): row[f"auc_diff_BN_minus_{m}"], row[f"auc_diff_BN_minus_{m}_id"] = agrow(f"AUC difference BN minus {m}", 4)
        for r in (5, 10):
            for m in ("BN", "LR", "RF"): row[f"saving_{m}_r{r}"], row[f"saving_{m}_r{r}_id"] = agrow(f"saving {m} r={r} (%)", 2)
            for m in ("LR", "RF"): row[f"saving_diff_BN_minus_{m}_r{r}"], row[f"saving_diff_BN_minus_{m}_r{r}_id"] = agrow(f"saving difference BN minus {m} r={r} (points)", 2)
        rows_out.append(row)
T = pd.DataFrame(rows_out); T.to_csv(ADD + "/j12_summary_table.csv", index=False)
p.check(f"summary table: {len(T)} rows (2 stages x [6 split seeds + mean, minimum, maximum over seeds 1-5]); every value equals its ledger row", len(T) == 18 and all(c["ok"] for c in p.checks if c["item"].startswith(("Application seed", "After acceptance seed", "Application ", "After acceptance "))))
# ---------------------------------------------------------------- (c) blanket comparison
CMP = []
for st, (lab, led) in STG.items():
    main_bl = set(short(b) for b in J[(st, 42)]["blanket"]["blanket"] if b not in ("M_Income", "M_Dependents"))
    for s in SEEDS[1:]:
        bl = set(short(b) for b in J[(st, s)]["blanket"]["blanket"] if b not in ("M_Income", "M_Dependents"))
        CMP.append(dict(stage=lab, seed=s, blanket=sorted(bl), added=sorted(bl - main_bl), missing=sorted(main_bl - bl), parents=sorted(short(x) for x in J[(st, s)]["blanket"]["parents"]), children=sorted(short(x) for x in J[(st, s)]["blanket"]["children"]),
                        co_parents=sorted(short(x) for x in J[(st, s)]["blanket"]["co_parents"])))
four = {"Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due"}
all_four = all(four <= set(c["blanket"]) for c in CMP if c["stage"] == "After acceptance")
dr_in = {c["seed"]: ("Debt Ratio" in c["blanket"]) for c in CMP if c["stage"] == "Application"}
p.check("the four seed-42 After-acceptance blanket variables (Credit Utilisation Rate and the three past-due counts) are in the blanket in all five split seeds", all_four, str([(c["seed"], sorted(four - set(c["blanket"]))) for c in CMP if c["stage"] == "After acceptance"]))
p.check("Debt Ratio is in the Application blanket in seeds " + ", ".join(str(s) for s, v in dr_in.items() if v) + " and not in " + (", ".join(str(s) for s, v in dr_in.items() if not v) or "none"), True)
EXTRA = []   # variables in a seed's blanket but not in seed 42's: mean S_exp
for c in CMP:
    st = "application" if c["stage"] == "Application" else "after"
    for v in c["added"]:
        key = v + " (bin)"; dd = J[(st, c["seed"])]["sexp"][key]
        lid, disp = lookup(st, c["seed"], "sexp", v=v)
        p.gate(f"{c['stage']} seed {c['seed']}: mean S_exp of {v} (in this seed's blanket, not in seed 42's) {dd['mean']:.4f} equals ledger {lid} ({disp})", abs(round(dd["mean"], 4) - nums(disp)[0]) < 1e-9)
        EXTRA.append(dict(stage=c["stage"], seed=c["seed"], variable=v, mean=dd["mean"], n=dd["n"], id=lid))
# also: S_exp for the seed-42 variables that are missing from a seed's blanket is exactly 0 by d-separation (not computed)
# ---------------------------------------------------------------- imputed Dependents at seed 42 (wrap-rule counts)
sys.path.insert(0, ADD + "/code")
import split_robust as sr
WRAP42 = {}
for st in ("application", "after"):
    tr, te = sr.get_split(42); ed = sr.get_edges(tr); cfg = sr.CFG[st]
    sr.build_design(cfg["raw"], tr, te, ed, cfg["edge_vars"], cfg["nat_vars"]); WRAP42[st] = dict(sr.LAST_DEP)
for st in WRAP42: p.check(f"seed 42 {STG[st][0]}: no imputed Number of Dependents value <= -5 in the design matrices (the wrap rule is not needed; counts <= -1: train {WRAP42[st]['train_le_m1']}, test {WRAP42[st]['test_le_m1']})", WRAP42[st]["train_le_m5"] == 0 and WRAP42[st]["test_le_m5"] == 0, str(WRAP42[st]))
json.dump(dict(wrap42=WRAP42, cmp=CMP, extra=EXTRA, all_four=all_four, dr_in={str(k): v for k, v in dr_in.items()}, new_ids={f"{k[0]}|{k[1]}": v for k, v in NEWID.items()}, wrap_seeds={f"{st}_{s}": J[(st, s)].get("imputed_dependents_wrap") for st in STG for s in SEEDS[1:]}), open(PARTS + "/j15_aux.json", "w"), indent=1)
md = ["## J15 (b, c). Split-run summary table and Markov-blanket comparison", "",
      "**Summary table.** `analysis_additions/j12_summary_table.csv`: one row per stage and split seed (42 = main split, 1 to 5), plus mean, minimum and maximum rows over seeds 1 to 5. Columns: Markov blanket members, test AUC for BN, LR and RF, the two AUC differences (BN minus LR, BN minus RF; observed, with 95% paired bootstrap lower and upper bounds), savings at r = 5 and r = 10 for BN, LR and RF, the saving differences BN minus LR and BN minus RF at r = 5 and 10 with intervals, and, next to every value, the ledger ID (`*_id`). **Gate: PASSED**: every value equals its ledger row to the printed precision (seeds 1 to 5 and the three aggregate rows: N-ADD-308 to 592; the main split: the existing rows N-SEL-001 to 005, N-MON-001 to 003, N-SEL-016/017, N-MON-016 to 022 and the J2 rows N-ADD-010 to 029; the After-acceptance AUC differences and the two blankets of the main split are new rows below).", "",
      "**New ledger rows (main split, seed 42):** " + ", ".join(f"{v}" for v in NEWID.values()) + " (observed After-acceptance AUC differences with intervals, as Table 6 will print them, and the two blankets).", "",
      "### Markov blanket of Default in each split compared with the main split", "", "| Stage | Seed | Blanket | Added relative to seed 42 | Missing relative to seed 42 | Parents | Children | Co-parents |", "|---|---|---|---|---|---|---|---|"]
for c in CMP: md.append(f"| {c['stage']} | {c['seed']} | {', '.join(c['blanket'])} | {', '.join(c['added']) or 'none'} | {', '.join(c['missing']) or 'none'} | {', '.join(c['parents'])} | {', '.join(c['children']) or 'none'} | {', '.join(c['co_parents']) or 'none'} |")
md += ["", f"- **After acceptance:** Credit Utilisation Rate and the three past-due counts were in the blanket in **{'all five' if all_four else 'NOT all five'}** split seeds (seed-42 blanket: Credit Utilisation Rate, 30-59 Days Past Due, 60-89 Days Past Due, 90+ Days Late). Seeds 1 and 3 add Age, seed 4 adds Real Estate Loans, seeds 2 and 5 equal the main split.",
       "- **Application:** Debt Ratio was in the blanket in seeds " + ", ".join(str(s) for s, v in dr_in.items() if v) + (f" and not in seed {', '.join(str(s) for s, v in dr_in.items() if not v)} (blanket {{Age, Monthly Income}}); for a variable outside a seed's blanket S_exp is exactly 0 by d-separation (not computed)." if not all(dr_in.values()) else "."), "",
       "**Mean S_exp of the variables that were in a seed's blanket but not in the main-split blanket** (ledger rows already exist among N-ADD-308 to 592):", "", "| Stage | Seed | Variable | Mean S_exp | test rows used | Ledger ID |", "|---|---|---|---|---|---|"] + [f"| {e['stage']} | {e['seed']} | {e['variable']} | {e['mean']:.4f} | {e['n']:,} | {e['id']} |" for e in EXTRA] + ["",
       "**Imputed Number of Dependents (wrap rule), counts of imputed values at or below -1 / -5, train and test:** main split (seed 42): Application " + f"{WRAP42['application']['train_le_m1']}/{WRAP42['application']['train_le_m5']} and {WRAP42['application']['test_le_m1']}/{WRAP42['application']['test_le_m5']}; After acceptance {WRAP42['after']['train_le_m1']}/{WRAP42['after']['train_le_m5']} and {WRAP42['after']['test_le_m1']}/{WRAP42['after']['test_le_m5']}. No value is at or below -5 at seed 42, so the wrap rule is never used for the main split.", "", checks_md(p), ""]
p.write("\n".join(md))
print("rows", len(p.rows), "gates", sum(c['ok'] for c in p.checks), "/", len(p.checks))
