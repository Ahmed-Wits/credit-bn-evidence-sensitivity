"""Tier A driver: draws Fig 1a, 1b, 2, 3, 4, 5, audits them, writes caption_notes.md, verify_report.md, contact_sheet.png."""
import sys, os, shutil, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import fig1, fig2, fig3, fig4, fig5
import figsB
import fig4x, fig6, figS8, figS9
from PIL import Image, ImageDraw, ImageFont

H1 = 2.9
POS_A = {"M_Income": (1.0, 2.6), "Monthly Income": (2.6, 2.6), "Debt Ratio": (1.2, 1.5), "Real Estate Loans": (4.05, 1.8),
         "Open Credit Lines": (3.6, 0.95), "Age": (2.0, 0.4), "Default": (0.35, 0.4), "Number of Dependents": (5.3, 0.4), "M_Dependents": (6.3, 1.5)}
POS_B = {"Credit Utilisation Rate": (3.0, 2.55), "Open Credit Lines": (1.9, 2.4), "Default": (5.5, 2.5), "90+ Days Late": (4.1, 2.3),
         "60-89 Days Past Due": (4.5, 1.45), "30-59 Days Past Due": (5.5, 1.2), "Debt Ratio": (0.7, 1.45), "Monthly Income": (2.3, 1.45),
         "M_Income": (3.3, 1.0), "Real Estate Loans": (1.5, 0.85), "Age": (0.7, 0.3), "Number of Dependents": (2.7, 0.3), "M_Dependents": (3.8, 0.7)}
LEG_A, LEG_B = (4.3, 2.85), (4.0, 0.92)
LEGRECT_A, LEGRECT_B = (4.3, 1.95, W2, 2.9), (4.0, 0.0, W2, 0.92)

CURVES_A = {("Real Estate Loans", "Age"): 0.15}
CURVES_B = {("90+ Days Late", "Open Credit Lines"): -0.2}

def fig1_pair():
    out = {}
    for tag, name, E, pos, lg, lr, nodes_expected, edges_expected, ledger_note in (
            ("a", "fig1a_structure_application", fig1.load_app(), POS_A, LEG_A, LEGRECT_A, 9, 15, None),
            ("b", "fig1b_structure_after_acceptance", fig1.load_after(), POS_B, LEG_B, LEGRECT_B, 13, 24, None)):
        sides, ai = fig1.auto_sides(E, pos, H1, lr)
        if tag == "b": sides["90+ Days Late"] = ("r", 0, -0.05)
        f, info = fig1.draw(name, E, pos, sides, H1, lg, curves=(CURVES_A if tag == 'a' else CURVES_B))
        check(name, "number of nodes drawn", info["nodes"], nodes_expected, 0, "edge list")
        check(name, "number of edges drawn (all edges of the stored list)", info["edges"], edges_expected, 0, "edge list")
        if tag == "a":
            check(name, "best-structure edges (learned, excl. 2 indicator edges) vs ledger N-SEL-024", info["edges"] - 2, lv("N-SEL-024"), 0, "ledger N-SEL-024")
            exp = {"Age", "Debt Ratio"}; check_text(name, "derived Markov blanket of Default", ", ".join(info["blanket"]), ", ".join(sorted(exp)), "derived from sel_dag_9node_edges.json")
        else:
            check_text(name, "derived parents of Default", ", ".join(info["parents"]), ", ".join(sorted(["Credit Utilisation Rate", "90+ Days Late"])), "derived from bn_final_model_iteration79.pkl")
            check_text(name, "derived children of Default", ", ".join(info["children"]), ", ".join(sorted(["30-59 Days Past Due", "60-89 Days Past Due"])), "derived from bn_final_model_iteration79.pkl")
            check_text(name, "derived blanket (parents, children, co-parents)", ", ".join(info["blanket"]), ", ".join(sorted(["Credit Utilisation Rate", "90+ Days Late", "30-59 Days Past Due", "60-89 Days Past Due"])), "derived")
        assert not info["edge_through_node"], info["edge_through_node"]
        out[name] = (f, info, ai)
    return out

def main():
    figs = {}
    f1 = fig1_pair()
    for name, (f, info, ai) in f1.items():
        a = audit(f, name); a["structure"] = info; a["edge_label_hits"] = ai["edge_label_hits"]; figs[name] = f
    NOTES["fig1a_structure_application"] = [
        "Removed in-figure title of the old sel_dag_9node_v3.png: \"9-Node Selection Bayesian Network Structure (application-stage variables) (Yellow = target (Default), Blue = substantive features, Orange outline = Default's Markov blanket, Gray squares = missingness indicators (root nodes))\". The key is now a legend inside the figure.",
        "Layout is fixed (coordinates in code/run_tierA.py), node names are beside the nodes. Orange outline = Default's Markov blanket {Age, Debt Ratio}, derived programmatically from the stored edge list; dashed arrows are the fixed edges from the missingness indicators M_Income and M_Dependents.",
        "Edge list compared with the structure shown in sel_dag_9node_v3.png: all 15 edges identical (13 learned edges plus the 2 indicator edges)."]
    NOTES["fig1b_structure_after_acceptance"] = [
        "Removed in-figure title of the old bn_13node_dag_final.png: \"13-Node MNAR-Aware Bayesian Network Structure (Yellow = target (Default), Blue = substantive features, Gray squares = missingness indicators (root nodes), ...\" (the original title was cut off at the end of the second line). The key is now a legend inside the figure.",
        "Layout is fixed (coordinates in code/run_tierA.py); node names are beside the nodes. Orange outline = Default's Markov blanket, derived from the stored edge list: parents Credit Utilisation Rate and 90+ Days Late, children 30-59 Days Past Due and 60-89 Days Past Due (the other parent of the children is already in the set).",
        "Edge list compared with the structure shown in bn_13node_dag_final.png: all 24 edges identical (22 learned plus the 2 indicator edges)."]
    EDITS.insert(0, "Fig 1 panel order: panel (a) is now Application (9 nodes) and panel (b) After acceptance (13 nodes), so the (a)/(b) sub-captions and the file names in the LaTeX must be swapped (old: (a) After acceptance, (b) Application). Files: fig1a_structure_application, fig1b_structure_after_acceptance.")
    EDITS.insert(1, "Fig 1 caption: both panels now mark Default's Markov blanket (orange outline); the old caption said only panel (b) did. Add that dashed arrows come from the missingness indicators M_Income and M_Dependents (squares). Fonts and markers are now final size, node names are beside nodes.")
    EDITS.insert(2, "Fig 2 merge: Fig 2 is now ONE single-column image with panels (a) Application (top) and (b) After acceptance (bottom) and panel letters inside the image. The two sub-figure files/sub-captions of the old Fig 2 must be replaced by this one file (fig2_sexp_application_and_after_acceptance). The caption should say that only variables inside Default's Markov blanket are drawn and that the 4 (Application) and 8 (After acceptance, including Age, Debt Ratio and the two missingness indicators) other variables have S_exp = 0 on every row; whiskers are bootstrap 95% intervals (very narrow).")
    NOTES["fig1a_structure_application"].append("Wording used for the missingness indicators: M_Income, M_Dependents (full names; no short forms were needed).")
    for mod in (fig2, fig3, fig4, fig5):
        f = mod.make(); nm = mod.NAME; audit(f, nm); figs[nm] = f
    order = ["fig1a_structure_application", "fig1b_structure_after_acceptance", "fig2_sexp_application_and_after_acceptance", "fig3_application_lookup_heatmap", "fig4_cost_saving", "fig5_masking_paired_differences"]
    for mk in figsB.MAKERS:
        f, nm = mk(); audit(f, nm); figs[nm] = f; order.append(nm)
    for mod in (fig4x, fig6, figS8, figS9):
        f = mod.make(); audit(f, mod.NAME); figs[mod.NAME] = f; order.append(mod.NAME)
    import figS7
    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s7_em_history.csv")
    f7, h7 = figS7.make(csv)
    global S7_ACCEPTED
    S7_ACCEPTED = f7 is not None
    if S7_ACCEPTED:
        audit(f7, figS7.NAME); figs[figS7.NAME] = f7; order.append(figS7.NAME)
    else:
        shutil.copyfile(figS7.ORIG, f"{OUT}/{figS7.NAME}.png")
    from figsB import STATS as SB
    s6 = {v: SB["S6_" + v] for v in ["Credit Utilisation Rate", "90+ Days Late"]}
    EDITS.extend([
        "Supplement Fig S1 caption: add the AUCs in the legend (BN 0.6627, LR 0.6904, RF 0.6926) if wanted; no other change needed. S1 is single column (83.7 mm).",
        "Supplement Fig S2 caption: say that BN marker area is proportional to the square root of the number of test rows at each distinct predicted value, that the two panels have different axis limits (BN 0 to 0.15, LR and RF 0 to 1), and that the LR and RF probabilities are inflated by the balanced class weights (they lie far below the diagonal). The old panel titles and the footer text are now only in caption_notes.md.",
        "Supplement Fig S3 and S4 captions: say that solid lines are MCAR and dashed lines dependent masking, that the bands are SD across the 10 mask seeds (mask randomness only), and that the change is relative to no masking. S3 and S4 are drawn without the 0% point.",
        "Supplement Fig S5 caption and text (Section S4.2): the figure now shows only Age and Debt Ratio (not 'the six application variables'). Replace 'The four variables outside Default's Markov blanket had S_exp = 0 on every row' by an explicit list (Monthly Income, Number of Dependents, Real Estate Loans, Open Credit Lines: S_exp = 0 on every row). Describe the encoding: box 25th to 75th percentile, whiskers 5th to 95th, line = median, diamond = mean, cross = 99th percentile, log axis. Current caption says only 'with the mean and the 99th percentile marked'.",
        "Supplement Fig S6 caption and text (Section S4.2): the tick labels no longer say row-level or pattern-level. The text already defines both terms; add which variables are which: Credit Utilisation Rate and 90+ Days Late are row-level, 30-59 Days Past Due and 60-89 Days Past Due are pattern-level. Describe the box/whisker encoding as for S5.",
        "Supplement Section S4.2 numbers: the text says the means exceed the medians 'by about 1.4 times for Credit Utilisation Rate and about 8.6 times for 90+ Days Late'. The stored statistics give %.2f and %.2f, so 90+ Days Late should read 'about 8.5 times' (8.6 results from dividing the rounded means 0.0223 and 0.0026). Credit Utilisation Rate (%.2f) is consistent with 'about 1.4'." % (s6["Credit Utilisation Rate"]["mean"] / s6["Credit Utilisation Rate"]["q"][2], s6["90+ Days Late"]["mean"] / s6["90+ Days Late"]["q"][2], s6["Credit Utilisation Rate"]["mean"] / s6["Credit Utilisation Rate"]["q"][2]),
    ])
    S7 = SB["S7"]
    if S7_ACCEPTED:
        EDITS.append("Supplement Section S5 and Fig S7: the text says 'The per-iteration history was not saved, so this figure is the only record.' This is no longer true as written: Fig S7 is a deterministic rerun of the 100-iteration EM (same structure, ML start on the complete rows, no prior, no early stop) that reproduces the stored iteration-79 values (train %.6f, test %.6f; differences from the stored values %+.6f and %+.6f) and the stored iteration-79 CPDs (max absolute difference about 1e-14). Say so in the text and the caption (for example: 'The per-iteration history of the original run was not saved; the curves are from a deterministic rerun that reproduces the iteration-79 model'). The text claim that the test log-likelihood peaked at iteration 79 is confirmed (peak at %d); the claim that iteration 100 lay about 0.6 units below the peak is %s (rerun: %.3f). The caption should also say that the figure now has stacked train and test panels and a zoom panel marking iteration 79 (dashed line, star in the zoom)." % (S7["train79"], S7["test79"], S7["d_train_pkl"], S7["d_test_pkl"], S7["peak"], "confirmed" if abs(S7["test100_minus_peak"] + 0.6) < 0.1 else "NOT matched", S7["test100_minus_peak"]))
    for nm in order: save(figs[nm], nm)
    # pdf font check
    for nm in order:
        b = open(f"{OUT}/{nm}.pdf", "rb").read()
        AUDITS[nm]["pdf_fonts_embedded"] = (b"/FontFile" in b) and (b"/Type3" not in b)
    return order

S7_ACCEPTED = None

def contact_sheet(order):
    DPI = 200; pad = 24; lab_h = 22
    ims = {nm: Image.open(f"{OUT}/{nm}_preview.png").convert("RGB") for nm in order}
    try: font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except Exception: font = ImageFont.load_default()
    wfull = ims[order[0]].width
    names = set(order); rows = [[order[0]], [order[1]], [order[2], order[3]], [order[4]], ["fig4_cost_saving_with_differences"], ["fig6_flip_rate_vs_r"], [order[5]], ["figS1_roc", "figS5_sexp_distribution_application"], ["figS6_sexp_distribution_after_acceptance"],
                                ["figS2_calibration"], ["figS3_masking_application"], ["figS4_masking_after_acceptance"], ["figS8_masking_paired_differences_after_acceptance"], ["figS9_split_robustness"]]
    if "figS7_em_convergence" in names: rows.append(["figS7_em_convergence"])
    else:
        import figS7
        o = Image.open(figS7.ORIG).convert("RGB"); o = o.resize((wfull, round(o.height * wfull / o.width)), Image.LANCZOS)
        ims["figS7_em_convergence"] = o; AUDITS["figS7_em_convergence"] = dict(size_mm=(173.6, round(o.height / 200 * 25.4, 1))); rows.append(["figS7_em_convergence"])
    height = pad
    for r in rows: height += lab_h + max(ims[n].height for n in r) + pad
    sheet = Image.new("RGB", (wfull + 2 * pad, height), "white"); d = ImageDraw.Draw(sheet); y = pad
    for r in rows:
        x = pad; hmax = max(ims[n].height for n in r)
        for n in r:
            d.text((x, y), n.split("_")[0] + "  %.1f x %.1f mm, 1:1 at %d dpi" % (AUDITS[n]["size_mm"][0], AUDITS[n]["size_mm"][1], DPI) + ("  (ORIGINAL figure, copied unmodified, scaled to full width)" if n == "figS7_em_convergence" and "figS7_em_convergence" not in names else ""), fill=(90, 90, 90), font=font)
            sheet.paste(ims[n], (x, y + lab_h)); d.rectangle([x - 1, y + lab_h - 1, x + ims[n].width, y + lab_h + ims[n].height], outline=(190, 190, 190))
            x += ims[n].width + pad
        y += lab_h + hmax + pad
    sheet.save(f"{OUT}/contact_sheet.png", dpi=(DPI, DPI))

def reports(order):
    ok_all = all(c["ok"] for c in CHECKS) and all(AUDITS[n]["pass"] for n in order)
    src = dict(figsB.SRC); src.update({
        "fig1a_structure_application": "selection/tables/sel_dag_9node_edges.json",
        "fig1b_structure_after_acceptance": "full_variable/models/bn_final_model_iteration79.pkl (model_edges)",
        "fig2_sexp_application_and_after_acceptance": "selection/tables/sel_rq2_sensitivity_table.csv; full_variable/tables/rq2_sensitivity_table.csv",
        "fig3_application_lookup_heatmap": "selection/tables/sel_default_cells.csv",
        "fig4_cost_saving": "selection/tables/sel_decision_value_combined.csv",
        "fig5_masking_paired_differences": "selection/tables/sel_rq3_row_bootstrap_summary.csv",
        "fig4_cost_saving_with_differences": "selection/tables/sel_decision_value_combined.csv; analysis_additions/cost_saving_by_r_with_intervals.csv (J2 pipeline, every r)",
        "fig6_flip_rate_vs_r": "analysis_additions/flip_rate_by_r.csv (J10)",
        "figS8_masking_paired_differences_after_acceptance": "analysis_additions/mon_rq3_row_bootstrap_summary.csv (J8); monitoring_rq3/mon_rq3_summary_v2.csv",
        "figS9_split_robustness": "analysis_additions/j12_summary_table.csv (J12, J15); ledger rows N-ADD-308 to 592 and the main-split rows"})
    L = ["# Verify report: main article figures (Tier A) and supplement figures (Tier B)", "",
         "Ledger: `Submissions/4. Article/WRITEUP/ledgers/numbers_ledger_v3.csv` (408 rows). Plots only from saved outputs; no model was refitted or retrained. Overall result: **%s**." % ("ALL CHECKS PASS" if ok_all else "FAILURES PRESENT"), "",
         "Rounding rule: each plotted value or printed label is rounded to the number of decimals printed and compared with the ledger value rounded the same way.", "",
         "## Summary", "", "| Figure | Final size (mm) | PNG px (600 dpi) | Smallest font (pt) | 7 pt audit | Text/text overlap | Text vs node/marker overlap | Text outside canvas | Ledger/source assertions | Fonts embedded in PDF |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in order:
        a = AUDITS[n]; cs = [c for c in CHECKS if c["figure"] == n]
        L.append("| %s | %.1f x %.1f | %dx%d | %.1f | %s | %d | %d | %d | %d/%d pass | %s |" % (n, a["size_mm"][0], a["size_mm"][1], a["png_px"][0], a["png_px"][1], a["min_font"],
                 "PASS" if not a["under7"] else "FAIL", len(a["text_text"]) + len(a["legend_block"]), len(a["text_marker"]), len(a["outside"]), sum(c["ok"] for c in cs), len(cs), "yes" if a["pdf_fonts_embedded"] else "NO"))
    L += ["", "Final sizes: single column 83.7 mm, full width 173.6 mm; figsize equals the final size, so LaTeX should include each file at 1:1 (no scaling). Every PNG pixel size equals size x 600 dpi to within 1 px (rounding).", ""]
    for n in order:
        a = AUDITS[n]
        L += ["## " + n, "", "- Source data: " + src[n],
              "- Final size: %.2f x %.2f mm; 600 dpi PNG %dx%d px (expected %dx%d); vector PDF with embedded TrueType fonts (fonttype 42): %s." % (a["size_mm"][0], a["size_mm"][1], *a["png_px"], *a["png_px_expected"], "yes" if a["pdf_fonts_embedded"] else "NO"),
              "- 7 pt audit: %d text objects enumerated, smallest %.1f pt, under 7 pt: %s." % (a["n_text"], a["min_font"], a["under7"] or "none"),
              "- Overlap audit: text/text %s; text/node-or-marker %s; text/legend %s; text outside canvas %s; titles or suptitles %s." % (a["text_text"] or "none", a["text_marker"] or "none", a["legend_block"] or "none", a["outside"] or "none", a["titles"] or "none")]
        if "structure" in a:
            s = a["structure"]
            L += ["- Structure: %d nodes, %d edges, all edges of the stored edge list drawn (asserted). Derived blanket of Default: parents %s; children %s; co-parents of children %s; blanket (orange outline) = %s." % (s["nodes"], s["edges"], s["parents"], s["children"], s["coparents"], s["blanket"]),
                  "- No edge passes through a non-endpoint node (asserted). Label-vs-edge crossings left after the search: %d (an edge that passes behind a label does so under the label's thin white outline). Layout is fixed (coordinates in `code/run_tierA.py`); the label side of each node was chosen by a deterministic search, no randomness." % a["edge_label_hits"],
                  "- Full variable names were used at 8 pt (no short forms). Indicators are named M_Income and M_Dependents."]
        L.append("")
    L += ["## Deviations from the spec and judgement calls", "",
          "- Fig 1: edge lists compared with the figures currently in the article (sel_dag_9node_v3.png, bn_13node_dag_final.png): **no edge differs** (15 of 15 and 24 of 24; checked by reading the edges in the images against the stored lists). Derived blankets equal the expected sets.",
          "- Fig 1: heights are 73.7 mm (limit 75 mm); full names fit at 8 pt, so the short forms were not used.",
          "- Fig 2: single image 83.7 x %.1f mm (limit 70 mm). Bars are drawn with a light fill (40%% opacity) so the in-bar value labels stay legible; the bootstrap whiskers are very narrow because the intervals are narrow (about 0.0003 wide at Application). The old per-stage 'flip rate' panels are not part of this figure." % AUDITS["fig2_sexp_application_and_after_acceptance"]["size_mm"][1],
          "- Fig 2: the Application ledger prints 5 decimals for the S_exp means (for example Age 0.02697); labels show 4 decimals (0.0270, 0.0122) and are asserted against the ledger rounded to 4 decimals.",
          "- Fig 3: single column kept (83.7 x %.1f mm). Training count n per cell kept at 7 pt (printed as the lower number, without 'n ='). The colour bar runs 0 to 0.18 so that the 1/6 threshold is on the bar; no cell reaches 1/6, and the highest cell is 0.1194. Cell counts n are not in the ledger and are asserted against sel_default_cells.csv only (shown as the source in the table below)." % AUDITS["fig3_application_lookup_heatmap"]["size_mm"][1],
          "- Fig 4: r* is drawn per stage (13.96 at Application, 13.98 after acceptance; ledger N-XST-020 and N-XST-022). r = 15 and 20 are no longer marked by dotted lines. The y label is two lines ('Saving vs cheaper' / 'trivial policy (%)') so it fits the panel height at 8 pt.",
          "- Fig 5: height 60.0 mm. The y label is four lines for the same reason. Colour follows the comparator (blue = LR, orange = RF); mechanism is shown by filled/solid (MCAR) versus open/dashed (dependent).",
          "- Colours (Okabe-Ito): BN green #009E73, LR blue #0072B2, RF orange #E69F00. The Markov blanket outline in Fig 1 uses the same orange.",
          "- Font: Arial (sans-serif), mathtext set to Arial so that S_exp and the italic r match the text. Python: project virtual environment (matplotlib 3.11.1).",
          "- Not in the ledger, asserted against the source file instead: Fig 3 cell counts n; Fig 4 values at r other than 5, 10, 14, 15, 20 are plotted from the CSV without a ledger row (the 12 ledger-covered r values per model per stage are asserted); Fig 2 and Fig 5 are fully ledger-covered.", ""]
    from figsB import STATS
    L += ["## Tier B details", ""]
    L += ["### S5 and S6: statistics, rows used and what the stored columns are", "",
          "- S5 (Application). Source: `selection/predictions/sel_rq2_per_row.pkl`, column `S_exp`, one record per test row and variable (each test row counted once). Age: n = %d rows (2 test rows have undefined S_exp and are excluded; n_rows in the table is 29,998); Debt Ratio: n = %d rows (1 row has S_exp exactly 0, included in all statistics, not visible on the log axis). Each statistic (5th, 25th, 50th, 75th, 95th, 99th percentile and the mean) uses all rows listed. Mean asserted against ledger N-SEL-016/017; median and 99th percentile against `sel_rq2_sensitivity_table.csv`." % (STATS["S5_Age"]["n"], STATS["S5_Debt Ratio"]["n"]),
          "- The four omitted S5 variables (Monthly Income, Number of Dependents, Real Estate Loans, Open Credit Lines) were checked to have S_exp = 0 on every stored row.", "",
          "- S6 (After acceptance). IMPORTANT: the per-row column `S` of `full_variable/per_row/rq2_sensitivity_per_row.pkl` is **S_max**, not S_exp (its per-variable means 0.0847, 0.1648, 0.4388, 0.5786 equal the S_max means in `rq2_sensitivity_table.csv`, not the ledger S_exp means). It was therefore NOT used. Per-row S_exp is stored in `full_variable/per_row/vo_blanket_original_per_row.pkl` (column `S_exp`; 29,998 test rows per blanket variable, `row_index` unique within each variable, so every test row is counted once; no unique-pattern aggregation). Its means equal the table means to 8 decimals and the ledger values to 4 (asserted).",
          "- Stored column type per variable (from `rq2_pattern_level_check.csv`, 205 blanket patterns): Credit Utilisation Rate **row-level** (within-pattern spread 0.269, 5,747 distinct values); 30-59 Days Past Due **pattern-level** (spread 0, 203 distinct values); 90+ Days Late **row-level** (spread 0.074, 629 distinct values); 60-89 Days Past Due **pattern-level** (spread 0, 205 distinct values). In all four cases the stored `S_exp` column holds one value per test row; for the pattern-level variables rows sharing a pattern simply have equal values.",
          "", "| S6 variable | rows used (each statistic) | S_exp = 0 rows (not visible on log axis) | rows flagged `skipped` in the file | 5th | 25th | median | 75th | 95th | 99th | mean | mean / median |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in ["Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due"]:
        t = STATS["S6_" + v]; q = t["q"]
        L.append("| %s | %d | %d | %d | %.5f | %.5f | %.5f | %.5f | %.5f | %.5f | %.5f | %.2f |" % (v, t["n"], t["nzero"], t["skipped"], q[0], q[1], q[2], q[3], q[4], q[5], t["mean"], t["mean"] / q[2]))
    L += ["", "### S7: EM rerun", ""]
    S7 = STATS.get("S7")
    if S7:
        L += ["- Rerun (`code/s7_em_rerun.py`, history in `code/s7_em_history.csv`): saved `shared/src/em.py` (verbatim port of the locked E/M steps, VariableElimination E-step over unique rows) and `loglik.py`; structure = the 24 edges of the stored iteration-79 model; maximum-likelihood start on the 96,251 complete training rows; no prior; no early stop; 100 iterations; PYTHONHASHSEED=0. As a prior sanity check, `loglik.py` applied to the stored iteration-79 model reproduces the stored train and test log-likelihoods to 1e-9.",
              "- Peak of the test log-likelihood: iteration **%d** (required 79). Iterations run: %d." % (S7["peak"], S7["n_iter"]),
              "- Iteration 79 train LL %.6f; stored (pickle) %.6f; difference **%+.6f**; stated value -1,316,746.532478, difference %+.6f." % (S7["train79"], S7["stored_train"], S7["d_train_pkl"], S7["d_train_stated"]),
              "- Iteration 79 test LL %.6f; stored (pickle) %.6f; difference **%+.6f**; stated value -329,111.347037, difference %+.6f." % (S7["test79"], S7["stored_test"], S7["d_test_pkl"], S7["d_test_stated"]),
              "- Test LL at iteration 100 minus the peak: %+.3f (supplement text says about 0.6 below the peak)." % S7["test100_minus_peak"],
              "- Acceptance rule (peak at 79 and both differences within 0.001): **%s**. %s" % ("ACCEPTED" if S7["accepted"] else "NOT ACCEPTED", "The figure is the rerun." if S7["accepted"] else "The original `em_convergence_final_with_zoom.png` was copied unmodified (PNG only; no PDF or 600 dpi version exists for it).")]
        if S7.get("extra"): L += ["- " + S7["extra"]]
    L += [""]
    L += ["### Tier B deviations and judgement calls", "",
          "- S1: single column, 83.7 x 83.7 mm; legend AUCs are the ledger values (BN 0.6627, LR 0.6904, RF 0.6926).",
          "- S2: full width 173.6 x 70.0 mm, each panel keeps the limits of the old figure (BN 0 to 0.15, LR/RF 0 to 1). BN marker area is proportional to the square root of n (a legend gives n = 200, 2,000, 10,000); the old figure used area proportional to n. Ledger covers only the BN mean (0.0669) and observed rate (0.0668); the point coordinates are plotted from the saved predictions.",
          "- S3 and S4: lines are the means and bands the SD across the 10 mask seeds, from the stored summary files (SD not in the ledger). The plotted differences of the BN and LR/RF primary means are asserted against all ledger paired-difference rows (24 for S3, 48 for S4, refit-imputed rows) and against the stored `BN_minus_*_primary` rows. Only the three levels 20, 40 and 50% are drawn (no 0% point).",
          "- S5 and S6: the box shows the 25th to 75th percentile, whiskers 5th to 95th, as specified; log axes. S5 y range 0.001 to 0.3; S6 y range 0.0003 to 8 (the extra headroom holds the legend).",
          "- Colour/line encoding identical to Tier A: BN green circle solid (thicker), LR blue square dashed, RF orange triangle dotted.", ""]
    L += ["## Assertion table", "", "| Figure | Item | Plotted | Reference | Decimals | Source | Result |", "|---|---|---|---|---|---|---|"]
    for c in CHECKS:
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (c["figure"].split("_")[0], c["item"], c["plotted"], c["reference"], c["decimals"], c["source"], "PASS" if c["ok"] else "FAIL"))
    open(f"{OUT}/verify_report.md", "w").write("\n".join(L) + "\n")
    pd.DataFrame(CHECKS).to_csv(f"{OUT}/code/assertions_tierA.csv", index=False)
    C = ["# Caption notes", "", "Text removed from the figures (titles, footnotes, panel titles) and facts to move into the captions, per figure.", ""]
    for n in order:
        C += ["## " + n, ""] + ["- " + t for t in NOTES.get(n, [])] + [""]
    C += ["## ARTICLE EDITS NEEDED", "", "(The article was not edited. These follow from the figure changes.)", ""] + ["%d. %s" % (i + 1, t) for i, t in enumerate(EDITS)]
    open(f"{OUT}/caption_notes.md", "w").write("\n".join(C) + "\n")
    return ok_all

if __name__ == "__main__":
    order = main(); contact_sheet(order); ok = reports(order)
    print("ALL PASS" if ok else "FAILURES"); print([c for c in CHECKS if not c["ok"]])
    for n in order: print(n, {k: v for k, v in AUDITS[n].items() if k in ("size_mm", "min_font", "pass", "png_px", "pdf_fonts_embedded")})
