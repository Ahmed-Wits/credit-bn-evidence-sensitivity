"""Fig 4 extended: (a),(b) saving vs r (as Fig 4) and (c),(d) saving differences BN minus LR and BN minus RF with shaded 95% paired bootstrap intervals."""
import os
from common import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import fig4 as F4
NAME = "fig4_cost_saving_with_differences"
ADD = WR + "/analysis_additions"

def parse_pts(disp):
    d = disp.replace("−", "-").replace(" pts", "").replace("[", "").replace("]", "").replace(",", ""); return [float(x) for x in d.split()]

def make():
    d = pd.read_csv(R + "/selection/tables/sel_decision_value_combined.csv")
    D = pd.read_csv(ADD + "/cost_saving_by_r_with_intervals.csv")
    v4 = pd.read_csv(ADD + "/ledger_additions_v4.csv", dtype=str, keep_default_na=False).set_index("ledger_id")
    H = 92 * MM
    fig = plt.figure(figsize=(W2, H))
    Lm, Rm, Tm, gap = 0.66, 0.12, 0.27, 0.62
    aw = (W2 - Lm - Rm - gap) / 2; ah = 1.12; Bm = 0.62; rowgap = 0.36
    ys = [Bm + ah + rowgap, Bm]
    axs = [[fig.add_axes([(Lm + k * (aw + gap)) / W2, ys[row] / H, aw / W2, ah / H]) for k in range(2)] for row in range(2)]
    stages = (("selection", "Application", "(a)\nApplication", "(c)\nApplication", (32, (0, 31)), (-11.5, 3.0)),
              ("monitoring", "After acceptance", "(b) After\nacceptance", "(d) After\nacceptance", (74, None), (-5.5, 3.5)))
    for k, (st, lab, la, lc, (ymax, _), ylim2) in enumerate(stages):
        ax = axs[0][k]; s = d[d.contribution == st]
        n0 = s.reject_all.iloc[0]; n1 = s.accept_all.iloc[0] / s.r.iloc[0]; rstar = n0 / n1
        check(NAME, f"{lab}: r* = N0/N1", rstar, lv(F4.RSTAR[st]), 2, "ledger " + F4.RSTAR[st])
        for model in ("LR", "RF", "BN"):
            m = s[s.model == model].sort_values("r")
            ax.plot(m.r, 100 * m.saving, color=COL[model], ls=LS[model], marker=MK[model], ms=(3.6 if model == "BN" else 2.8), lw=(2.2 if model == "BN" else 0.9), zorder=(4 if model == "BN" else 3), mew=0)
            for r in (5, 10, 15, 20):
                v = float(m[m.r == r].saving.iloc[0]); F4.check_pct_named(NAME, f"{lab} {model} r={r}", v, F4.IDS[(st, model, r)])
        # the bottom row, top row decoration
        for row in (0, 1):
            a = axs[row][k]
            for xv, t in ((5, "$r$ = 5"), (10, "$r$ = 10"), (rstar, "$r^{*}$ = %.2f" % rstar)):
                a.axvline(xv, color="0.3", lw=0.8, ls=(0, (1, 2)), zorder=1)
                if row == 0: a.text(xv, 1.0, t, transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, clip_on=False)
            a.set_xlim(0.5, 20.5); a.set_xticks([1, 5, 10, 15, 20]); a.set_xlabel("Cost ratio $r$" if row == 1 else ""); a.grid(axis="y", color="0.92", lw=0.5, zorder=0); a.set_axisbelow(True)
        ax.set_ylim(0, ymax); ax.set_yticks(range(0, 31, 5) if ymax < 40 else range(0, 61, 10)); ax.set_ylabel("Saving vs cheaper\ntrivial policy (%)")
        ax.text(0.99, 0.98, la, transform=ax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold", linespacing=1.1)
        # differences with bands
        ax2 = axs[1][k]
        for comp in ("LR", "RF"):
            q = D[(D.stage == lab) & (D.kind == "difference") & (D.comparator == comp)].sort_values("r")
            assert len(q) == 20
            ax2.fill_between(q.r, q.lo, q.hi, color=COL[comp], alpha=0.20, lw=0, zorder=2)
            ax2.plot(q.r, q.point, color=COL[comp], ls=LS[comp], marker=MK[comp], ms=2.8, lw=1.0, mew=0, zorder=3)
        ax2.axhline(0, color="black", lw=0.7, zorder=1)
        ax2.set_ylim(*ylim2); ax2.set_ylabel("BN minus comparator\nin saving (points)")
        ax2.text(0.99, 0.02, lc, transform=ax2.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold", linespacing=1.1)
    # gate: r = 5, 10 and the named extremes equal the v4 ledger rows (via the CSV used for the bands)
    for lid, row in v4.iterrows():
        n = int(lid.split("-")[2])
        if not (22 <= n <= 34): continue
        q = row.quantity; stage = "Application" if "Application" in q else "After acceptance"; r = int(q.split("r=")[1].split()[0].rstrip(","))
        a_, b_ = q.split("difference ")[1].split(" with")[0].split(" minus ")
        x = D[(D.stage == stage) & (D.kind == "difference") & (D.model == a_) & (D.comparator == b_) & (D.r == r)].iloc[0]; pv, lo, hi = parse_pts(row.display)
        for nm, val, ref in (("difference", x.point, pv), ("lower bound", x.lo, lo), ("upper bound", x.hi, hi)):
            check(NAME, f"{lid} {stage} BN minus {b_} r={r}: plotted {nm}", val, ref, 2, "ledger " + lid)
    hs = [Line2D([], [], color=COL[m], ls=LS[m], marker=MK[m], ms=(3.6 if m == "BN" else 2.8), lw=(2.2 if m == "BN" else 0.9), label=m, mew=0) for m in ("BN", "LR", "RF")] + \
         [Patch(facecolor=COL[c], alpha=0.35, label=f"BN minus {c}, 95% interval") for c in ("LR", "RF")]
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.8, handlelength=2.4, borderaxespad=0.15)
    NOTES[NAME] = ["Extension of fig4_cost_saving (kept unchanged). Top row (a), (b): saving against the cheaper of flag-none and flag-all, as before. Bottom row (c), (d): saving of the BN minus the saving of LR (blue, dashed, squares) and of RF (orange, dotted, triangles), in percentage points, with shaded 95% intervals from the paired test-row bootstrap (2,000 resamples, seed 42, predictions and thresholds fixed; resample counts from numpy default_rng(42); the same resample for every model and r; the cheaper trivial policy is recomputed within each resample). The zero line marks equal savings; negative values mean the BN saves less. The dotted vertical lines mark r = 5, r = 10 and r* = N0/N1 (13.96 Application, 13.98 After acceptance) in all four panels.",
                   "Suggested caption: 'Cost saving against the cheaper of flag-none and flag-all for cost ratios r = 1 to 20 (top row) and the saving of the BN minus the saving of LR and of RF (bottom row), at application (a, c) and after acceptance (b, d). Bands are 95% paired bootstrap intervals (2,000 test-row resamples, thresholds and predictions fixed). Where a band includes zero the difference cannot be told from zero. Dotted lines mark r = 5, r = 10 and the switch ratio r*.'",
                   "Statement for the text: intervals in the bottom row are the ones in Supplement Table S-Numbers (ledger N-ADD-022 to 034 for r = 5, 10, 2, 3, 8, 11); the figure and table agree by construction (asserted).",
                   "Y labels are wrapped over two lines to fit the panel height at 8 pt."]
    EDITS.append("Fig 4 replaced by fig4_cost_saving_with_differences (the old fig4_cost_saving is kept): the caption must now describe the bottom row (BN minus LR / RF saving differences with 95% paired bootstrap bands) and the dotted lines (r = 5, 10, r*). Section 4.2.4 sentence 'No intervals were computed for cost saving' is no longer true; the named extremes now carry intervals (J2).")
    return fig

if __name__ == "__main__":
    os.makedirs(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out"), exist_ok=True); f = make(); print(audit(f, NAME)); f.savefig(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out", "f4x.png"), dpi=200)
    print([c for c in CHECKS if not c["ok"]], len(CHECKS))
