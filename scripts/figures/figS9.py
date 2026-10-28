"""Fig S9: split-seed robustness. (a),(b) paired AUC difference BN minus LR and BN minus RF per split; (c),(d) saving difference at r = 5 and 10 per split (95% paired bootstrap intervals)."""
import os
from common import *
from matplotlib.lines import Line2D
NAME = "figS9_split_robustness"
ADD = WR + "/analysis_additions"
SEEDS = [42, 1, 2, 3, 4, 5]

def nums(d): return [float(x) for x in d.replace("−", "-").replace(" pts", "").replace("%", "").replace("[", "").replace("]", "").replace(",", "").split()]

def ledger_lookup():
    L = {i: r.display for i, r in LED.iterrows()}
    for f in ("ledger_additions_v4.csv", "ledger_additions_v5.csv", "ledger_additions_v6.csv"):
        d = pd.read_csv(ADD + "/" + f, dtype=str, keep_default_na=False); L.update({r.ledger_id: r.display for r in d.itertuples()})
    return L

def make():
    T = pd.read_csv(ADD + "/j12_summary_table.csv"); T = T[T.row_type.isin(["main split", "split run"])].copy(); T["split_seed"] = T.split_seed.astype(int)
    L = ledger_lookup()
    H = 100 * MM; fig = plt.figure(figsize=(W2, H))
    Lm, Rm, Tm, Bm, gx, gy = 0.95, 0.10, 0.08, 0.92, 0.88, 0.52
    aw = (W2 - Lm - Rm - gx) / 2; ah = (H - Tm - Bm - gy) / 2
    ax = {}
    for (rr, cc, key) in ((0, 0, "a"), (0, 1, "b"), (1, 0, "c"), (1, 1, "d")):
        ax[key] = fig.add_axes([(Lm + cc * (aw + gx)) / W2, (Bm + (1 - rr) * (ah + gy)) / H, aw / W2, ah / H])
    stages = {"a": "Application", "b": "After acceptance", "c": "Application", "d": "After acceptance"}
    ylims = {"a": (-0.052, 0.012), "b": (-0.0215, 0.0040), "c": (-12.8, 4.2), "d": (-4.4, 3.8)}
    xl = [f"{s}\n(main split)" if s == 42 else str(s) for s in SEEDS]
    def chk(item, val, lid, idx, dec):
        ref = nums(L[lid])[idx]; check(NAME, item, val, ref, dec, "ledger " + lid)
    for key, a in ax.items():
        a.axvspan(-0.5, 0.5, color="0.93", lw=0, zorder=0); a.axhline(0, color="black", lw=0.7, zorder=1)
        a.set_xlim(-0.6, 5.6); a.set_xticks(range(6)); a.set_xticklabels(xl); a.set_ylim(*ylims[key]); a.grid(axis="y", color="0.92", lw=0.5, zorder=0); a.set_axisbelow(True)
        lab = {"a": "(a) Application", "b": "(b) After acceptance", "c": "(c) Application", "d": "(d) After acceptance"}[key]
        a.text(0.015, 0.985, lab, transform=a.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
        st = stages[key]; D = T[T.stage == st].set_index("split_seed")
        if key in "ab":
            for comp, off in (("LR", -0.16), ("RF", 0.16)):
                xs, ys, lo, hi = [], [], [], []
                for k, s in enumerate(SEEDS):
                    r = D.loc[s]; v, l_, h_ = r[f"auc_diff_BN_minus_{comp}"], r[f"auc_diff_BN_minus_{comp}_lo"], r[f"auc_diff_BN_minus_{comp}_hi"]; lid = r[f"auc_diff_BN_minus_{comp}_id"]
                    for nm, val, idx in (("estimate", v, 0), ("lower bound", l_, 1), ("upper bound", h_, 2)): chk(f"{st} seed {s} AUC difference BN minus {comp}: plotted {nm}", val, lid, idx, 4)
                    xs.append(k + off); ys.append(v); lo.append(v - l_); hi.append(h_ - v)
                a.errorbar(xs, ys, yerr=[lo, hi], fmt=MK[comp], color=COL[comp], ms=3.8, mew=0, elinewidth=0.9, capsize=1.8, capthick=0.9, zorder=3, ls="none")
            a.set_ylabel("AUC difference,\nBN minus comparator")
            if key == "a": a.set_yticks([-0.05, -0.04, -0.03, -0.02, -0.01, 0]); a.set_yticklabels(["−0.05", "−0.04", "−0.03", "−0.02", "−0.01", "0.00"])
            else: a.set_yticks([-0.020, -0.015, -0.010, -0.005, 0]); a.set_yticklabels(["−0.020", "−0.015", "−0.010", "−0.005", "0.000"])
        else:
            for comp, r_, off, filled in (("LR", 5, -0.27, True), ("LR", 10, -0.09, False), ("RF", 5, 0.09, True), ("RF", 10, 0.27, False)):
                xs, ys, lo, hi = [], [], [], []
                for k, s in enumerate(SEEDS):
                    r = D.loc[s]; pre = f"saving_diff_BN_minus_{comp}_r{r_}"; v, l_, h_ = r[pre], r[pre + "_lo"], r[pre + "_hi"]; lid = r[pre + "_id"]
                    for nm, val, idx in (("estimate", v, 0), ("lower bound", l_, 1), ("upper bound", h_, 2)): chk(f"{st} seed {s} saving difference BN minus {comp} r={r_}: plotted {nm}", val, lid, idx, 2)
                    xs.append(k + off); ys.append(v); lo.append(v - l_); hi.append(h_ - v)
                a.errorbar(xs, ys, yerr=[lo, hi], fmt=MK[comp], color=COL[comp], mfc=(COL[comp] if filled else "white"), mec=COL[comp], mew=0.9, ms=3.6, elinewidth=0.9, capsize=1.6, capthick=0.9, zorder=3, ls="none")
            a.set_ylabel("Saving difference,\nBN minus comparator (points)")
        a.set_xlabel("Split seed" if key in "cd" else "")
    hs = [Line2D([], [], color=COL["LR"], marker=MK["LR"], ls="", ms=3.8, mew=0, label="BN minus LR"), Line2D([], [], color=COL["RF"], marker=MK["RF"], ls="", ms=3.8, mew=0, label="BN minus RF"),
          Line2D([], [], color="0.35", marker="o", ls="", ms=3.6, mew=0.9, mfc="0.35", label="r = 5 (panels c, d)"), Line2D([], [], color="0.35", marker="o", ls="", ms=3.6, mew=0.9, mfc="white", label="r = 10 (panels c, d)")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=2.0, handlelength=1.4, borderaxespad=0.15)
    NOTES[NAME] = ["New figure. Split-seed robustness of the main comparisons: the whole pipeline (split, bin edges, structure search, BN parameters, LR and RF, thresholds) was redone for split seeds 1 to 5 and compared with the main split (seed 42, shaded). (a), (b): paired AUC difference, BN minus LR (blue squares) and BN minus RF (orange triangles), with 95% paired bootstrap intervals; (c), (d): saving difference at r = 5 (filled) and r = 10 (open) with 95% intervals (2,000 test-row resamples, predictions and thresholds fixed). Positive values mean that the BN is better. The zero line marks equal values.",
                   "Values are asserted against ledger rows N-ADD-308 to 592 (split seeds 1 to 5) and the main-split rows N-SEL-004, N-SEL-005, N-ADD-010 to 029 and the new rows N-ADD-729 to 732; the table behind the figure is analysis_additions/j12_summary_table.csv.",
                   "Y labels are wrapped over two lines to fit the panel height at 8 pt."]
    EDITS.append("Fig S9 (new, figS9_split_robustness) and Table S-Splits / Table S-Calibrated (J12, J13, J15): new Results subsection and supplement material; Table 6 LR minus BN to be printed as the observed difference +0.0098 (J14); the Markov blanket variation across splits to be stated in the text (Application 4 of 5, After acceptance 2 of 5 splits equal the main split).")
    return fig

if __name__ == "__main__":
    os.makedirs(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out"), exist_ok=True); f = make(); print(audit(f, NAME)); f.savefig(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out", "fS9.png"), dpi=200)
    print([c for c in CHECKS if not c["ok"]], len(CHECKS))
