"""Fig 5: paired BN minus LR / BN minus RF change in AUC loss under masking (Application)."""
import os
from common import *
from matplotlib.lines import Line2D
NAME = "fig5_masking_paired_differences"

def ledger_row(variable, mech, level, comp):
    m = LED[(LED.stage == "rq3-selection") & (LED.variable == variable) & (LED.mechanism == mech) & (LED.level == "%d%%" % round(level * 100))
            & (LED.comparator == comp) & (LED.imputer == "refit-imputed")]
    assert len(m) == 1, (variable, mech, level, comp, len(m)); return m.index[0]

def make():
    d = pd.read_csv(R + "/selection/tables/sel_rq3_row_bootstrap_summary.csv")
    assert len(d) == 24
    H = 60 * MM
    fig = plt.figure(figsize=(W2, H))
    Lm, Rm, Tm, Bm = 0.92, 0.10, 0.08, 0.74; gap = 0.18
    aw = (W2 - Lm - Rm - gap) / 2; ah = H - Tm - Bm
    axs = [fig.add_axes([(Lm + k * (aw + gap)) / W2, Bm / H, aw / W2, ah / H]) for k in range(2)]
    series = [("LR", "mcar", -1.8), ("LR", "dependent", -0.6), ("RF", "mcar", 0.6), ("RF", "dependent", 1.8)]
    levels = [0.2, 0.4, 0.5]
    for ax, var, lab in zip(axs, ("Age", "Debt Ratio"), ("(a) Age", "(b) Debt Ratio")):
        for comp, mech, off in series:
            x, y, lo, hi = [], [], [], []
            for lv_ in levels:
                r = d[(d.variable == var) & (d.mechanism == mech) & (d.level == lv_) & (d.contrast == f"BN minus {comp}")]
                assert len(r) == 1; r = r.iloc[0]
                lid = ledger_row(var, mech, lv_, comp)
                check(NAME, f"{var} BN-{comp} {mech} {int(lv_*100)}%: estimate", r.point_estimate, lv(lid), 4, "ledger " + lid)
                c0, c1 = lci(lid)
                check(NAME, f"{var} BN-{comp} {mech} {int(lv_*100)}%: CI low", r.ci_lo, c0, 4, "ledger " + lid)
                check(NAME, f"{var} BN-{comp} {mech} {int(lv_*100)}%: CI high", r.ci_hi, c1, 4, "ledger " + lid)
                x.append(lv_ * 100 + off); y.append(r.point_estimate); lo.append(r.point_estimate - r.ci_lo); hi.append(r.ci_hi - r.point_estimate)
            filled = mech == "mcar"
            ax.errorbar(x, y, yerr=[lo, hi], color=COL[comp], ls=("-" if filled else "--"), lw=0.9, marker=MK[comp], ms=3.6,
                        mfc=(COL[comp] if filled else "white"), mec=COL[comp], mew=0.9, elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=3)
        ax.axhline(0, color="black", lw=0.7, zorder=1)
        ax.set_xlim(15, 55); ax.set_xticks([20, 40, 50]); ax.set_xticklabels(["20%", "40%", "50%"])
        ax.set_ylim(-0.009, 0.0185)
        ax.set_yticks([-0.005, 0, 0.005, 0.010, 0.015]); ax.set_yticklabels(["−0.005", "0.000", "0.005", "0.010", "0.015"])
        ax.set_xlabel("Share of test rows with the variable masked")
        ax.text(0.02, 0.97, lab, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
        ax.grid(axis="y", color="0.92", lw=0.5, zorder=0); ax.set_axisbelow(True)
    axs[1].tick_params(labelleft=False)
    axs[0].set_ylabel("AUC loss,\nBN minus comparator")
    hs = [Line2D([], [], color=COL[c], ls=("-" if m == "mcar" else "--"), marker=MK[c], ms=3.6, lw=0.9, mfc=(COL[c] if m == "mcar" else "white"), mec=COL[c], mew=0.9,
                 label=f"BN minus {c}, {'MCAR' if m == 'mcar' else 'dependent'}") for c, m, _ in series]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=2.6, borderaxespad=0.15)
    NOTES[NAME] = ["Removed in-figure texts (old sel_rq3_paired_ci.png): title 'Selection RQ3: paired BN minus LR / RF change in AUC, 95% test-row bootstrap intervals'; panel titles 'Age' and 'Debt Ratio' (now '(a) Age', '(b) Debt Ratio' inside the panels).",
                   "Removed footnote (move to caption): 'Positive = the BN degrades less. Intervals reflect test-row sampling and averaging over 10 mask seeds; models fixed. Primary imputer.'",
                   "Points are the stored point estimates with the stored 95% test-row bootstrap intervals (2,000 resamples), primary (refit-imputed) imputer; all 24 points and intervals are asserted against the ledger. Points are dodged horizontally within each masking level; the lines join the levels.",
                   "Y label is now 'AUC loss, BN minus comparator' (two lines). The statement 'positive = BN degrades less' is no longer in the figure and MUST be in the caption (it was in the old footnote and in the old y label).",
                   "Colour follows the comparator (blue = LR, orange = RF); filled markers and solid lines are MCAR masking, open markers and dashed lines are dependent masking; marker shape follows the comparator (square = LR, triangle = RF)."]
    EDITS.append("Fig 5 caption: state that positive values mean the BN degrades less (no longer in the y label); mention that filled markers/solid lines are MCAR and open markers/dashed lines are dependent masking, that colours and marker shapes identify the comparator (LR, RF), and carry over the footnote about test-row sampling, 10 mask seeds and the primary imputer.")
    return fig

if __name__ == "__main__":
    os.makedirs(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out"), exist_ok=True); f = make(); print(audit(f, NAME)); f.savefig(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out", "f5.png"), dpi=250)
    print([c for c in CHECKS if not c["ok"]], len(CHECKS))
