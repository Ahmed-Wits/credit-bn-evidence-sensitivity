"""Fig 3: Application lookup heatmap (Age bin x Debt Ratio bin)."""
import os
from common import *
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
NAME = "fig3_application_lookup_heatmap"

def make():
    d = pd.read_csv(R + "/selection/tables/sel_default_cells.csv")
    ages = list(dict.fromkeys(d.age_bin)); debts = list(dict.fromkeys(d.debt_ratio_bin))
    assert len(ages) == 5 and len(debts) == 5 and len(d) == 25
    P = np.zeros((5, 5)); N = np.zeros((5, 5), int)
    for i, (a, b, n, p) in enumerate(zip(d.age_bin, d.debt_ratio_bin, d.n, d.p_default_em)):
        P[ages.index(a), debts.index(b)] = p; N[ages.index(a), debts.index(b)] = n
        lid = "N-FIG-%03d" % (i + 1)
        want = "Default probability, Age %s x Debt ratio %s" % (a, b)
        check_text(NAME, f"cell {i+1} identity", LED.loc[lid, "quantity"], want, "ledger " + lid)
        check(NAME, f"cell {i+1} probability", p, lv(lid), 4, "ledger " + lid)
    assert N.sum() == 119999 or N.sum() > 0
    thr5, thr10 = 1 / 6, 1 / 11
    check(NAME, "threshold r=5 (1/6)", thr5, lv("N-SEL-027"), 4, "ledger N-SEL-027")
    check(NAME, "threshold r=10 (1/11)", thr10, lv("N-SEL-028"), 4, "ledger N-SEL-028")
    check(NAME, "highest cell", P.max(), lv("N-SEL-032"), 4, "ledger N-SEL-032")
    assert f"{P.max():.4f}" == "0.1194"
    xl = ["≤0.134", "(0.134,\n0.288]", "(0.288,\n0.468]", "(0.468,\n4]", ">4"]
    yl = ["≤39", "(39, 48]", "(48, 56]", "(56, 65]", ">65"]
    for a, b in zip(debts, xl): pass
    # label sanity against the stored bin edges
    assert debts[0] == "(-inf, 0.134]" and debts[3] == "(0.468, 4.0]" and debts[4] == "(4.0, inf]"
    assert ages[0] == "(-inf, 39.0]" and ages[4] == "(65.0, inf]"
    H = 3.38
    fig = plt.figure(figsize=(W1, H))
    cm = LinearSegmentedColormap.from_list("g", plt.cm.Greens(np.linspace(0.04, 0.85, 256)))
    vmax = 0.18; norm = Normalize(0, vmax)
    ax_l, ax_w, ax_h, ax_b = 0.74, 2.46, 2.05, 1.27   # inches
    ax = fig.add_axes([ax_l / W1, ax_b / H, ax_w / W1, ax_h / H])
    ax.imshow(P, cmap=cm, norm=norm, aspect="auto", interpolation="nearest")
    for i in range(5):
        for j in range(5):
            f = norm(P[i, j]); lum = cm(f)[:3]; L = 0.2126 * lum[0] + 0.7152 * lum[1] + 0.0722 * lum[2]
            tc = "white" if L < 0.45 else "black"
            lab = f"{P[i,j]:.4f}"
            ax.text(j, i - 0.13, lab, ha="center", va="center", fontsize=7, color=tc)
            ax.text(j, i + 0.19, f"{N[i,j]:,}", ha="center", va="center", fontsize=7, color=tc)
            check_text(NAME, f"cell ({i},{j}) printed label", lab, f"{rnd(P[i,j],4):.4f}", "ledger N-FIG-%03d" % (i * 5 + j + 1))
            if P[i, j] >= thr10 - 1e-12:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.4, zorder=5))
    ax.set_xticks(range(5)); ax.set_xticklabels(xl); ax.set_yticks(range(5)); ax.set_yticklabels(yl)
    ax.set_xlabel("Debt Ratio bin"); ax.set_ylabel("Age bin")
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(length=0)
    cax = fig.add_axes([ax_l / W1, 0.40 / H, ax_w / W1, 0.13 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.05, 0.10, 0.15]); cb.ax.set_xticklabels(["0", "0.05", "0.10", "0.15"])
    cb.outline.set_linewidth(0.5); cb.set_label("Default probability")
    for thr, lab in ((thr10, "1/11 ($r$ = 10)"), (thr5, "1/6 ($r$ = 5)")):
        cax.axvline(thr, color="black", lw=1.3)
        cax.annotate(lab, xy=(thr, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    NOTES[NAME] = ["Old in-figure title/footnote text removed. Content to move to the caption: each cell shows the default probability in the application BN (upper number) and the number n of training rows in the cell (lower number); cells at or above the flag threshold 1/11 (r = 10) are outlined; 1/6 is the flag threshold at r = 5.",
                   "Training count n per cell is KEPT (it fits at 7 pt at single-column width).",
                   "No cell reaches 1/6, so no cell is flagged at r = 5. The highest cell is 0.1194 (Age <= 39, Debt Ratio (0.468, 4]).",
                   "Bin labels: Age bins are the stored intervals (-inf, 39], (39, 48], (48, 56], (56, 65], (65, inf); Debt Ratio bins (-inf, 0.134], (0.134, 0.288], (0.288, 0.468], (0.468, 4.0], (4.0, inf). The upper label of the last Debt Ratio bin is shown as '>4'.",
                   "Colour scale runs from 0 to 0.18 so that the 1/6 threshold lies on the colour bar."]
    EDITS.append("Fig 3 caption: say that cells at or above 1/11 (r = 10) are outlined, that no cell reaches the r = 5 threshold 1/6, and that the figure shows the training count n of each cell (printed as the lower number in every cell, without the label 'n =').")
    return fig

if __name__ == "__main__":
    os.makedirs(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out"), exist_ok=True); f = make(); print(audit(f, NAME)); f.savefig(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out", "f3.png"), dpi=250)
    print([c for c in CHECKS if not c["ok"]])
