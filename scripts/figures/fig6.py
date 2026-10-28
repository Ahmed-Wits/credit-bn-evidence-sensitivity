"""Fig 6: flip rate against r (1..20) with 95% interval bands, Application and After acceptance (from analysis_additions/flip_rate_by_r.csv, J10)."""
import os
from common import *
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
NAME = "fig6_flip_rate_vs_r"
ADD = WR + "/analysis_additions"
VC = {"Age": ("#CC79A7", "o", "-"), "Debt Ratio": ("#000000", "s", "--"),
      "Credit Utilisation Rate": ("#D55E00", "o", "-"), "30-59 Days Past Due": ("#56B4E9", "s", "--"), "90+ Days Late": ("#000000", "^", ":"), "60-89 Days Past Due": ("#CC79A7", "D", "-.")}

def pnum(disp):
    d = disp.replace("%", "").replace("[", "").replace("]", "").replace(",", ""); return [float(x) for x in d.split()]

def make():
    F = pd.read_csv(ADD + "/flip_rate_by_r.csv"); L6 = pd.read_csv(ADD + "/ledger_additions_v6.csv", dtype=str, keep_default_na=False)
    H = 62 * MM; fig = plt.figure(figsize=(W2, H))
    Lm, Tm, Bm = 0.52, 0.27, 0.50; wa, gap, wb, Rm = 1.80, 1.28, 1.80, 1.42
    ah = H - Tm - Bm
    axa = fig.add_axes([Lm / W2, Bm / H, wa / W2, ah / H]); axb = fig.add_axes([(Lm + wa + gap) / W2, Bm / H, wb / W2, ah / H])
    for ax, stage, names, lab in ((axa, "Application", ["Age", "Debt Ratio"], "(a) Application"), (axb, "After acceptance", ["Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due"], "(b) After acceptance")):
        ymax = 0
        for v in names:
            q = F[(F.stage == stage) & (F.variable == v)].sort_values("r"); assert len(q) == 20
            col, mk, ls = VC[v]
            ax.fill_between(q.r, 100 * q.lower, 100 * q.upper, color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(q.r, 100 * q.flip_rate, color=col, marker=mk, ls=ls, ms=2.8, lw=1.1, mew=0, zorder=3)
            ymax = max(ymax, 100 * q.upper.max())
            # ledger gate at r = 5, 10, 15, 20
            for r in (5, 10, 15, 20):
                x = q[q.r == r].iloc[0]
                m = L6[(L6.stage == ("selection" if stage == "Application" else "monitoring")) & (L6.variable == v) & (L6.level == f"r={r}") & L6.quantity.str.startswith("Flip rate at r=")]
                assert len(m) == 1, (stage, v, r)
                pv, lo, hi = pnum(m.display.iloc[0]); dec = 2 if stage == "After acceptance" else 1
                for nm, val, ref in (("rate", 100 * x.flip_rate, pv), ("lower bound", 100 * x.lower, lo), ("upper bound", 100 * x.upper, hi)):
                    check(NAME, f"{stage} {v} r={r}: plotted {nm}", val, ref, dec, "ledger " + m.ledger_id.iloc[0])
        for xv in (5, 10):
            ax.axvline(xv, ymax=0.86, color="0.3", lw=0.8, ls=(0, (1, 2)), zorder=1); ax.text(xv, 0.865, "$r$ = %d" % xv, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, clip_on=False, bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)
        ax.set_xlim(0.5, 20.5); ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlabel("Cost ratio $r$"); ax.grid(axis="y", color="0.92", lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_ylim(0, np.ceil(ymax / 5.0) * 5.0 * 1.28); ax.set_ylabel("Flip rate (%)")
        ax.text(0.015, 0.98, lab, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
    # direct labels at the right end of each curve
    for ax, stage, names in ((axa, "Application", ["Age", "Debt Ratio"]), (axb, "After acceptance", ["Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due"])):
        ends = []
        for v in names:
            q = F[(F.stage == stage) & (F.variable == v) & (F.r == 20)].iloc[0]; ends.append((100 * q.flip_rate, v))
        ends.sort(); ylo, yhi = ax.get_ylim(); min_sep = (yhi - ylo) * 0.095; ys_ = []
        for y_, v in ends:
            ys_.append(y_ if not ys_ else max(y_, ys_[-1] + min_sep))
        for (y_, v), yl in zip(ends, ys_):
            col, mk, ls = VC[v]
            ax.text(22.4, yl, v, ha="left", va="center", fontsize=7, color="black", clip_on=False)
            ax.annotate("", xy=(20.6, y_), xycoords="data", xytext=(22.2, yl), textcoords="data", annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=0.6, color=col, shrinkA=0, shrinkB=0))
    NOTES[NAME] = ["New figure: flip rate (percentage of test rows whose flag changes when the variable is marginalised out) against the cost ratio r = 1 to 20 at the BN threshold 1/(1+r), with shaded 95% bootstrap intervals (2,000 resamples, seed 42, the resample sequence of Tables 3 and 5). (a) Application: Age and Debt Ratio. (b) After acceptance: the four Markov-blanket variables (iteration-79 BN). Dotted lines mark r = 5 and r = 10. Colours and marker/line styles identify variables here, not models.",
                   "Suggested caption: 'Flip rate against the cost ratio r at the flag threshold 1/(1+r): the share of test rows whose flag changes when the variable is marginalised out of the evidence, at application (a) and after acceptance (b). Bands are 95% bootstrap intervals. At application a flag can change only for r above 7.4, because no cell of the lookup table reaches 1/(1+r) before; after acceptance flips occur at every r.'",
                   "Values at r = 5, 10, 15, 20 are asserted against ledger rows N-ADD-235 to 258 (J10); the full r = 1..20 series is in analysis_additions/flip_rate_by_r.csv."]
    EDITS.append("Fig 6 (new, fig6_flip_rate_vs_r): flip rate against r at both stages; Tables 3 and 5 can be reported at 1/(1+r), and the After-acceptance numbers in Table 5, Section 4.2.2 and Section 3.6 (last sentence) change from the locked 0.17 to 1/(1+r) (J10). Section 4.3.2: reword 'the after-acceptance threshold was chosen on the test set' to refer to the EM iteration.")
    return fig

if __name__ == "__main__":
    os.makedirs(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out"), exist_ok=True); f = make(); print(audit(f, NAME)); f.savefig(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out", "f6.png"), dpi=200)
    print([c for c in CHECKS if not c["ok"]], len(CHECKS))
