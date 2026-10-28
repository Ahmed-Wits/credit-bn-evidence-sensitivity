"""Fig S8: After-acceptance counterpart of Fig 5: paired BN minus LR / RF change in AUC loss under masking, 95% test-row bootstrap intervals (J8)."""
import os
from common import *
from matplotlib.lines import Line2D
NAME = "figS8_masking_paired_differences_after_acceptance"
ADD = WR + "/analysis_additions"

def pnum(disp):
    d = disp.replace("−", "-").replace("[", "").replace("]", "").replace(",", ""); return [float(x) for x in d.split()]

def make():
    d = pd.read_csv(ADD + "/mon_rq3_row_bootstrap_summary.csv"); assert len(d) == 48
    sm = pd.read_csv(R + "/monitoring_rq3/mon_rq3_summary_v2.csv"); sm["level"] = sm["level"].round(2)
    v5 = pd.read_csv(ADD + "/ledger_additions_v5.csv", dtype=str, keep_default_na=False)
    vs = ["Credit Utilisation Rate", "30-59 Days Past Due", "90+ Days Late", "60-89 Days Past Due"]
    H = 96 * MM; fig = plt.figure(figsize=(W2, H))
    Lm, Rm, Tm, Bm, gx, gy = 0.92, 0.10, 0.08, 0.90, 0.30, 0.50
    aw = (W2 - Lm - Rm - gx) / 2; ah = (H - Tm - Bm - gy) / 2
    series = [("LR", "mcar", -1.8), ("LR", "dependent", -0.6), ("RF", "mcar", 0.6), ("RF", "dependent", 1.8)]
    axs = []
    for i, v in enumerate(vs):
        rr, cc = divmod(i, 2)
        ax = fig.add_axes([(Lm + cc * (aw + gx)) / W2, (Bm + (1 - rr) * (ah + gy)) / H, aw / W2, ah / H]); axs.append(ax)
        for comp, mech, off in series:
            x, y, lo, hi = [], [], [], []
            for lev in (0.2, 0.4, 0.5):
                r = d[(d.variable == v) & (d.mechanism == mech) & (d.level == lev) & (d.contrast == f"BN minus {comp}")]; assert len(r) == 1; r = r.iloc[0]
                ref = float(sm[(sm.variable == v) & (sm.level == lev) & (sm.mechanism == mech) & (sm.variant == "BN")].mean_delta_auc.iloc[0]) - float(sm[(sm.variable == v) & (sm.level == lev) & (sm.mechanism == mech) & (sm.variant == comp + "_primary")].mean_delta_auc.iloc[0])
                check(NAME, f"{v} BN-{comp} {mech} {int(lev*100)}%: point estimate equals the stored paired difference (mon_rq3_summary_v2.csv)", r.point_estimate, ref, 6, "mon_rq3_summary_v2.csv")
                m = v5[(v5.variable == v) & (v5.mechanism == mech) & (v5.level == "%d%%" % round(lev * 100)) & (v5.comparator == comp)]; assert len(m) == 1
                pv, l_, h_ = pnum(m.display.iloc[0])
                for nm, val, rf in (("estimate", r.point_estimate, pv), ("lower bound", r.ci_lo, l_), ("upper bound", r.ci_hi, h_)):
                    check(NAME, f"{v} BN-{comp} {mech} {int(lev*100)}%: plotted {nm}", val, rf, 4, "ledger " + m.ledger_id.iloc[0])
                x.append(lev * 100 + off); y.append(r.point_estimate); lo.append(r.point_estimate - r.ci_lo); hi.append(r.ci_hi - r.point_estimate)
            filled = mech == "mcar"
            ax.errorbar(x, y, yerr=[lo, hi], color=COL[comp], ls=("-" if filled else "--"), lw=0.9, marker=MK[comp], ms=3.6, mfc=(COL[comp] if filled else "white"), mec=COL[comp], mew=0.9, elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=3)
        ax.axhline(0, color="black", lw=0.7, zorder=1)
        ax.set_xlim(15, 55); ax.set_xticks([20, 40, 50]); ax.set_xticklabels(["20%", "40%", "50%"]); ax.set_ylim(-0.0023, 0.0074)
        ax.set_yticks([0, 0.002, 0.004, 0.006]); ax.set_yticklabels(["0.000", "0.002", "0.004", "0.006"])
        ax.text(0.02, 0.97, "(%s) %s" % ("abcd"[i], v), transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
        ax.grid(axis="y", color="0.92", lw=0.5, zorder=0); ax.set_axisbelow(True)
        if cc == 0: ax.set_ylabel("AUC loss,\nBN minus comparator")
        else: ax.tick_params(labelleft=False)
        if rr == 1: ax.set_xlabel("Share of test rows with the variable masked")
    hs = [Line2D([], [], color=COL[c], ls=("-" if m == "mcar" else "--"), marker=MK[c], ms=3.6, lw=0.9, mfc=(COL[c] if m == "mcar" else "white"), mec=COL[c], mew=0.9, label=f"BN minus {c}, {'MCAR' if m == 'mcar' else 'dependent'}") for c, m, _ in series]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=2.6, borderaxespad=0.15)
    NOTES[NAME] = ["New figure: After-acceptance counterpart of Fig 5 (J8). Points are the paired differences in AUC loss, BN minus LR and BN minus RF (positive = the BN degrades less), averaged over 10 mask seeds, with 95% intervals from the paired test-row bootstrap (2,000 resamples, seed 42; models fixed; primary refit-imputed imputer only). Filled markers and solid lines = MCAR masking, open markers and dashed lines = dependent masking; blue/square = LR, orange/triangle = RF. Points are dodged horizontally within each masking level. One panel per Markov-blanket variable; all panels share the vertical axis.",
                   "Suggested caption: 'Paired difference in AUC loss when one blanket variable was masked after acceptance, BN minus LR and BN minus RF, under MCAR and dependent masking, with 95% test-row bootstrap intervals. Positive values mean that the BN degraded less. Intervals reflect test-row sampling and averaging over 10 mask seeds; the models were fixed.'",
                   "Result to quote (J8): 16 of the 48 intervals exclude 0, all positive; 32 include 0. Values are asserted against the stored paired differences (mon_rq3_summary_v2.csv) and ledger rows N-ADD-187 to 234."]
    EDITS.append("Fig S8 (new, figS8_masking_paired_differences_after_acceptance): After-acceptance paired differences with intervals. Supplement Section S4.1 and article Section 4.3.3 currently say that no intervals were computed after acceptance and that Fig S4 'describes size only'; with J8 (16 of 48 intervals exclude 0, all positive; strongest for Credit Utilisation Rate and for dependent masking against LR) these sentences and the Table S-Numbers entries need updating.")
    return fig

if __name__ == "__main__":
    os.makedirs(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out"), exist_ok=True); f = make(); print(audit(f, NAME)); f.savefig(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out", "fS8.png"), dpi=200)
    print([c for c in CHECKS if not c["ok"]], len(CHECKS))
