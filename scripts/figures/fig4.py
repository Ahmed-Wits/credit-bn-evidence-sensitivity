"""Fig 4: cost saving against the cheaper trivial policy, r = 1..20, both stages."""
import os
from common import *
from matplotlib.lines import Line2D
NAME = "fig4_cost_saving"
IDS = {  # (stage, model, r) -> ledger id
    ("selection", "BN", 5): "N-SEL-042", ("selection", "LR", 5): "N-SEL-043", ("selection", "RF", 5): "N-SEL-044",
    ("monitoring", "BN", 5): "N-MON-045", ("monitoring", "LR", 5): "N-MON-046", ("monitoring", "RF", 5): "N-MON-047"}
for st, base in (("selection", 1), ("monitoring", 10)):
    for k, r in enumerate((10, 15, 20)):
        for m, model in enumerate(("BN", "LR", "RF")):
            IDS[(st, model, r)] = "N-XST-%03d" % ((0 if st == "selection" else 9) + 3 * k + m + 1)
PEAK = {("selection", "BN"): "N-SEL-061", ("selection", "LR"): "N-SEL-062", ("selection", "RF"): "N-SEL-063",
        ("monitoring", "BN"): "N-MON-061", ("monitoring", "LR"): "N-MON-062", ("monitoring", "RF"): "N-MON-063"}
RSTAR = {"selection": "N-XST-020", "monitoring": "N-XST-022"}

def check_pct(item, frac, lid):
    disp = ldisp(lid); assert disp.endswith("%"), disp
    dec = ndec(disp); plotted = round(100 * frac + 1e-12, dec); ref = float(disp[:-1])
    CHECKS.append(dict(figure=NAME, item=item, plotted=plotted, reference=ref, decimals=dec, source="ledger " + lid, ok=abs(plotted - ref) < 1e-9))

def check_pct_named(name, item, frac, lid):
    disp = ldisp(lid); dec = ndec(disp); plotted = round(100 * frac + 1e-12, dec); ref = float(disp[:-1])
    CHECKS.append(dict(figure=name, item=item, plotted=plotted, reference=ref, decimals=dec, source="ledger " + lid, ok=abs(plotted - ref) < 1e-9))

def make():
    d = pd.read_csv(R + "/selection/tables/sel_decision_value_combined.csv")
    assert sorted(d.r.unique()) == list(range(1, 21)) and len(d) == 120
    H = 60 * MM
    fig = plt.figure(figsize=(W2, H))
    Lm, Rm, Tm, Bm = 0.50, 0.12, 0.27, 0.66; gap = 0.62
    aw = (W2 - Lm - Rm - gap) / 2; ah = H - Tm - Bm
    axs = [fig.add_axes([(Lm + k * (aw + gap)) / W2, Bm / H, aw / W2, ah / H]) for k in range(2)]
    for ax, st, lab, ymax in zip(axs, ("selection", "monitoring"), ("(a)\nApplication", "(b) After\nacceptance"), (32, 74)):
        s = d[d.contribution == st]
        n0 = s.reject_all.iloc[0]; n1 = s.accept_all.iloc[0] / s.r.iloc[0]
        assert (s.reject_all == n0).all() and np.allclose(s.accept_all / s.r, n1)
        rstar = n0 / n1
        check(NAME, f"{st}: r* = N0/N1", rstar, lv(RSTAR[st]), 2, "ledger " + RSTAR[st])
        for model in ("LR", "RF", "BN"):
            m = s[s.model == model].sort_values("r")
            ax.plot(m.r, 100 * m.saving, color=COL[model], ls=LS[model], marker=MK[model], ms=(3.6 if model == "BN" else 2.8),
                    lw=(2.2 if model == "BN" else 0.9), zorder=(4 if model == "BN" else 3), label=model, mew=0)
            for r in (5, 10, 15, 20):
                v = float(m[m.r == r].saving.iloc[0]); check_pct(f"{st} {model} r={r}", v, IDS[(st, model, r)])
            pk = m.loc[m.saving.idxmax()]; assert int(pk.r) == 14, (st, model, pk.r)
            check_pct(f"{st} {model} peak (r=14)", float(pk.saving), PEAK[(st, model)])
        for xv, t in ((5, "$r$ = 5"), (10, "$r$ = 10"), (rstar, "$r^{*}$ = %.2f" % rstar)):
            ax.axvline(xv, color="0.3", lw=0.8, ls=(0, (1, 2)), zorder=1)
            ax.text(xv, 1.0, t, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, clip_on=False)
        ax.set_xlim(0.5, 20.5); ax.set_ylim(0, ymax); ax.set_xticks([1, 5, 10, 15, 20]); ax.set_yticks(range(0, 31, 5) if ymax < 40 else range(0, 61, 10))
        ax.set_xlabel("Cost ratio $r$"); ax.set_ylabel("Saving vs cheaper\ntrivial policy (%)")
        ax.text(0.99, 0.98, lab, transform=ax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold", linespacing=1.1)
        ax.grid(axis="y", color="0.92", lw=0.5, zorder=0); ax.set_axisbelow(True)
        if st == "selection": app_rstar = rstar
    hs = [Line2D([], [], color=COL[m], ls=LS[m], marker=MK[m], ms=(3.6 if m == "BN" else 2.8), lw=(2.2 if m == "BN" else 0.9), label=m, mew=0) for m in ("BN", "LR", "RF")]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=2.5, handlelength=3.0, borderaxespad=0.15)
    NOTES[NAME] = ["Removed in-figure texts (old sel_decision_value_combined_v8.png): title 'Cost saving across cost ratios: saving vs the cheaper of flag-none and flag-all (test set)'; panel titles 'Selection (application stage)' and 'Monitoring (full variable set)' (now '(a) Application', '(b) After acceptance' inside the panels).",
                   "Removed footnote (move to caption): 'Saving = 1 - cost / cost of the cheaper trivial policy at that r. BN threshold = 1/(1+r), fixed a priori; LR and RF thresholds are the cost-minimising value on out-of-fold TRAIN predictions at each r (0.01 grid, ties to the highest). Dotted lines mark the pre-specified r = 5, 10, 15, 20. Monitoring uses the 29,998 test rows where the BN is defined, so LR/RF costs differ slightly from the locked 30,000-row figures. Descriptive only.'",
                   "In the new figure the dotted lines mark r = 5, r = 10 and the exact switch ratio r* = N0/N1 (13.96 at Application, 13.98 after acceptance); r = 15 and r = 20 are no longer marked. The caption should say so.",
                   "Y label is the requested text, wrapped over two lines ('Saving vs cheaper' / 'trivial policy (%)') so that it fits the panel height at 8 pt."]
    EDITS.append("Fig 4 caption: the dotted vertical lines now mark r = 5, r = 10 and r* = N0/N1 (13.96 Application, 13.98 After acceptance), not r = 5, 10, 15, 20. Include the saving definition from the old footnote if wanted.")
    return fig

if __name__ == "__main__":
    os.makedirs(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out"), exist_ok=True); f = make(); print(audit(f, NAME)); f.savefig(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out", "f4.png"), dpi=250)
    print([c for c in CHECKS if not c["ok"]], len(CHECKS))
