"""Fig 2: mean S_exp of the blanket variables, Application (a) and After acceptance (b), one image."""
import os
from common import *
NAME = "fig2_sexp_application_and_after_acceptance"
APP = [("Age", "N-SEL-016"), ("Debt Ratio", "N-SEL-017")]
AFT = [("Credit Utilisation Rate", "N-MON-016"), ("30-59 Days Past Due", "N-MON-018"), ("90+ Days Late", "N-MON-020"), ("60-89 Days Past Due", "N-MON-022")]
ZERO_APP = ["N-SEL-033", "N-SEL-034", "N-SEL-035", "N-SEL-036"]
ZERO_AFT = ["N-MON-025", "N-MON-026", "N-MON-027", "N-MON-028", "N-MON-051", "N-MON-052", "N-MON-053", "N-MON-054"]

def build():
    sa = pd.read_csv(R + "/selection/tables/sel_rq2_sensitivity_table.csv").set_index("variable")
    sm = pd.read_csv(R + "/full_variable/tables/rq2_sensitivity_table.csv"); sm = sm[sm.measure == "S_exp"]; sm["v"] = sm.variable.str.replace(" (bin)", "", regex=False); sm = sm.set_index("v")
    # omitted variables: ledger zero rows and counts
    for ids, n, tab, nm in ((ZERO_APP, 4, "Application", None), (ZERO_AFT, 8, "After acceptance", None)):
        assert len(ids) == n
        for i in ids:
            ok = lv(i) == 0.0 and lci(i) == (0.0, 0.0) and "outside Default's blanket" in LED.loc[i, "quantity"]
            CHECKS.append(dict(figure=NAME, item=f"{tab}: omitted variable {LED.loc[i,'quantity'].split('), ')[-1] if False else LED.loc[i,'quantity'].split(', ',1)[1]}", plotted="omitted", reference="mean S_exp = 0, CI [0,0] (" + i + ")", decimals="", source="ledger " + i, ok=bool(ok)))
            assert ok, "omitted variable with non-zero S_exp: " + i
    omit_app = [v for v in sa.index if not sa.loc[v, "in_blanket"]]
    omit_aft = [v for v in sm.index if not sm.loc[v, "in_blanket"]]
    assert len(omit_app) == 4 and len(omit_aft) == 8, (omit_app, omit_aft)
    assert {"M_Income", "M_Dependents", "Age", "Debt Ratio"} <= set(omit_aft)
    assert all(sa.loc[v, "mean_S_exp"] == 0 for v in omit_app) and all(sm.loc[v, "mean"] == 0 for v in omit_aft)
    for tab, om, ids in (("Application", omit_app, ZERO_APP), ("After acceptance", omit_aft, ZERO_AFT)):
        for v in om:
            assert any(v.replace(" (bin)", "") in LED.loc[i, "quantity"] for i in ids), "omitted variable not in ledger zero rows: " + v
        CHECKS.append(dict(figure=NAME, item=f"{tab}: number of omitted variables", plotted=len(om), reference=len(ids), decimals="", source="table in_blanket==False vs ledger zero rows", ok=len(om) == len(ids)))
    rows = {"a": [], "b": []}
    for v, lid in APP:
        assert sa.loc[v, "in_blanket"]
        rows["a"].append((v, sa.loc[v, "mean_S_exp"], sa.loc[v, "ci_lo"], sa.loc[v, "ci_hi"], lid))
    for v, lid in AFT:
        assert sm.loc[v, "in_blanket"]
        rows["b"].append((v, sm.loc[v, "mean"], sm.loc[v, "ci_low"], sm.loc[v, "ci_high"], lid))
    return rows, omit_aft

def make():
    rows, omit_aft = build()
    P = 0.235; HEAD = 0.17; BAND = {"a": 0.15, "b": 0.28}; GAP = 0.10; TOP = 0.03; BOT = 0.42
    hts = {k: HEAD + len(rows[k]) * P + BAND[k] for k in "ab"}
    H = TOP + hts["a"] + GAP + hts["b"] + BOT
    fig = plt.figure(figsize=(W1, H))
    L, Rr = 0.385, 0.955
    ybot = {"b": BOT, "a": BOT + hts["b"] + GAP}
    axs = {}
    for k in "ba":
        axs[k] = fig.add_axes([L, ybot[k] / H, Rr - L, hts[k] / H], sharex=axs.get("b"))
    notes = {"a": "4 other variables: " + SEXP + " = 0 on every row",
             "b": "8 other variables, including Age and Debt Ratio:\n" + SEXP + " = 0 on every row"}
    for key in "ab":
        ax = axs[key]; data = rows[key]; n = len(data)
        y = np.arange(n)[::-1] + 1.0
        for yy, (v, m, lo, hi, lid) in zip(y, data):
            ax.barh(yy, m, height=0.66, color=COL["BN"], alpha=0.40, ec=COL["BN"], lw=0.7, zorder=2)
            ax.errorbar([m], [yy], xerr=[[m - lo], [hi - m]], fmt="none", ecolor="black", elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=3)
            lab = f"{m:.4f}"
            ax.text(0.0004, yy, lab, ha="left", va="center", fontsize=7, zorder=4)
            check(NAME, f"{key} {v}: mean", m, lv(lid), 4, "ledger " + lid + " vs table")
            c0, c1 = lci(lid); check(NAME, f"{key} {v}: CI low", lo, c0, 4, "ledger " + lid); check(NAME, f"{key} {v}: CI high", hi, c1, 4, "ledger " + lid)
            check_text(NAME, f"{key} {v}: bar label", lab, f"{rnd(lv(lid),4):.4f}", "ledger " + lid)
        ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data]); ax.tick_params(axis="y", length=0)
        ymax = n + 0.5 + HEAD / P; ymin = 0.5 - BAND[key] / P
        ax.set_ylim(ymin, ymax); ax.set_xlim(0, 0.03)
        ax.spines["left"].set_bounds(0.5, n + 0.5)
        ax.text(0.0004, ymax, "(%s) %s" % (key, "Application" if key == "a" else "After acceptance"), ha="left", va="top", fontsize=8, fontweight="bold", zorder=4)
        fig.text(0.02, (ybot[key] + 0.02) / H, notes[key], ha="left", va="bottom", fontsize=7, linespacing=1.15)
        ax.grid(axis="x", color="0.9", lw=0.5, zorder=0); ax.set_axisbelow(True)
    plt.setp(axs["a"].get_xticklabels(), visible=False)
    axs["b"].set_xlabel("Mean " + SEXP)
    axs["b"].set_xticks([0, 0.01, 0.02, 0.03]); axs["b"].set_xticklabels(["0", "0.01", "0.02", "0.03"])
    NOTES[NAME] = ["No in-figure title or footnote was removed from the old figures; they were replaced by this single two-panel image.",
                   "Bars are the stored mean S_exp. Whiskers are the stored bootstrap 95% intervals (same CI columns as the ledger); at this scale they are only about 0.0003 wide.",
                   "Only variables inside Default's Markov blanket are drawn. Application: 4 variables outside the blanket (Monthly Income, Number of Dependents, Real Estate Loans, Open Credit Lines) have S_exp = 0. After acceptance: 8 variables outside the blanket (Age, Debt Ratio, Monthly Income, Number of Dependents, Real Estate Loans, Open Credit Lines, M_Income, M_Dependents) have S_exp = 0. Both are asserted against ledger rows N-SEL-033 to 036 and N-MON-025 to 028, 051 to 054.",
                   "Both panels share the x axis from 0 to 0.03."]
    return fig

if __name__ == "__main__":
    os.makedirs(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out"), exist_ok=True); f = make(); print(audit(f, NAME)); f.savefig(os.path.join(os.environ["CREDIT_BN_ROOT"], "figures_out", "f2.png"), dpi=250)
    print([c for c in CHECKS if not c["ok"]])
