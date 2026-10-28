"""J11b: which CPDs differ between EM iteration 79 and iteration 100 (explains why the BN's predictions do not depend on the iteration)."""
from addcommon import *
import pickle
p = Part("j11b")
K = pickle.load(open(ADD + "/intermediate/em_iter79_iter100_params.pkl", "rb"))
rows = []
for c in K[79]:
    a = K[79][c]; b = K[100][c]
    assert a["variables"] == b["variables"]
    d = float(np.max(np.abs(a["values"] - b["values"]))); rows.append((c, a["variables"][1:], d))
rows.sort(key=lambda x: x[2])
same = [r for r in rows if r[2] <= 1e-12]; diff = [r for r in rows if r[2] > 1e-12]
BL = ["Default", "Credit Utilisation Rate (bin)", "90+ Days Late (bin)", "60-89 Days Past Due (bin)", "30-59 Days Past Due (bin)"]
p.check("the CPDs of Default and of the other three blanket variables and of Credit Utilisation Rate are identical at iteration 79 and 100 (max abs diff <= 1e-12)", all(r[2] <= 1e-12 for r in rows if r[0] in BL), "; ".join(f"{r[0]}: {r[2]:.1e}" for r in rows if r[0] in BL))
for c, par, d in rows:
    p.row("monitoring", f"Largest absolute difference between the EM iteration-100 and iteration-79 CPD of {c.replace(' (bin)', '')} (any entry)", d, f"{d:.2e}", "probability", "Submissions/4. Article/WRITEUP/analysis_additions/intermediate/em_iter79_iter100_params.pkl" if False else "Submissions/4. Article/WRITEUP/analysis_additions/j11_iteration_comparison.csv",
          f"max |CPD(100) - CPD(79)| | node={c} | parents={', '.join(par) if par else 'none'}", derived="Y", notes="the CPDs come from the deterministic EM rerun (j11_em.py); a CPD whose node and parents are always observed is the plain maximum-likelihood count estimate and cannot change with the iteration", display_rule="sci", n=119999, model="BN", variable=c.replace(" (bin)", ""), caveat="C6")
md = ["## J11 (supplement): which CPDs change between iteration 79 and iteration 100", "",
      "Largest absolute change of any CPT entry between the iteration-79 and iteration-100 parameters, per node:", "", "| Node | Parents | max abs difference |", "|---|---|---|"] + [f"| {c} | {', '.join(par) if par else 'none'} | {d:.2e} |" for c, par, d in rows] + ["", checks_md(p), "",
      "**Reading.** Only the CPDs of nodes whose own value or parents are partly unobserved (Monthly Income and Number of Dependents, which are missing in about 20% and 3% of training rows, and the nodes that depend on them) are re-estimated by EM from soft counts and change between iterations; a CPD of an always-observed node with always-observed parents is the plain count estimate and is identical at every iteration. The BN's posterior for Default and the sensitivities of the four blanket variables depend only on the CPDs near Default, so every quantity in J11 is identical for iterations 79 and 100 (to rounding, about 1e-16); the choice of iteration therefore has no effect on any reported result, although it affects the test log-likelihood of the whole joint model (J1: 0.64 units below the peak at iteration 100).", ""]
p.write("\n".join(md))
for c, par, d in rows: print("%-34s %.2e" % (c, d))
