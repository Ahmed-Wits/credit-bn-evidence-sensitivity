"""S7: EM convergence, After acceptance. Accepts the rerun (code/s7_em_history.csv) only if the test-LL peak is at iteration 79
and train and test LL at 79 are each within 0.001 of the stored values; otherwise the original figure is copied unmodified."""
import shutil
from common import *
from figsB import SRC, STATS
NAME = "figS7_em_convergence"
TOL = 0.001
STORED_TEST, STORED_TRAIN = -329111.347037, -1316746.532478   # as stated in the task and in the supplement
ORIG = R + "/full_variable/figures/em_convergence_final_with_zoom.png"

def decide(hist_csv):
    h = pd.read_csv(hist_csv)
    d = pickle.load(open(R + "/full_variable/models/bn_final_model_iteration79.pkl", "rb"))
    pk = int(h.test_ll.idxmax()) + 1
    dtr = float(h.train_ll[78] - d["train_ll_at_peak"]); dte = float(h.test_ll[78] - d["test_ll_at_peak"])
    dtr_t = float(h.train_ll[78] - STORED_TRAIN); dte_t = float(h.test_ll[78] - STORED_TEST)
    ok = (len(h) == 100) and pk == 79 and abs(dtr) <= TOL and abs(dte) <= TOL and abs(dtr_t) <= TOL and abs(dte_t) <= TOL
    STATS["S7"] = dict(peak=pk, n_iter=len(h), train79=float(h.train_ll[78]), test79=float(h.test_ll[78]), d_train_pkl=dtr, d_test_pkl=dte, d_train_stated=dtr_t, d_test_stated=dte_t,
                       test100_minus_peak=float(h.test_ll[99] - h.test_ll.max()), accepted=bool(ok), stored_train=float(d["train_ll_at_peak"]), stored_test=float(d["test_ll_at_peak"]))
    cj = json.load(open(hist_csv.replace(".csv", "_cpd79.json")))["max_abs_cpd_diff_at_79"]
    STATS["S7"]["extra"] = "All 13 CPDs of the rerun at iteration 79 equal the stored iteration-79 CPDs to within %.1e (largest absolute difference over all CPT entries), so the rerun reproduces the stored model, not only its log-likelihoods." % max(cj.values())
    return h, ok

def make(hist_csv):
    h, ok = decide(hist_csv)
    S = STATS["S7"]
    for item, p, r in (("peak iteration of the test log-likelihood", S["peak"], 79), ("number of iterations", S["n_iter"], 100)):
        CHECKS.append(dict(figure=NAME, item=item, plotted=p, reference=r, decimals="", source="rerun vs spec", ok=(p == r)))
    for item, p, r in (("train LL at iteration 79 vs stored (pickle)", S["train79"], S["stored_train"]), ("test LL at iteration 79 vs stored (pickle)", S["test79"], S["stored_test"]),
                       ("train LL at iteration 79 vs stated -1,316,746.532478", S["train79"], STORED_TRAIN), ("test LL at iteration 79 vs stated -329,111.347037", S["test79"], STORED_TEST)):
        CHECKS.append(dict(figure=NAME, item=item + " (tolerance 0.001)", plotted=round(p, 6), reference=round(r, 6), decimals=6, source="s7_em_history.csv", ok=abs(p - r) <= TOL))
    if not ok: return None, h
    H = 72 * MM; fig = plt.figure(figsize=(W2, H))
    Lm, Tm, Bm = 0.95, 0.10, 0.50; gap_v = 0.18; gap_h = 0.95; Rm = 0.12
    wl = 2.95; wr = W2 - Lm - Rm - wl - gap_h; ah = (H - Tm - Bm - gap_v) / 2
    axt = fig.add_axes([Lm / W2, (Bm + ah + gap_v) / H, wl / W2, ah / H])
    axe = fig.add_axes([Lm / W2, Bm / H, wl / W2, ah / H], sharex=axt)
    axz = fig.add_axes([(Lm + wl + gap_h) / W2, Bm / H, wr / W2, (2 * ah + gap_v) / H])
    it = h.iteration.values
    for ax, col, lab in ((axt, "train_ll", "Train\nlog-likelihood"), (axe, "test_ll", "Test\nlog-likelihood")):
        ax.plot(it, h[col], color=COL["BN"], lw=1.3, marker="o", ms=2.2, mew=0, markevery=5)
        ax.axvline(79, color="0.3", lw=0.8, ls="--"); ax.set_ylabel(lab); ax.grid(color="0.93", lw=0.5); ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}".replace("-", "−")))
    axt.text(0.03, 0.05, "(a) Train", transform=axt.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
    axe.text(0.03, 0.05, "(b) Test", transform=axe.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
    plt.setp(axt.get_xticklabels(), visible=False)
    axe.set_xlabel("EM iteration"); axe.set_xlim(0, 101)
    z = h[(h.iteration >= 60) & (h.iteration <= 100)]
    axz.plot(z.iteration, z.test_ll, color=COL["BN"], lw=1.3, marker="o", ms=2.8, mew=0)
    axz.axvline(79, color="0.3", lw=0.8, ls="--")
    pk_y = float(h.test_ll[78]); axz.plot([79], [pk_y], marker="*", ms=8, color="black", mew=0, zorder=5)
    axz.text(83.5, pk_y + 0.11, "Peak, iteration 79", ha="left", va="center", fontsize=7)
    axz.annotate("", xy=(79.4, pk_y + 0.015), xytext=(83.3, pk_y + 0.095), arrowprops=dict(arrowstyle="-", lw=0.6, color="0.2", shrinkA=0, shrinkB=2))
    axz.set_ylim(float(z.test_ll.min()) - 0.03, pk_y + 0.42)
    axz.set_xlim(59, 101); axz.set_xlabel("EM iteration"); axz.set_ylabel("Test log-likelihood")
    axz.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v:,.1f}".replace("-", "−")))
    axz.text(0.03, 0.97, "(c) Test,\niterations\n60 to 100", transform=axz.transAxes, ha="left", va="top", fontsize=8, fontweight="bold", linespacing=1.1)
    axz.grid(color="0.93", lw=0.5); axz.set_axisbelow(True)
    yl = axz.get_ylim(); STATS["S7"]["zoom_axis_span"] = float(yl[1] - yl[0]); STATS["S7"]["zoom_axis_lo"] = float(yl[0]); STATS["S7"]["zoom_axis_hi"] = float(yl[1])
    STATS["S7"]["zoom_data_span"] = float(z.test_ll.max() - z.test_ll.min())
    NOTES[NAME] = ["Old figure (em_convergence_final_with_zoom.png) is replaced by a deterministic rerun (code/s7_em_rerun.py, history in code/s7_em_history.csv): same structure, maximum-likelihood start on the complete training rows, no prior, no early stop, 100 iterations. Iteration 79 reproduces the stored log-likelihoods (train and test) to within 0.001; the actual differences are in verify_report.md.",
                   "Panel (c) spans only %.2f log-likelihood units on its vertical axis (from %.2f to %.2f; the plotted curve itself spans %.2f units) out of a test log-likelihood of about 329,111, so the curvature in panel (c) is about one part in 300,000 of the log-likelihood and the iteration-79 peak is a very small effect." % (STATS["S7"]["zoom_axis_span"], STATS["S7"]["zoom_axis_lo"], STATS["S7"]["zoom_axis_hi"], STATS["S7"]["zoom_data_span"]),
                   "The dashed line marks iteration 79, the iteration used. The stored per-iteration history of the original run was lost, so this rerun, not the original run, is what the figure shows."]
    SRC[NAME] = "code/s7_em_history.csv (rerun with shared/src/em.py and loglik.py); stored peak values in full_variable/models/bn_final_model_iteration79.pkl"
    return fig, h
