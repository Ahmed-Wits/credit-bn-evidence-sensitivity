"""Shared style, ledger checks, audits and saving for the final dissertation figures.
Plots only from saved outputs. No model is refitted."""
import os, io, re, json, pickle
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.transforms import Bbox
from matplotlib.legend import Legend

R = os.environ["CREDIT_BN_ROOT"]
WR = R + "/Submissions/4. Article/WRITEUP"
OUT = WR + "/figures_final"
LEDP = WR + "/ledgers/numbers_ledger_v3.csv"
MM = 1 / 25.4
W1, W2 = 83.7 * MM, 173.6 * MM          # single column, full width (inches)
COL = {"BN": "#009E73", "LR": "#0072B2", "RF": "#E69F00"}   # Okabe-Ito
MK = {"BN": "o", "LR": "s", "RF": "^"}
LS = {"BN": "-", "LR": "--", "RF": ":"}
SEXP = r"$S_{\mathrm{exp}}$"
MINFONT = 7.0

def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.default": "it",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": True, "figure.dpi": 100})
style()

LED = pd.read_csv(LEDP, dtype=str, keep_default_na=False).set_index("ledger_id")
CHECKS = []      # one dict per assertion
NOTES = {}       # figure -> list of removed titles / footnotes / remarks
EDITS = []       # article edits needed
AUDITS = {}      # figure -> audit dict

def rnd(x, d): return round(float(x) + 0.0, d)

def check(fig, item, plotted, ref, dec, source):
    ok = abs(rnd(plotted, dec) - rnd(ref, dec)) < 1e-9
    CHECKS.append(dict(figure=fig, item=item, plotted=rnd(plotted, dec), reference=rnd(ref, dec), decimals=dec, source=source, ok=bool(ok)))
    return ok

def check_text(fig, item, plotted, ref, source):
    ok = (plotted == ref)
    CHECKS.append(dict(figure=fig, item=item, plotted=plotted, reference=ref, decimals="", source=source, ok=bool(ok)))
    return ok

def lv(lid): return float(LED.loc[lid, "value"])
def lci(lid): return float(LED.loc[lid, "ci_lo"]), float(LED.loc[lid, "ci_hi"])
def ldisp(lid): return LED.loc[lid, "display"]

def ndec(s):
    m = re.search(r"\.(\d+)", s); return len(m.group(1)) if m else 0

# ---------------------------------------------------------------- audits
def _tick_texts(ax):
    out = []
    for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = min(lim), max(lim)
        locs = axis.get_majorticklocs()
        labs = axis.get_majorticklabels()
        for loc, t in zip(locs, labs):
            if lo - 1e-12 <= loc <= hi + 1e-12 and t.get_visible() and t.get_text().strip():
                out.append(t)
    return out

def _all_ticklabel_ids(fig):
    ids = set()
    for ax in fig.axes:
        for t in ax.xaxis.get_majorticklabels() + ax.yaxis.get_majorticklabels() + ax.xaxis.get_minorticklabels() + ax.yaxis.get_minorticklabels():
            ids.add(id(t))
    return ids

def visible_texts(fig):
    tl = _all_ticklabel_ids(fig)
    ts = [t for t in fig.findobj(mtext.Text) if id(t) not in tl and t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:
        if ax.get_visible() and ax.axison: ts += _tick_texts(ax)
    return ts

def _inter(a, b, tol=0.6):
    return (a.x0 + tol < b.x1 and b.x0 + tol < a.x1 and a.y0 + tol < b.y1 and b.y0 + tol < a.y1)

def avoid_boxes(fig):
    """Display-space boxes of markers (Line2D with marker) and artists tagged gid='avoid'."""
    r = fig.canvas.get_renderer(); boxes = []
    leg_desc = set()
    for lg in fig.findobj(Legend):
        leg_desc |= {id(o) for o in lg.findobj()}
    for ln in fig.findobj(mpl.lines.Line2D):
        if id(ln) in leg_desc or not ln.get_visible(): continue
        mk = ln.get_marker()
        if mk in (None, "None", "", " ") or ln.get_markersize() <= 0: continue
        xy = ln.get_xydata()
        if len(xy) == 0 or ln.axes is None: continue
        pts = ln.get_transform().transform(xy)
        h = ln.get_markersize() * fig.dpi / 72 / 2 + ln.get_markeredgewidth() * fig.dpi / 72 / 2
        for (x, y) in pts:
            boxes.append((ln.get_label() or "marker", Bbox([[x - h, y - h], [x + h, y + h]])))
    for a in fig.findobj(lambda o: o.get_gid() == "avoid"):
        boxes.append(("node:" + str(getattr(a, "_name", "")), a.get_window_extent(r)))
    return boxes

import re as _re
BANNED = _re.compile(r"hidden|hide|hiding|retract|rank", _re.I)

def audit(fig, name, extra_legend_block=True):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    texts = visible_texts(fig)
    info = dict(n_text=len(texts)); bad = []
    for t in texts:
        if t.get_fontsize() < MINFONT - 1e-9: bad.append((t.get_text()[:25], t.get_fontsize()))
    info["min_font"] = min(t.get_fontsize() for t in texts); info["under7"] = bad
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    out = [t.get_text()[:25] for t, b in boxes if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5]
    info["outside"] = out
    tt = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _inter(boxes[i][1], boxes[j][1]): tt.append((boxes[i][0].get_text()[:20], boxes[j][0].get_text()[:20]))
    info["text_text"] = tt
    av = avoid_boxes(fig); tm = []
    for t, b in boxes:
        for n, ab in av:
            if _inter(b, ab, tol=0.3): tm.append((t.get_text()[:20], n))
    info["text_marker"] = tm
    lb = []
    for lg in fig.findobj(Legend):
        if not lg.get_visible(): continue
        fb = lg.get_frame().get_window_extent(r)
        own = {id(o) for o in lg.findobj()}
        for t, b in boxes:
            if id(t) not in own and _inter(fb, b): lb.append(("legend", t.get_text()[:20]))
        for n, ab in av:
            if _inter(fb, ab): lb.append(("legend", n))
    info["legend_block"] = lb
    # lines (data lines, axvline/axhline, not gridlines, not legend handles) crossing a text box
    leg_ids = set()
    for lg in fig.findobj(Legend): leg_ids |= {id(o) for o in lg.findobj()}
    grid_ids = set()
    for ax_ in fig.axes:
        for ga in (ax_.xaxis, ax_.yaxis):
            grid_ids |= {id(l) for t_ in ga.get_major_ticks() for l in (t_.gridline, t_.tick1line, t_.tick2line)}
            grid_ids |= {id(l) for t_ in ga.get_minor_ticks() for l in (t_.gridline, t_.tick1line, t_.tick2line)}
    lt = []
    for ln in fig.findobj(mpl.lines.Line2D):
        if id(ln) in leg_ids or id(ln) in grid_ids or not ln.get_visible() or ln.axes is None: continue
        xy = ln.get_xydata()
        if len(xy) < 2 or ln.get_linestyle() in ("None", "none", "", " "): continue
        pts = ln.get_transform().transform(xy)
        for t, b in boxes:
            hit = False
            for p0, p1 in zip(pts[:-1], pts[1:]):
                for u in np.linspace(0, 1, 60):
                    x_, y_ = p0 + u * (p1 - p0)
                    if b.x0 + 0.5 < x_ < b.x1 - 0.5 and b.y0 + 0.5 < y_ < b.y1 - 0.5: hit = True; break
                if hit: break
            if hit: lt.append((t.get_text()[:25], str(ln.get_label())[:20]))
    info["line_text"] = lt
    info["banned_words"] = [t.get_text()[:30] for t in texts if BANNED.search(t.get_text())]
    info["titles"] = [ax.get_title() for ax in fig.axes if ax.get_title()] + ([fig._suptitle.get_text()] if fig._suptitle else [])
    w, h = fig.get_size_inches(); info["size_mm"] = (round(w * 25.4, 2), round(h * 25.4, 2))
    info["pass"] = not (bad or out or tt or tm or lb or lt or info["titles"] or info["banned_words"])
    AUDITS[name] = info
    return info

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=600)
    fig.savefig(f"{OUT}/{name}_preview.png", dpi=200)
    from PIL import Image
    w, h = fig.get_size_inches(); im = Image.open(f"{OUT}/{name}.png")
    AUDITS[name]["png_px"] = im.size; AUDITS[name]["png_px_expected"] = (round(w * 600), round(h * 600))
    plt.close(fig)
