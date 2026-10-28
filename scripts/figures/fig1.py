"""Fig 1a/1b: learned structures. Fixed layout, labels beside nodes, blanket derived from the edge list."""
import json, pickle, itertools, math
from common import *
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

ORANGE = "#E69F00"; GOLD = "#F0E442"; NODEFILL = "#DCE9F2"; INDFILL = "#BFBFBF"
RAD = 0.075; SQ = 0.075

def load_app():
    E = [tuple(e) for e in json.load(open(R + "/selection/tables/sel_dag_9node_edges.json"))["edges"]]
    return E
def load_after():
    d = pickle.load(open(R + "/full_variable/models/bn_final_model_iteration79.pkl", "rb"))
    return [tuple(e) for e in d["model_edges"]]

def blanket(E, t="Default"):
    par = {a for a, b in E if b == t}; ch = {b for a, b in E if a == t}
    cop = {a for a, b in E if b in ch and a != t}
    return par, ch, cop, par | ch | cop

def clean(n): return n.replace(" (bin)", "")
FULL = {"M_Income": "M_Income", "M_Dependents": "M_Dependents"}
SHORT = {"Monthly Income": "Income", "Number of Dependents": "Dependents", "Credit Utilisation Rate": "Utilisation Rate",
         "30-59 Days Past Due": "30-59 Days", "60-89 Days Past Due": "60-89 Days", "90+ Days Late": "90+ Days",
         "M_Income": "Income indicator", "M_Dependents": "Dependents indicator"}

def _label_pos(x, y, side, ind):
    g = 0.07 if ind else 0.05; r = SQ if ind else RAD
    if side == "r": return x + r + g, y, "left", "center"
    if side == "l": return x - r - g, y, "right", "center"
    if side == "a": return x, y + r + g, "center", "bottom"
    if side == "b": return x, y - r - g, "center", "top"
    if side == "ar": return x + r * 0.7, y + r + g, "left", "bottom"
    if side == "al": return x - r * 0.7, y + r + g, "right", "bottom"
    if side == "br": return x + r * 0.7, y - r - g, "left", "top"
    if side == "bl": return x - r * 0.7, y - r - g, "right", "top"
    raise ValueError(side)

def seg_pt_dist(p, a, b):
    p, a, b = map(np.array, (p, a, b)); ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1); return float(np.linalg.norm(p - (a + t * ab)))

def draw(name, E, pos, sides, H, legend_xy, short=False, curves=None):
    curves = curves or {}
    names = {n: clean(n) for n in pos}
    par, ch, cop, bl = blanket(E)
    blanket_nodes = {clean(n) for n in bl}
    E2 = [(clean(a), clean(b)) for a, b in E]
    assert set(pos) == {clean(n) for n in set(itertools.chain(*E))}, "layout nodes differ from edge-list nodes"
    ind = {n for n in pos if n.startswith("M_")}
    fig = plt.figure(figsize=(W2, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W2); ax.set_ylim(0, H); ax.axis("off")
    node_art = {}
    for n, (x, y) in pos.items():
        if n in ind:
            p = Rectangle((x - SQ, y - SQ), 2 * SQ, 2 * SQ, fc=INDFILL, ec="black", lw=0.6, zorder=3)
        elif n == "Default":
            p = Circle((x, y), RAD, fc=GOLD, ec="black", lw=0.9, zorder=3)
        elif n in blanket_nodes:
            p = Circle((x, y), RAD, fc=NODEFILL, ec=ORANGE, lw=1.8, zorder=3)
        else:
            p = Circle((x, y), RAD, fc=NODEFILL, ec="black", lw=0.6, zorder=3)
        p.set_gid("avoid"); p._name = n; ax.add_patch(p); node_art[n] = p
    drawn = set()
    for a, b in E2:
        dashed = a in ind
        rA = (SQ if a in ind else RAD) * 72 + 0.5; rB = (SQ if b in ind else RAD) * 72 + 1.0
        ar = FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=7, shrinkA=rA, shrinkB=rB, connectionstyle="arc3,rad=%g" % curves.get((a, b), 0.0),
                             lw=0.6, ls=("--" if dashed else "-"), color="0.35", zorder=2, joinstyle="miter")
        ax.add_patch(ar); drawn.add((a, b))
    assert drawn == set(E2) and len(E2) == len(set(E2)), "drawn edge set differs from edge list"
    nm = (lambda n: SHORT.get(n, n)) if short else (lambda n: n)
    for n, (x, y) in pos.items():
        side = sides[n]; dx = dy = 0
        if isinstance(side, tuple): side, dx, dy = side
        lx, ly, ha, va = _label_pos(x, y, side, n in ind)
        t = ax.text(lx + dx, ly + dy, nm(n), fontsize=8, ha=ha, va=va, color="black", zorder=4)
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    # legend inside the figure
    hs = [Line2D([], [], marker="o", ls="", mfc=GOLD, mec="black", mew=0.9, ms=7, label="Default"),
          Line2D([], [], marker="o", ls="", mfc=NODEFILL, mec=ORANGE, mew=1.8, ms=7, label="Markov blanket of Default"),
          Line2D([], [], marker="o", ls="", mfc=NODEFILL, mec="black", mew=0.6, ms=7, label="Other variable"),
          Line2D([], [], marker="s", ls="", mfc=INDFILL, mec="black", mew=0.6, ms=7, label="Missingness indicator (dashed edge)")]
    for h in hs: h.set_label(h.get_label())
    lg = ax.legend(handles=hs, loc="upper left", bbox_to_anchor=legend_xy, bbox_transform=ax.transData, frameon=True, fontsize=8,
                   handletextpad=0.4, borderpad=0.4, labelspacing=0.35, fancybox=False, edgecolor="0.6")
    lg.get_frame().set_linewidth(0.5)
    # structural checks: edge crossing a non-endpoint node
    thru = []
    for a, b in E2:
        for n, p in pos.items():
            if n in (a, b): continue
            if seg_pt_dist(p, pos[a], pos[b]) < RAD + 0.03: thru.append((a, b, n))
    return fig, dict(blanket=sorted(blanket_nodes), parents=sorted(clean(x) for x in par), children=sorted(clean(x) for x in ch),
                     coparents=sorted(clean(x) for x in cop), edges=len(E2), nodes=len(pos), edge_through_node=thru)


# ---------------------------------------------------------------- deterministic label-side search
SIDES = ["r", "l", "a", "b", "ar", "al", "br", "bl"]

def _text_wh(txt, fig):
    r = fig.canvas.get_renderer()
    w, h, d = r.get_text_width_height_descent(txt, mpl.font_manager.FontProperties(family="sans-serif", size=8), False)
    return w / fig.dpi, (h + d) / fig.dpi * 1.0

def _seg_rect(a, b, rc):
    x0, y0, x1, y1 = rc
    n = 600
    for t in np.linspace(0, 1, n):
        x = a[0] + t * (b[0] - a[0]); y = a[1] + t * (b[1] - a[1])
        if x0 <= x <= x1 and y0 <= y <= y1: return True
    return False

def _rect_inter(p, q, pad=0.0):
    return p[0] - pad < q[2] and q[0] - pad < p[2] and p[1] - pad < q[3] and q[1] - pad < p[3]

def auto_sides(E, pos, H, legend_rect, short=False, forced=None):
    E2 = [(clean(a), clean(b)) for a, b in E]; ind = {n for n in pos if n.startswith("M_")}
    nm = (lambda n: SHORT.get(n, n)) if short else (lambda n: n)
    fig = plt.figure(figsize=(W2, H)); fig.canvas.draw()
    wh = {n: _text_wh(nm(n), fig) for n in pos}; plt.close(fig)
    def rect(n, side):
        x, y = pos[n]; lx, ly, ha, va = _label_pos(x, y, side, n in ind); w, h = wh[n]
        x0 = lx if ha == "left" else (lx - w if ha == "right" else lx - w / 2)
        y0 = ly - h / 2 if va == "center" else (ly if va == "bottom" else ly - h)
        return (x0, y0, x0 + w, y0 + h)
    node_rect = {n: (pos[n][0] - RAD - 0.02, pos[n][1] - RAD - 0.02, pos[n][0] + RAD + 0.02, pos[n][1] + RAD + 0.02) for n in pos}
    def cost(n, side, cur):
        rc = rect(n, side); c = 0.0
        if rc[0] < 0.02 or rc[1] < 0.02 or rc[2] > W2 - 0.02 or rc[3] > H - 0.02: c += 1000
        if _rect_inter(rc, legend_rect, 0.03): c += 1000
        for m, nr in node_rect.items():
            if _rect_inter(rc, nr): c += 1000
        for m, s in cur.items():
            if m != n and _rect_inter(rc, rect(m, s), 0.03): c += 1000
        for a, b in E2:
            if _seg_rect(pos[a], pos[b], rc): c += 10
        c += SIDES.index(side) * 0.1
        return c
    cur = {n: "r" for n in pos}
    if forced: cur.update(forced)
    order = sorted(pos, key=lambda n: (-sum(n in e for e in E2), n))
    for _ in range(6):
        changed = False
        for n in order:
            if forced and n in forced: continue
            best = min(SIDES, key=lambda s: cost(n, s, cur))
            if best != cur[n]: cur[n] = best; changed = True
        if not changed: break
    hits = sum(cost(n, cur[n], cur) >= 10 and cost(n, cur[n], cur) < 1000 for n in pos)
    total = sum(int(cost(n, cur[n], cur) // 10) for n in pos)
    return cur, dict(edge_label_hits=total)
