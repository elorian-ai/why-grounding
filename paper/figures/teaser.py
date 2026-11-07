"""Figure 1 (teaser): six scenes, red rings = objects the words fit, blue dot = object the coordinate names.
CountQA image 424 and SpatialMQA idx 94 ($GR_DATA/tvg/spatialmqa/meta.jsonl) -> $GR_RESULTS/paper/teaser.{png,pdf}.
    python paper/figures/teaser.py
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from common import first_where, save, setup_sans
from matplotlib.patches import Circle, Polygon, Rectangle

from grounded_reasoning import DATA
from grounded_reasoning.benchmarks import load_items

_, BOLD = setup_sans(base_size=7)
INK, INK2 = "#2B2B2B", "#8A8A8A"
MONO = "DejaVu Sans Mono"  # reference lines (phrases + coordinates)
FRAME = "#4D4D4D"
TXT = "#D1495B"  # text reference (muted red)
GND = "#2E6FB0"  # coordinate (muted blue)
SHAPE_FC, SHAPE_EC = "#C7CCD4", "#9AA1AC"

W_IN, H_IN = 5.5, 1.82
S = 0.8  # scene side (in)
R = 0.06  # object radius (in)
LM = 0.03
GAP, DIVGAP = 0.1, 0.22  # gap between scenes; gap between the two groups
Y_SCENE = 0.56  # scene bottom (in)
Y_GND = 0.1  # coordinate line (in)


def ngon(n, rot=90, r=1.0):
    a = np.deg2rad(rot + 360 * np.arange(n) / n)
    return np.c_[np.cos(a), np.sin(a)] * r


def star(n=5):
    v = np.empty((2 * n, 2))
    v[0::2] = ngon(n)
    v[1::2] = ngon(n, 90 + 180 / n, 0.45)
    return v


# shapes in unit radius; object layouts in scene fraction (x right, y up); SCATTER 0, 1, 3 = the top-left cluster
# fmt: off
BLOB = np.array([[-0.2, 0.95], [0.35, 0.55], [1, 0.75], [0.8, -0.1], [0.95, -0.7], [0.05, -0.45], [-0.55, -1],
                 [-0.8, -0.2], [-1, 0.45]])
CROSS = np.array([[-0.33, 1], [0.33, 1], [0.33, 0.33], [1, 0.33], [1, -0.33], [0.33, -0.33],
                  [0.33, -1], [-0.33, -1], [-0.33, -0.33], [-1, -0.33], [-1, 0.33], [-0.33, 0.33]]) * 0.9
NAMED = [("circle", None), ("square", ngon(4, 45, 1.05)), ("triangle", ngon(3, 90, 1.1)), ("star", star()),
         ("diamond", ngon(4, 90, 1.1)), ("cross", CROSS), ("hexagon", ngon(6, 0, 1.0))]
SCATTER = np.array([[0.15, 0.86], [0.42, 0.88], [0.82, 0.80], [0.28, 0.59], [0.55, 0.47], [0.68, 0.63],
                    [0.12, 0.31], [0.86, 0.24], [0.32, 0.22], [0.60, 0.20], [0.87, 0.50]])
# fmt: on
TGT = 3  # target; objects 0 and 1 are its look-alikes in the top-left cluster
# wire scene (illustrative): y(t), t in [0, 1] left->right; wire 0 starts at the TOP but is the LOWER wire at t = .5.
WIRE_X0, WIRE_X1 = 0.12, 0.88
WIRES = [
    lambda t: 0.5 + 0.28 * np.cos(2.2 * np.pi * t) + 0.03 * np.sin(5 * np.pi * t),
    lambda t: 0.5 - 0.28 * np.cos(2.2 * np.pi * t + 0.3) + 0.03 * np.sin(4 * np.pi * t + 1),
]
wire_xy = lambda k, t: (WIRE_X0 + (WIRE_X1 - WIRE_X0) * t, WIRES[k](t))
WIRE_TGT = wire_xy(0, 0.5)  # coordinate = the middle of the top wire
WIRE_HITS = [wire_xy(0, 0.5), wire_xy(1, 0.5)]  # the words fit the mid-point of either wire
GRID = np.array([[0.2 + 0.3 * c, 0.8 - 0.3 * r] for r in range(3) for c in range(3)])  # row-major, row 1 at the top
G_TGT = 1 * 3 + 2  # row 2, column 3
NAMED_XY = np.array([[0.2, 0.78], [0.55, 0.82], [0.84, 0.62], [0.5, 0.45], [0.18, 0.35], [0.78, 0.2], [0.38, 0.14]])
N_TGT = [n for n, _ in NAMED].index("star")

meta = first_where(DATA / "tvg/spatialmqa/meta.jsonl", lambda r: r["idx"] == 94)
CAT = next(o for o in meta["objects"] if o["phrase"] == "cat")["points"][0]
CQA_IMAGE = ["benchmarks/countqa/images/00424.png"]  # its first question
CQA = next(item for item in load_items("countqa") if item["images"] == CQA_IMAGE)
# CountQA has no point labels, so the three top-left tomato centres (0-1000, y down) are hand-annotated.
TOM_TGT, TOM_NEAR = (325, 480), [(205, 335), (440, 265)]  # "the top left tomato" fits all three


def square_crop(path, focus_x, top=None):
    """-> (crop, to_scene): largest square crop, horizontally centred near focus_x (0-1000) or vertically from `top` (0-1000);
    to_scene maps a full-image 0-1000 point to crop scene-fraction (x right, y up)."""
    im = mpimg.imread(path)
    h, w = im.shape[:2]
    side = min(h, w)
    x0 = int(np.clip(focus_x / 1000 * w - side / 2, 0, w - side))
    y0 = (h - side) // 2 if top is None else int(top / 1000 * h)

    def to_scene(p):
        return (p[0] / 1000 * w - x0) / side, 1 - (p[1] / 1000 * h - y0) / side

    return im[y0 : y0 + side, x0 : x0 + side], to_scene


def coord(p):  # scene fraction -> Qwen 0-1000 [x, y], y down
    return f"[{round(p[0] * 1000)}, {round((1 - p[1]) * 1000)}]"


fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W_IN), ylim=(0, H_IN))
ax.set_axis_off()


def ring(p, dashed, r=R * 1.6):
    ax.add_patch(Circle(p, r, fc="none", ec=TXT, lw=1.0, ls=(0, (2.2, 1.4)) if dashed else "-", zorder=5))


def dot(p):
    ax.add_patch(Circle(p, 0.02, fc=GND, ec="white", lw=0.5, zorder=6))


def scene(x0, objs, tgt, text_hits):
    """objs = [(verts_unit or None for a circle, (fx, fy), letter)]."""
    ax.add_patch(Rectangle((x0, Y_SCENE), S, S, fc="white", ec=FRAME, lw=0.5, zorder=2))
    P = []
    for verts, (fx, fy), letter in objs:
        cx, cy = x0 + fx * S, Y_SCENE + fy * S
        P.append((cx, cy))
        shape = Circle((cx, cy), R * 0.92) if verts is None else Polygon(verts * R + [cx, cy], closed=True)
        shape.set(fc=SHAPE_FC, ec=SHAPE_EC, lw=0.45, zorder=3)
        ax.add_patch(shape)
        if letter:
            label = {"fontsize": 5.2, "fontweight": "bold", "family": BOLD, "color": INK2, "zorder": 4}
            ax.text(cx, cy - R * 1.02, letter, **label, ha="center", va="top")  # just below its shape
    for i in text_hits:  # labelled scene: the ring takes in the shape and its letter
        if objs[i][2]:
            ring((P[i][0], P[i][1] - R * 0.5), dashed=False, r=R * 1.95)
        else:
            ring(P[i], dashed=len(text_hits) > 1)
    dot(P[tgt])


def photo_scene(x0, crop, to_scene, tgt, hits, r):
    ax.imshow(crop, extent=(x0, x0 + S, Y_SCENE, Y_SCENE + S), interpolation="lanczos", zorder=2)
    ax.add_patch(Rectangle((x0, Y_SCENE), S, S, fc="none", ec=FRAME, lw=0.5, zorder=3))
    at = lambda p: (x0 + to_scene(p)[0] * S, Y_SCENE + to_scene(p)[1] * S)
    for h in hits:  # white halo under the red ring so it reads on any photo
        ax.add_patch(Circle(at(h), r, fc="none", ec="white", lw=2.0, zorder=4))
        ring(at(h), dashed=len(hits) > 1, r=r)
    dot(at(tgt))


def wire_scene(x0):
    ax.add_patch(Rectangle((x0, Y_SCENE), S, S, fc="white", ec=FRAME, lw=0.5, zorder=2))
    t = np.linspace(0, 1, 300)
    at = lambda fx, fy: (x0 + fx * S, Y_SCENE + fy * S)
    for k in range(len(WIRES)):
        fx, fy = wire_xy(k, t)
        X, Y = x0 + fx * S, Y_SCENE + fy * S
        z = 3.1 - 0.1 * k  # outline + fill per wire -> crossings read as over/under; wire 0 (top start) on top
        ax.plot(X, Y, color=SHAPE_EC, lw=2.3, solid_capstyle="round", zorder=z)
        ax.plot(X, Y, color=SHAPE_FC, lw=1.4, solid_capstyle="round", zorder=z + 0.01)
        for te in (0, 1):  # terminals
            ax.add_patch(Circle(at(*wire_xy(k, te)), R * 0.55, fc=SHAPE_FC, ec=SHAPE_EC, lw=0.45, zorder=3.5))
    for h in WIRE_HITS:
        ring(at(*h), dashed=True, r=R * 1.3)
    dot(at(*WIRE_TGT))


def refs(x0, words, xy):
    # one monospace font + size for both lines; phrases on fixed baselines across scenes
    lines = words.split("\n")
    yc = Y_SCENE - 0.205  # every phrase block is centred on the same height
    for k, line in enumerate(lines):
        y = yc + ((len(lines) - 1) / 2 - k) * 0.11
        ax.text(x0 + S / 2, y, line, fontsize=6.4, color=TXT, ha="center", va="baseline", family=MONO)
    ax.text(x0 + S / 2, Y_GND, xy, fontsize=6.4, color=GND, ha="center", va="center", family=MONO)


xs = [LM, LM + S + GAP] + [LM + 2 * S + GAP + DIVGAP + k * (S + GAP) for k in range(4)]
synthetic = [  # (x slot, objects, target, text hits, words, target point)
    (2, [(BLOB, p, "abcdefghijk"[i]) for i, p in enumerate(SCATTER)], TGT, [TGT], "“shape d”", SCATTER[TGT]),
    (3, [(BLOB, p, None) for p in GRID], G_TGT, [G_TGT], "“row 2,\ncolumn 3”", GRID[G_TGT]),
    (4, [(v, p, None) for (_, v), p in zip(NAMED, NAMED_XY)], N_TGT, [N_TGT], "“the star”", NAMED_XY[N_TGT]),
]
for k, objs, tgt, hits, words, p in synthetic:
    scene(xs[k], objs, tgt, hits)
    refs(xs[k], words, coord(p))

crop, to_scene = square_crop(DATA / CQA["images"][0], 500, top=120)  # CountQA tomatoes (portrait: keep the top 3/4)
photo_scene(xs[0], crop, to_scene, TOM_TGT, [TOM_TGT] + TOM_NEAR, r=0.1)
refs(xs[0], "“the top left\ntomato”", f"[{TOM_TGT[0]}, {TOM_TGT[1]}]")
wire_scene(xs[1])
refs(xs[1], "“the middle of\nthe top wire”", coord(WIRE_TGT))
crop, to_scene = square_crop(DATA / meta["image"], CAT[0] + 140)  # SpatialMQA cat; coordinate = GT point
photo_scene(xs[5], crop, to_scene, CAT, [CAT], r=0.12)
refs(xs[5], "“the cat”", f"[{CAT[0]:.0f}, {CAT[1]:.0f}]")
ax.set(xlim=(0, W_IN), ylim=(0, H_IN))
ax.set_aspect("auto")  # imshow resets limits/aspect

xd = (xs[1] + S + xs[2]) / 2  # thin vertical divider, centred between the two groups
ax.plot([xd, xd], [0.02, H_IN - 0.04], color="black", lw=0.5, solid_capstyle="butt", zorder=1)

yh = H_IN - 0.05  # group headers, each centred on its group
GROUPS = [
    ((xs[0], xs[1] + S), "Requires undescribable reference", r"$\gg$"),
    ((xs[2], xs[5] + S), "Textual reference available", r"$\approx$"),
]
# shared header size: largest <= 9pt at which every title fits inside its group's span (both headers stay the same size)
rend = fig.canvas.get_renderer()
fs_head = 9.0
for (a, b), title, _ in GROUPS:
    tt = ax.text(0, 0, title, fontsize=9, fontweight="bold", family=BOLD)
    fs_head = min(fs_head, 9 * (b - a) / (tt.get_window_extent(rend).width / fig.dpi))
    tt.remove()
for (a, b), title, rel in GROUPS:
    ax.text((a + b) / 2, yh, title, fontsize=fs_head, fontweight="bold", family=BOLD, ha="center", va="top", color=INK)
    xm, yr = (a + b) / 2, yh - 0.19  # "Grounding <rel> Text", each word in its reference colour
    parts = [("Grounding", GND), (f" {rel} ", INK), ("Text", TXT)]
    ws = []
    for t, _ in parts:  # measure each piece (inches) so the whole line is centred on the group
        tt = ax.text(0, 0, t, fontsize=8.2)
        ws.append(tt.get_window_extent(rend).width / fig.dpi)
        tt.remove()
    x = xm - sum(ws) / 2
    for (t, c), w in zip(parts, ws):
        ax.text(x, yr, t, fontsize=8.2, ha="left", va="top", color=c)
        x += w
save(fig, "teaser")
