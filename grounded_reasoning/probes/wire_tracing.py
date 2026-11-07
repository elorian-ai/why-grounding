"""Wire tracing: 8 identical wires join lettered left terminals to numbered right terminals; which terminal does the
queried wire reach? Data: $GR_DATA/probes/wire_tracing/, results: $GR_RESULTS/probes/wire_tracing/.

  python -m grounded_reasoning.probes.wire_tracing generate | sft | summarize
  python -m grounded_reasoning.probes.wire_tracing eval --format F [--labeled]
"""

import argparse
import functools
import json
import math
import os
import random
import re
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from grounded_reasoning import DATA
from grounded_reasoning.probes.common import (
    FORMATS,
    checkpoint_dir,
    data_dir,
    results_dir,
    run_name,
    sft_path,
    summarize,
)

TASK = "wire_tracing"
OUT = data_dir(TASK)

# ---------------------------------------------------------------- generate
W, H = 960, 640
SS = 2  # supersample factor, downscaled at the end for antialiasing
STUB = 28.0  # horizontal lead-in/out at each terminal
N_SAMPLES = 640  # shared x grid
N_COLS = 5  # interior control columns
N_WIRES = 8
SIGMA = 74.0  # per-segment sd of the Brownian-bridge excursion (px)
CTRL_INSET = 24.0  # control points stay this far inside the band, absorbing spline overshoot

LEFT_X, RIGHT_X = 0.115 * W, 0.885 * W
TOP_Y, BOT_Y = 0.115 * H, 0.885 * H
MARGIN_Y = 0.069 * H
_INNER = np.linspace(LEFT_X + STUB, RIGHT_X - STUB, N_COLS + 2)[1:-1]  # (N_COLS,)
COLS = np.concatenate([[LEFT_X, LEFT_X + STUB], _INNER, [RIGHT_X - STUB, RIGHT_X]])  # (N_COLS+4,)
XS = np.linspace(LEFT_X, RIGHT_X, N_SAMPLES)  # (N_SAMPLES,) shared by EVERY wire
_SEG = np.clip(np.searchsorted(COLS, XS, side="right") - 1, 0, len(COLS) - 2)
_U = (XS - COLS[_SEG]) / (COLS[_SEG + 1] - COLS[_SEG])

# legibility gates (px at 1x); a wire draw violating any of them is redrawn
MIN_ANGLE = 22.0  # deg between tangents at a crossing
MIN_CROSS_SEP = 40.0  # between any two crossing points
TERM_CLEAR = 32.0  # crossing must be this far from every terminal
MIN_GAP = 14.0  # min wire-to-wire distance away from their own crossings
CLEAR_PX = 18.0  # x-radius around a crossing that is exempt from MIN_GAP
MAX_SLOPE = 2.4  # |dy/dx|
TERM_SEP = 50.0  # between terminal centres in one column

# appearance: every wire identical, so nothing but the path identifies it
BG = TERM_FILL = (255, 255, 255)
WIRE = (58, 60, 66)
CORE_W = 4  # wire core width (1x)
R_BREAK = 6.0  # over/under break: a background disc of this radius punched into the lower wire
BREAK_CLEAR = R_BREAK + CORE_W / 2 + 3.0  # a crossing must be this clear of every uninvolved wire
TERM_EDGE = (34, 34, 38)
TERM_R = 15  # terminal box half-size
LEFT_LABELS = "PQRSTUVW"  # not a-h: those are the MCQ option letters
assert MIN_CROSS_SEP >= 6 * R_BREAK

# numbered beads (labeled twin): ids are the 2-digit pool minus the right-terminal labels, assigned in shuffled
# order; an unplaceable bead is dropped, never the scene
MARK_SPACING = 110.0  # bead spacing in 0-1000 x-units
MARK_R = 4.0  # bead radius (1x px)
MARK_FONT = 12  # label font size (1x)
MARK_GAP = 6.0  # min distance label box <-> any wire that is not its own
MARK_OWN_MARGIN = 3.0  # label box must sit closer to its own bead than to any other wire by this
MARK_CROSS_CLEAR = 14.0  # bead centre min distance from any crossing
MARK_DROPS = Counter()  # why beads were dropped, reported in the labeled meta.json

LABEL_POOL = list(range(11, 90))  # 2-digit only => no option text is a substring of another
XBANDS = {"X0": (1, 3), "X1": (4, 6), "X2": (8, 11)}  # a derangement forces >= 1 crossing
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen"]  # fmt: skip
MAX_DRAWS = 400000  # per-cell cap on scene draws
SPLITS = (("train", 1667, 20260729), ("heldout", 167, 77001))  # (split, scenes per crossing band, seed)

FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


@functools.cache
def _font(px):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    raise SystemExit("no TrueType font found; add a path to FONT_PATHS (e.g. any DejaVuSans-Bold.ttf)")


def sample_wire(y_ctrl):
    """Catmull-Rom through y_ctrl (N_COLS+4,) evaluated on XS -> (N_SAMPLES,) wire y; horizontal at both stubs."""
    yy = np.concatenate([[y_ctrl[0]], y_ctrl, [y_ctrl[-1]]])  # (N_COLS+6,)
    p0, p1, p2, p3 = yy[_SEG], yy[_SEG + 1], yy[_SEG + 2], yy[_SEG + 3]
    u = _U
    u2, u3 = u * u, u * u * u
    return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


def bridge_ctrl(y0, y1, rng):
    """Control y's (N_COLS+4,) for one wire: a Brownian bridge from terminal y0 to y1."""
    n = N_COLS + 1  # segments between the two stub columns
    inc = np.array([rng.gauss(0, SIGMA) for _ in range(n)])
    walk = np.concatenate([[0.0], np.cumsum(inc)])  # (n+1,)
    b = walk - np.linspace(0, 1, n + 1) * walk[-1]  # b[0] == b[-1] == 0
    y = np.clip(np.linspace(y0, y1, n + 1) + b, MARGIN_Y + CTRL_INSET, H - MARGIN_Y - CTRL_INSET)
    y[0], y[-1] = y0, y1
    return np.concatenate([[y0], y, [y1]])


def crossings(ys_a, ys_b):
    """Exact crossings of two wires on the shared grid -> list of (x, y, angle_deg)."""
    d = ys_a - ys_b
    k = np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    out = []
    for i in k:
        t = d[i] / (d[i] - d[i + 1])
        x = XS[i] + t * (XS[i + 1] - XS[i])
        y = ys_a[i] + t * (ys_a[i + 1] - ys_a[i])
        lo, hi = max(0, i - 3), min(N_SAMPLES - 1, i + 4)  # slope over a small window
        dx = XS[hi] - XS[lo]
        ang_a = math.atan2(ys_a[hi] - ys_a[lo], dx)
        ang_b = math.atan2(ys_b[hi] - ys_b[lo], dx)
        out.append((float(x), float(y), abs(math.degrees(ang_a - ang_b))))
    return out


def _dist_to_wire(px, py, ys_m, halfwin=40.0):
    """Min distance from point (px, py) to wire ys_m (N_SAMPLES,), searched in an x-window around px."""
    lo, hi = np.searchsorted(XS, [px - halfwin, px + halfwin])
    lo, hi = max(0, lo - 1), min(N_SAMPLES, hi + 1)
    if hi <= lo:
        return float("inf")
    return float(np.min(np.hypot(XS[lo:hi] - px, ys_m[lo:hi] - py)))


def _perp_min_dist(ys_a, ys_b, xcross):
    """Min perpendicular distance between two wires, excluding CLEAR_PX x-windows around xcross."""
    keep = np.ones(N_SAMPLES, dtype=bool)
    for xc in xcross:
        keep &= np.abs(XS - xc) > CLEAR_PX
    if not keep.any():
        return float("inf")
    m = 0.5 * (np.gradient(ys_a, XS) + np.gradient(ys_b, XS))
    return float(np.min(np.abs(ys_a - ys_b)[keep] / np.sqrt(1 + m[keep] ** 2)))


def slots(rng, n):
    """n sorted terminal y positions in [TOP_Y, BOT_Y], jittered but >= TERM_SEP apart."""
    span = BOT_Y - TOP_Y
    base = np.linspace(TOP_Y, BOT_Y, n)
    jit = span / (2.6 * n)
    for _ in range(200):
        y = np.sort(np.array([b + rng.uniform(-jit, jit) for b in base]))
        if np.all(np.diff(y) >= TERM_SEP):
            return y
    return base


def build_scene(rng, tries=200):
    """One candidate scene, or None. Wire i joins left slot i to right slot perm[i] (a derangement); wires are
    placed one at a time and a wire failing a gate is redrawn on its own."""
    ly = slots(rng, N_WIRES)
    ry = slots(rng, N_WIRES)
    perm = list(range(N_WIRES))
    for _ in range(400):
        rng.shuffle(perm)
        if all(p != i for i, p in enumerate(perm)):
            break
    else:
        return None

    terms = [(LEFT_X, float(v)) for v in ly] + [(RIGHT_X, float(v)) for v in ry]
    ys, cx, pair_cross = [], [], {}
    cmat = [[0] * N_WIRES for _ in range(N_WIRES)]

    for i in range(N_WIRES):
        for _ in range(tries):
            w = sample_wire(bridge_ctrl(ly[i], ry[perm[i]], rng))
            if w.min() < MARGIN_Y or w.max() > H - MARGIN_Y:
                continue
            if np.max(np.abs(np.gradient(w, XS))) > MAX_SLOPE:
                continue
            new, ok = {}, True
            for j in range(i):
                cs = crossings(w, ys[j])
                if any(a < MIN_ANGLE for _, _, a in cs):
                    ok = False
                    break
                new[j] = cs
            if not ok:
                continue
            npts = [(x, y) for cs in new.values() for x, y, _ in cs]
            if any(math.dist(p, q) < MIN_CROSS_SEP for a, p in enumerate(npts) for q in npts[a + 1 :] + cx):
                continue
            if any(math.dist(p, t) < TERM_CLEAR for p in npts for t in terms):
                continue
            if any(_perp_min_dist(w, ys[j], [x for x, _, _ in cs]) < MIN_GAP for j, cs in new.items()):
                continue
            # break discs must not bite an uninvolved wire, in both directions
            if any(
                _dist_to_wire(x, y, ys[m]) < BREAK_CLEAR
                for j, cs in new.items()
                for x, y, _ in cs
                for m in range(i)
                if m != j
            ):
                continue
            if any(_dist_to_wire(x, y, w) < BREAK_CLEAR for x, y in cx):
                continue
            for j, cs in new.items():
                pair_cross[(j, i)] = cs
                cmat[i][j] = cmat[j][i] = len(cs)
            cx.extend(npts)
            ys.append(w)
            break
        else:
            return None

    labels = rng.sample(LABEL_POOL, N_WIRES)  # shuffled w.r.t. height
    return {
        "ly": ly,
        "ry": ry,
        "perm": perm,
        "ys": ys,
        "cmat": cmat,
        "pair_cross": pair_cross,
        "labels": labels,
        "cross_pts": cx,
        "min_angle": min([a for cs in pair_cross.values() for _, _, a in cs], default=90.0),
        "min_cross_sep": min(
            [math.dist(cx[a], cx[b]) for a in range(len(cx)) for b in range(a + 1, len(cx))], default=999.0
        ),
    }


def shortcut_picks(sc, wid):
    """Answer of each non-tracing heuristic for wire `wid`: nearest_y (right terminal closest in y to the start),
    departure (closest to the extrapolated post-stub tangent), identity (same slot index)."""
    y0 = sc["ly"][wid]
    ry = sc["ry"]
    near = int(np.argmin(np.abs(ry - y0)))

    w = sc["ys"][wid]
    i0 = int(np.searchsorted(XS, LEFT_X + STUB))
    i1 = min(i0 + 60, N_SAMPLES - 1)
    slope = (w[i1] - w[i0]) / (XS[i1] - XS[i0])
    y_hat = w[i1] + slope * (RIGHT_X - XS[i1])
    dep = int(np.argmin(np.abs(ry - y_hat)))

    return {"nearest_y": sc["labels"][near], "departure": sc["labels"][dep], "identity": sc["labels"][wid]}


def pick_wire(sc, band, want_dep, rng, bal):
    """The queried wire, or None: crossing count inside `band`, (departure pick == gold) == want_dep, and among
    those the least-used exact crossing count in this cell."""
    lo, hi = band
    cand = []
    for w in range(N_WIRES):
        n = sum(sc["cmat"][w])
        if not (lo <= n <= hi):
            continue
        gold = sc["labels"][sc["perm"][w]]
        if (shortcut_picks(sc, w)["departure"] == gold) != want_dep:
            continue
        cand.append(w)
    if not cand:
        return None
    rng.shuffle(cand)  # break ties randomly, not by wire index
    return min(cand, key=lambda w: bal[("xcount", sum(sc["cmat"][w]))])


@functools.cache
def _label_envelope():
    """(w, h) in 1x px that every 2-digit bead label is gated with: the rendered size of "88" plus 1 px."""
    img = Image.new("L", (8, 8))
    bb = ImageDraw.Draw(img).textbbox((0, 0), "88", font=_font(int(MARK_FONT * SS)))
    return (bb[2] - bb[0]) / SS + 1.0, (bb[3] - bb[1]) / SS + 1.0


def _rect_wire_dist(rect, ys_w, pad_x=4.0):
    """Min distance from rect (x0, y0, x1, y1) to one wire, sampled at every grid x within the rect +- pad_x."""
    x0, y0, x1, y1 = rect
    i0 = int(np.searchsorted(XS, x0 - pad_x))
    i1 = int(np.searchsorted(XS, x1 + pad_x)) + 1
    if i0 >= len(XS) or i1 <= 0 or i0 >= i1:
        return float("inf")
    xs = XS[i0:i1]
    ys = ys_w[i0:i1]
    dx = np.maximum(np.maximum(x0 - xs, xs - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - ys, ys - y1), 0.0)
    return float(np.min(np.hypot(dx, dy)))


def _pt_wire_dist(px, py, ys_w, win=40.0):
    i0 = int(np.searchsorted(XS, px - win))
    i1 = int(np.searchsorted(XS, px + win)) + 1
    xs = XS[i0:i1]
    ys = ys_w[i0:i1]
    if len(xs) == 0:
        return float("inf")
    return float(np.min(np.hypot(xs - px, ys - py)))


def _rects_overlap(a, b, pad=2.0):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def place_marks(sc, rng_m):
    """Numbered beads every MARK_SPACING along every wire -> [{wid, k, x, y (px), lbox (px rect), side, id}].
    rng_m is independent of the geometry rng, so the plain and labeled datasets share every scene."""
    spacing = MARK_SPACING * W / 1000.0
    cross_pts = [(x, y) for pts in sc["pair_cross"].values() for x, y, _ in pts]
    term_boxes = [
        (cx - TERM_R, cy - TERM_R, cx + TERM_R, cy + TERM_R)
        for cx, col in ((LEFT_X, sc["ly"]), (RIGHT_X, sc["ry"]))
        for cy in col
    ]
    placed = []
    for wid in range(N_WIRES):
        ys_w = sc["ys"][wid]
        k = 0
        x = LEFT_X + STUB + 0.8 * spacing
        while x < RIGHT_X - STUB - 0.4 * spacing:
            m = _try_place(sc, wid, ys_w, x, spacing, cross_pts, term_boxes, placed)
            if m is None:
                MARK_DROPS["unplaceable"] += 1
            else:
                m["k"] = k
                placed.append(m)
                k += 1
            x += spacing
    pool = [v for v in LABEL_POOL if v not in set(sc["labels"])]
    if len(pool) < len(placed):
        raise SystemExit(f"mark id pool exhausted: {len(placed)} beads > {len(pool)} free ids")
    rng_m.shuffle(pool)
    for m, mid in zip(placed, pool):
        m["id"] = mid
    return placed


def _try_place(sc, wid, ys_w, x_nom, spacing, cross_pts, term_boxes, placed):
    """One bead near x_nom on wire wid, nudged along x until every gate passes, else None."""
    lw, lh = _label_envelope()
    for dmag in range(0, int(0.46 * spacing), 4):
        for dx in (0,) if dmag == 0 else (dmag, -dmag):
            px = x_nom + dx
            if not (LEFT_X + STUB + 6 <= px <= RIGHT_X - STUB - 6):
                continue
            py = float(np.interp(px, XS, ys_w))
            # gate on the stored values: snap to the 0.1-unit grid items.jsonl carries
            px = round(1000.0 * px / W, 1) * W / 1000.0
            py = round(1000.0 * py / H, 1) * H / 1000.0
            if any(math.hypot(px - cx, py - cy) < MARK_CROSS_CLEAR for cx, cy in cross_pts):
                MARK_DROPS["bead_near_crossing"] += 1
                continue
            if any(_pt_wire_dist(px, py, sc["ys"][j]) < 9.0 for j in range(N_WIRES) if j != wid):
                MARK_DROPS["bead_near_other_wire"] += 1
                continue
            # the label sits above or below its bead; keep the slot with the most clearance from foreign wires
            best = None
            for sy in (-1, 1):
                for fx in (0.0, 0.55, -0.55, 1.0, -1.0):
                    cx = px + fx * lw * 0.5
                    cy = py + sy * (MARK_R + 2.5 + lh / 2)
                    rect = (cx - lw / 2 - 1.5, cy - lh / 2 - 1.5, cx + lw / 2 + 1.5, cy + lh / 2 + 1.5)
                    if rect[0] < 3 or rect[1] < 3 or rect[2] > W - 3 or rect[3] > H - 3:
                        continue
                    if _rect_wire_dist(rect, ys_w) < 2.0:
                        continue
                    other_d = min(
                        (_rect_wire_dist(rect, sc["ys"][j]) for j in range(N_WIRES) if j != wid), default=float("inf")
                    )
                    if other_d < MARK_GAP:
                        continue
                    bead_d = math.hypot(max(rect[0] - px, px - rect[2], 0), max(rect[1] - py, py - rect[3], 0))
                    if bead_d + MARK_OWN_MARGIN > other_d:
                        continue  # label must bind to its own bead
                    if any(_rects_overlap(rect, t, pad=6.0) for t in term_boxes):
                        continue
                    if any(_rects_overlap(rect, m["lbox"]) for m in placed):
                        continue
                    if any(
                        rect[0] - 8 <= ox <= rect[2] + 8 and rect[1] - 8 <= oy <= rect[3] + 8 for ox, oy in cross_pts
                    ):
                        continue
                    if best is None or other_d > best[0]:
                        best = (other_d, rect, sy)
            if best is None:
                MARK_DROPS["no_label_slot"] += 1
                continue
            _, rect, sy = best
            return {"wid": wid, "x": px, "y": py, "lbox": [round(v, 1) for v in rect], "side": sy}
    return None


def render(sc):
    """Wires, then beads (if sc has "marks"), then terminals -> (W, H) RGB image. Wire k passes over every wire
    j < k: a BG disc is punched at each of its crossings before it is stroked."""
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)
    for k, w in enumerate(sc["ys"]):
        for j in range(k):
            for x, y, _ in sc["pair_cross"][(j, k)]:
                d.ellipse([(x - R_BREAK) * SS, (y - R_BREAK) * SS, (x + R_BREAK) * SS, (y + R_BREAK) * SS], fill=BG)
        pts = [(float(x) * SS, float(y) * SS) for x, y in zip(XS, w)]
        d.line(pts, fill=WIRE, width=CORE_W * SS, joint="curve")

    for m in sc.get("marks", ()):
        d.ellipse(
            [(m["x"] - MARK_R) * SS, (m["y"] - MARK_R) * SS, (m["x"] + MARK_R) * SS, (m["y"] + MARK_R) * SS], fill=WIRE
        )
        fm = _font(int(MARK_FONT * SS))
        lab = str(m["id"])
        x0, y0, x1, y1 = m["lbox"]
        bb = d.textbbox((0, 0), lab, font=fm)
        d.text(
            ((x0 + x1) / 2 * SS - (bb[2] - bb[0]) / 2 - bb[0], (y0 + y1) / 2 * SS - (bb[3] - bb[1]) / 2 - bb[1]),
            lab,
            fill=TERM_EDGE,
            font=fm,
        )

    f = _font(int(19 * SS))
    for col, ys_, labs in (
        (LEFT_X, sc["ly"], list(LEFT_LABELS[:N_WIRES])),
        (RIGHT_X, sc["ry"], [str(v) for v in sc["labels"]]),
    ):
        for y, lab in zip(ys_, labs):
            x0, y0 = (col - TERM_R) * SS, (float(y) - TERM_R) * SS
            x1, y1 = (col + TERM_R) * SS, (float(y) + TERM_R) * SS
            d.rounded_rectangle([x0, y0, x1, y1], radius=5 * SS, fill=TERM_FILL, outline=TERM_EDGE, width=2 * SS)
            bb = d.textbbox((0, 0), lab, font=f)
            d.text(
                (col * SS - (bb[2] - bb[0]) / 2 - bb[0], float(y) * SS - (bb[3] - bb[1]) / 2 - bb[1]),
                lab,
                fill=TERM_EDGE,
                font=f,
            )
    return img.resize((W, H), Image.LANCZOS)


def _term(lab):
    return f"terminal {lab}"


def questions(sc, wid, rng, bal):
    """Items B0 (which terminal), B1 (crossing count) and B2 (pick of two) on one scene and wire. Only B0 is trained
    and evaluated; B1/B2 draw from the shared RNG stream, so the seeded scenes depend on them."""
    start = LEFT_LABELS[wid]
    gold = sc["labels"][sc["perm"][wid]]
    ncross = sum(sc["cmat"][wid])
    picks = shortcut_picks(sc, wid)
    opts_all = [_term(v) for v in sorted(sc["labels"])]
    stem = (
        "Each wire connects one lettered terminal on the left to one numbered "
        f"terminal on the right. Trace the wire attached to terminal {start}. "
    )

    b0 = {
        "template": "endpoint",
        "B": "B0",
        "question": stem + "Which numbered terminal does it connect to?",
        "options": opts_all,
        "answer": _term(gold),
    }

    # B1: 4 consecutive counts containing the truth; gold's rank is the least-used feasible one
    feasible = [r for r in range(4) if ncross - r >= 0]
    rank = min(feasible, key=lambda r: bal[("b1rank", r)])
    vals = list(range(ncross - rank, ncross - rank + 4))
    b1 = {
        "template": "crossings",
        "B": "B1",
        "question": stem + "How many other wires does it cross along the way?",
        "options": [WORDS[v] for v in vals],
        "answer": WORDS[ncross],
    }

    # B2: half the items pit gold against the departure pick, half against a random other terminal
    adversarial = bal[("b2adv", True)] <= bal[("b2adv", False)]
    others = [v for v in sc["labels"] if v != gold]
    foil = picks["departure"] if (adversarial and picks["departure"] != gold) else rng.choice(others)
    pair = [_term(gold), _term(foil)]
    if bal[("b2pos", 0)] > bal[("b2pos", 1)]:
        pair.reverse()
    b2 = {
        "template": "pick2",
        "B": "B2",
        "question": stem + "Which of these two numbered terminals does it connect to?",
        "options": pair,
        "answer": _term(gold),
    }

    bal[("b1rank", rank)] += 1
    bal[("b2adv", adversarial)] += 1
    bal[("b2pos", pair.index(_term(gold)))] += 1
    return [b0, b1, b2]


def pack(sc, wid, it, X):
    """Attach the ground-truth payload (0-1000 coordinates) to one question dict."""
    gold_slot = sc["perm"][wid]
    start = LEFT_LABELS[wid]
    picks = shortcut_picks(sc, wid)
    n1000 = lambda v, s: round(1000.0 * float(v) / s, 1)
    keep = np.linspace(0, N_SAMPLES - 1, 48).astype(int)  # 48-point polylines
    gold_poly = [[n1000(XS[i], W), n1000(sc["ys"][wid][i], H)] for i in keep]

    it.update(
        A=X,
        X=X,
        queried_attr=None,
        anchor_color=None,
        n_crossings=sum(sc["cmat"][wid]),
        n_wires=N_WIRES,
        start_label=start,
        gold_label=sc["labels"][gold_slot],
        floor_nearest_y=_term(picks["nearest_y"]),
        floor_departure=_term(picks["departure"]),
        floor_identity=_term(picks["identity"]),
        floor_nearest_y_is_gold=picks["nearest_y"] == sc["labels"][gold_slot],
        min_cross_angle_deg=round(sc["min_angle"], 1),
        min_cross_sep_px=round(sc["min_cross_sep"], 1),
        waypoint_probe=f"Trace the wire attached to terminal {start}. Output 8 points along it, "
        f"in order from left to right, as [x, y] normalized to 0-1000.",
        localization_probe=f"Point at terminal {start}. Answer with [x, y] normalized to 0-1000.",
        gold_polyline=gold_poly,
        wires=[
            {
                "wid": w,
                "left_label": LEFT_LABELS[w],
                "right_label": sc["labels"][sc["perm"][w]],
                "left_y": n1000(sc["ly"][w], H),
                "right_y": n1000(sc["ry"][sc["perm"][w]], H),
                "n_crossings": sum(sc["cmat"][w]),
                "polyline": [[n1000(XS[i], W), n1000(sc["ys"][w][i], H)] for i in keep],
            }
            for w in range(N_WIRES)
        ],
        # [x, y, angle_deg, partner_wid, gold_passes_over] in left-to-right order
        crossings_on_gold=sorted(
            (
                [n1000(x, W), n1000(y, H), round(a, 1), j, wid > j]
                for j in range(N_WIRES)
                if j != wid
                for x, y, a in sc["pair_cross"][(min(wid, j), max(wid, j))]
            ),
            key=lambda c: c[0],
        ),
        left_terminals=[{"label": LEFT_LABELS[w], "y": n1000(sc["ly"][w], H)} for w in range(N_WIRES)],
        marks=[
            {
                "id": m["id"],
                "wid": m["wid"],
                "k": m["k"],
                "x": n1000(m["x"], W),
                "y": n1000(m["y"], H),
                "lbox": m["lbox"],
                "side": m["side"],
            }
            for m in sc["marks"]
        ],
        right_terminals=[{"label": sc["labels"][s], "y": n1000(sc["ry"][s], H)} for s in range(N_WIRES)],
    )
    return it


def build_split(out, marks_out, per_cell, seed):
    """One split: the plain dataset in `out`, its bead-labeled twin in `marks_out`."""
    for d in (out, marks_out):
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
    MARK_DROPS.clear()
    rng = random.Random(seed)
    items, idx, scene = [], 0, 0
    for X, band in XBANDS.items():
        got, draws, bal = 0, 0, Counter()
        while got < per_cell and draws < MAX_DRAWS:
            draws += 1
            sc = build_scene(rng)
            if sc is None:
                continue
            # departure pick == gold on exactly 1/N_WIRES of scenes -> that floor sits at chance
            wid = pick_wire(sc, band, got % N_WIRES == 0, rng, bal)
            if wid is None:
                continue
            group = questions(sc, wid, rng, bal)
            name = f"{X}_{scene:04d}.png"
            render(sc).save(os.path.join(out, "images", name))
            sc["marks"] = place_marks(sc, random.Random(f"marks:{seed}:{scene}"))
            render(sc).save(os.path.join(marks_out, "images", name))
            for it in group:
                pack(sc, wid, it, X)
                it["idx"], it["image"], it["scene"], it["queried_wid"] = idx, name, scene, wid
                items.append(it)
                idx += 1
            bal[("xcount", sum(sc["cmat"][wid]))] += 1
            scene += 1
            got += 1

    meta = {
        "seed": seed,
        "n_wires": N_WIRES,
        "sigma": SIGMA,
        "cols": N_COLS,
        "samples": N_SAMPLES,
        "term_sep": TERM_SEP,
        "ctrl_inset": CTRL_INSET,
        "min_angle": MIN_ANGLE,
        "min_cross_sep": MIN_CROSS_SEP,
        "min_gap": MIN_GAP,
        "clear_px": CLEAR_PX,
        "max_slope": MAX_SLOPE,
        "stub": STUB,
        "W": W,
        "H": H,
        "shortcut_control": True,
        "xbands": XBANDS,
        "n_items": len(items),
        "n_scenes": scene,
        "white": True,
    }
    meta_m = dict(
        meta,
        marks_spacing_units=MARK_SPACING,
        mark_r=MARK_R,
        mark_font=MARK_FONT,
        mark_gap=MARK_GAP,
        mark_drops=dict(MARK_DROPS),
    )
    for d, m, with_marks in ((out, meta, False), (marks_out, meta_m, True)):
        with open(os.path.join(d, "items.jsonl"), "w") as f:
            for it in items:
                f.write(json.dumps({k: v for k, v in it.items() if with_marks or k != "marks"}) + "\n")
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(m, f, indent=2)


def generate():
    """train (5,001 scenes) and heldout (501), each with its _labeled twin."""
    jobs = [(OUT / split, OUT / f"{split}_labeled", n, seed) for split, n, seed in SPLITS]
    with Pool(2) as pool:
        pool.starmap(build_split, jobs)


# ---------------------------------------------------------------- sft
# One ground-truth step list per B0 train item (gold polyline, crossings, slopes, terminal rank), spelled per trace
# style: direct (answer only), cot (positions in words), mark (printed bead ids), ground ([x, y] in 0-1000).
STYLES = {  # reasoning format -> trace style, on plain (False) and labeled (True) images
    False: {"direct": "direct", "text": "cot", "grounded": "ground"},
    True: {"direct": "direct", "text": "mark", "grounded": "ground"},
}
WIRE_STEM = " Answer with the terminal label in \\boxed{} format."
SUFFIXES = {
    "direct": WIRE_STEM,
    "cot": (
        " Think step by step inside <think> </think> tags, then give your final answer in \\boxed{} "
        "format. While thinking, follow the wire from its terminal across the image, describing in "
        "words where it runs and each wire it crosses, then say which terminal it reaches."
    ),
    "ground": (
        " Think step by step inside <think> </think> tags, then give your final answer in \\boxed{} "
        "format. While thinking, follow the wire from its terminal across the image, tracing it with "
        "[x, y] point coordinates (integers normalized to 0-1000) and marking each wire it crosses, "
        "then say which terminal it reaches."
    ),
    "mark": (
        " Think step by step inside <think> </think> tags, then give your final answer in \\boxed{} "
        "format. While thinking, follow the wire from its terminal across the image, naming each "
        "numbered mark printed on it in order and noting each wire it crosses, then say which "
        "terminal it reaches."
    ),
}

SPACING = 55.0  # x-spacing (0-1000 units) of the plain waypoints between event landmarks
LETTERS = "abcdefghijklmnopqrstuvwxyz"

# cot position words: seven bands per axis
XW = [(0, "the far left"), (140, "the left"), (290, "left of centre"), (430, "the centre"),
      (570, "right of centre"), (710, "the right"), (860, "the far right")]  # fmt: skip
YW = [(0, "near the top"), (140, "high up"), (290, "above the middle"), (430, "level with the middle"),
      (570, "below the middle"), (710, "low down"), (860, "near the bottom")]  # fmt: skip


def _band(v, table):
    out = table[0][1]
    for lo, name in table:
        if v >= lo:
            out = name
    return out


def words(x, y):
    """Position (0-1000) -> coordinate-free phrase."""
    return f"{_band(x, XW)}, {_band(y, YW)}"


def ordinal(n):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def question_block(it):
    """Question + lettered options."""
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(it["options"]))
    return f"{it['question']}\n{opts}"


def prompt(it, style):
    """Text prompt, shared by SFT and eval."""
    return question_block(it) + SUFFIXES[style]


def steps_for(it):
    """One B0 item -> ordered step dicts with kind and pos (x, y) in 0-1000; kinds: start | turn | cross | along
    (plain waypoint) | dot (printed bead) | arrive. On labeled data every non-dot step carries `near`, the id of
    the closest gold-wire bead."""
    gold = next(w for w in it["wires"] if w["wid"] == it["queried_wid"])
    poly = gold["polyline"]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]

    rights = sorted(it["right_terminals"], key=lambda t: t["y"])
    rank = next(i for i, t in enumerate(rights) if str(t["label"]) == str(gold["right_label"])) + 1

    def slope_at(x):
        """sign of dy/dx of the gold wire near x -> 'falls' | 'rises' | 'runs level'"""
        i = min(range(len(xs)), key=lambda k: abs(xs[k] - x))
        a, b = max(0, i - 2), min(len(xs) - 1, i + 2)
        d = ys[b] - ys[a]
        return "falls" if d > 22 else ("rises" if d < -22 else "runs level")

    out = [{"kind": "start", "pos": (poly[0][0], poly[0][1]), "label": it["start_label"], "dir": slope_at(xs[0] + 60)}]

    # turning points: local extrema in y
    turns = []
    for i in range(3, len(ys) - 3):
        seg = ys[i - 3 : i + 4]
        is_extremum = ys[i] == max(seg) or ys[i] == min(seg)
        deep = abs(ys[i] - ys[i - 3]) > 45 and abs(ys[i] - ys[i + 3]) > 45
        if is_extremum and deep and (not turns or xs[i] - turns[-1] > 120):
            turns.append(xs[i])
            out.append({"kind": "turn", "pos": (xs[i], ys[i]), "peak": "highest" if ys[i] == min(seg) else "lowest"})

    for n, c in enumerate(it["crossings_on_gold"], 1):  # c = [x, y, angle, partner, gold_passes_over]
        out.append({"kind": "cross", "pos": (c[0], c[1]), "n": n, "over": bool(c[4]), "dir": slope_at(c[0])})

    # plain waypoints along the whole wire, skipping ones that crowd an event
    def y_at(x):
        for k in range(len(xs) - 1):
            if xs[k] <= x <= xs[k + 1]:
                t = (x - xs[k]) / max(xs[k + 1] - xs[k], 1e-9)
                return ys[k] + t * (ys[k + 1] - ys[k])
        return ys[-1] if x > xs[-1] else ys[0]

    evx = [s["pos"][0] for s in out]
    x = xs[0] + SPACING
    while x < xs[-1] - SPACING * 0.5:
        if all(abs(x - e) > SPACING * 0.6 for e in evx):
            out.append({"kind": "along", "pos": (x, y_at(x)), "dir": slope_at(x)})
        x += SPACING

    # printed beads on the gold wire (x order = trace order, wires are x-monotone)
    dots = sorted((m for m in (it.get("marks") or ()) if m["wid"] == it["queried_wid"]), key=lambda m: m["x"])
    for m in dots:
        out.append({"kind": "dot", "pos": (m["x"], m["y"]), "id": m["id"], "dir": slope_at(m["x"])})

    out.sort(key=lambda s: s["pos"][0])
    out.append(
        {
            "kind": "arrive",
            "pos": (poly[-1][0], poly[-1][1]),
            "label": str(gold["right_label"]),
            "rank": rank,
            "n_rights": len(rights),
            "n_cross": it["n_crossings"],
        }
    )
    if dots:
        for s in out:
            if s["kind"] != "dot":
                s["near"] = min(dots, key=lambda m: abs(m["x"] - s["pos"][0]))["id"]
    return out


def trace(stps, mode):
    """Step list -> <think> body for mode in {'cot', 'ground', 'mark'}: the same sentence frames, only the position
    spelling differs. mark walks the printed beads and anchors events on the nearest bead ("near mark N")."""
    marked = mode == "mark"
    if marked:
        at = lambda s: f"mark {s['near']}"
    elif mode == "ground":
        at = lambda s: f"[{int(s['pos'][0])}, {int(s['pos'][1])}]"
    else:
        at = lambda s: words(*s["pos"])
    walk_kind = "dot" if marked else "along"
    lines = []
    last_named = None  # mark mode: a bead named by the previous sentence is not re-walked
    i = 0
    while i < len(stps):
        s = stps[i]
        k = s["kind"]
        if k in ("along", "dot") and k != walk_kind:
            i += 1
            continue
        # a run of consecutive waypoints becomes ONE "Following the wire: a -> b -> c." sentence
        if k == walk_kind:
            run = []
            while i < len(stps) and stps[i]["kind"] in ("along", "dot"):
                if stps[i]["kind"] == walk_kind:
                    run.append(stps[i])
                i += 1
            if run and marked and run[0]["id"] == last_named:
                run = run[1:]
            if run:
                body = " -> ".join(f"mark {r['id']}" if marked else at(r) for r in run)
                lines.append(f"Following the wire: {body}. It {run[-1]['dir']}.")
                if marked:
                    last_named = run[-1]["id"]
            continue
        i += 1
        if marked and "near" in s:
            last_named = s["near"]
        if k == "start":
            if marked:
                lines.append(
                    f"Terminal {s['label']} is on the left; its wire starts near {at(s)}. I'll follow its "
                    f"wire to the right. It leaves the terminal level, then {s['dir']}."
                )
            else:
                lines.append(
                    f"Terminal {s['label']} is on the left at {at(s)}. I'll follow its "
                    f"wire to the right. It leaves the terminal level, then {s['dir']}."
                )
        elif k == "turn":
            where = f"Near {at(s)}" if marked else f"At {at(s)}"
            lines.append(f"{where}, the wire reaches its {s['peak']} point and turns back.")
        elif k == "cross":
            where = f"Near {at(s)}" if marked else f"At {at(s)}"
            lines.append(
                f"{where}, it passes {'over' if s['over'] else 'under'} another "
                f"wire — crossing {s['n']}. Staying on the same wire, it {s['dir']}."
            )
        else:  # arrive
            nc = s["n_cross"]
            where = f"just past {at(s)}" if marked else f"at {at(s)}"
            lines.append(
                f"Having crossed {nc} wire{'' if nc == 1 else 's'}, it arrives on the right "
                f"{where}, the {ordinal(s['rank'])} terminal from the top of the "
                f"{s['n_rights']} on that side. That terminal is labelled {s['label']}."
            )
    return "\n".join(lines)


def target(it, style):
    if style == "direct":
        return f"\\boxed{{{it['answer']}}}"
    return f"<think>\n{trace(steps_for(it), style)}\n</think>\n\n\\boxed{{{it['answer']}}}"


def build_sft(data, out_paths):
    """B0 items of one split -> one json per trace style ({style: path}); image paths relative to $GR_DATA."""
    with open(os.path.join(data, "items.jsonl")) as f:
        b0 = [it for it in map(json.loads, f) if it["B"] == "B0"]
    images = os.path.relpath(os.path.join(data, "images"), DATA)
    for style, path in out_paths.items():
        rows = [
            {
                "image": f"{images}/{it['image']}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt(it, style)}"},
                    {"from": "gpt", "value": target(it, style)},
                ],
            }
            for it in b0
        ]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(rows, f)
        print(f"{len(rows)} rows -> {path}")


def prep_sft():
    for labeled in (False, True):
        build_sft(
            data_dir(TASK) / ("train_labeled" if labeled else "train"),
            {style: sft_path(TASK, fmt, labeled) for fmt, style in STYLES[labeled].items()},
        )


# ---------------------------------------------------------------- eval
# Greedy decoding on the 501 held-out B0 items; the answer is the last \boxed{} span mapped onto the options.
BATCH_SIZE = 8
MAX_NEW_TOKENS = 2048
# one level of nesting so \boxed{\text{(a) terminal 12}} is captured
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_TEXT_WRAP_RE = re.compile(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}")
_MCQ_LETTER_RE = re.compile(r"^\(?([a-z])\)?[.:) ]*$")
_LEADING_LETTER_RE = re.compile(r"^\s*\(?([a-z])\)[\s.:]")


def mcq_norm(s):
    return re.sub(r"[^a-z0-9 /]", "", s.lower()).strip()


def _letter_to_opt(ch, options):
    i = LETTERS.find(ch)
    return options[i] if 0 <= i < len(options) else None


def _opt_from_text(c, options):
    """A boxed/declared string -> matched option (exact, else the longest option contained in it)."""
    t = mcq_norm(c)
    if not t:
        return None
    for o in options:
        if mcq_norm(o) == t:
            return o
    contained = [o for o in options if mcq_norm(o) and mcq_norm(o) in t]
    return max(contained, key=lambda o: len(mcq_norm(o))) if contained else None


def parse_boxed_mcq(text, options):
    """Generation -> option text, or None. Priority: last \\boxed{...} (letter or option text), last "answer is
    (e)" declaration, last standalone "(letter)", the option mentioned last."""
    low = text.lower()
    valid = LETTERS[: len(options)]
    if not valid:
        return None
    bx = BOXED_RE.findall(text)
    if bx:
        c = _TEXT_WRAP_RE.sub(r"\1", bx[-1]).strip()
        m = _MCQ_LETTER_RE.match(c.lower()) or _LEADING_LETTER_RE.match(c.lower() + " ")
        if m:
            o = _letter_to_opt(m.group(1), options)
            if o:
                return o
        o = _opt_from_text(c, options)
        if o:
            return o
    decl = re.findall(
        rf"(?:correct answer|final answer|answer|option)\s*(?:is|:|=|\bwould be\b)?\s*\(?([{valid}])\)?(?![a-z])", low
    )
    if decl:
        o = _letter_to_opt(decl[-1], options)
        if o:
            return o
    parens = re.findall(rf"\(([{valid}])\)", low)
    if parens:
        o = _letter_to_opt(parens[-1], options)
        if o:
            return o
    tnorm = mcq_norm(text)
    best, best_pos = None, -1
    for o in options:
        no = mcq_norm(o)
        p = tnorm.rfind(no) if no else -1
        if p > best_pos:
            best, best_pos = o, p
    return best


def score(it, gen):
    pred = parse_boxed_mcq(gen, it["options"])
    return {"pred": pred, "hit": pred is not None and mcq_norm(pred) == mcq_norm(it["answer"])}


def evaluate(args):
    """-> one row per item (scene, X, pred, hit, resp) in $GR_RESULTS/probes/wire_tracing/heldout/<run>.jsonl."""
    style = STYLES[args.labeled][args.format]
    data = data_dir(TASK) / ("heldout_labeled" if args.labeled else "heldout")
    model_dir = str(args.model or checkpoint_dir(TASK, args.format, args.labeled))
    out_path = args.out or results_dir(TASK) / "heldout" / f"{run_name(args.format, args.labeled)}.jsonl"

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    with open(data / "items.jsonl") as f:
        items = [it for it in map(json.loads, f) if it["B"] == "B0"]
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()
    processor.tokenizer.padding_side = "left"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    hits = []
    with open(out_path, "w") as rf:
        for lo in range(0, len(items), BATCH_SIZE):
            chunk = items[lo : lo + BATCH_SIZE]
            images = [Image.open(data / "images" / it["image"]).convert("RGB") for it in chunk]
            msgs = [
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(it, style)}]}]
                for it in chunk
            ]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            with torch.no_grad():
                inp = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
                out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            gens = processor.tokenizer.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)
            for it, g in zip(chunk, gens):
                rec = {"scene": it["scene"], "X": it["X"], **score(it, g), "resp": g}
                hits.append(rec["hit"])
                rf.write(json.dumps(rec) + "\n")
    print(f"{out_path}: n={len(hits)} acc={sum(hits) / len(hits):.4f}")


def main():
    ap = argparse.ArgumentParser(prog="python -m grounded_reasoning.probes.wire_tracing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="$GR_DATA/probes/wire_tracing/{train,heldout}[_labeled]/ (CPU)")
    sub.add_parser("sft", help="$GR_DATA/probes/wire_tracing/sft/<run>.json")
    ev = sub.add_parser("eval", help="greedy held-out eval of one checkpoint (1 GPU)")
    ev.add_argument("--format", required=True, choices=FORMATS)
    ev.add_argument("--labeled", action="store_true", help="labeled images (beads with printed numbers)")
    ev.add_argument("--model", help="default: $GR_RUNS/probes/wire_tracing_<run>")
    ev.add_argument("--out", help="default: $GR_RESULTS/probes/wire_tracing/heldout/<run>.jsonl")
    sub.add_parser("summarize", help="$GR_RESULTS/probes/wire_tracing/summary.json")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate()
    elif args.cmd == "sft":
        prep_sft()
    elif args.cmd == "eval":
        evaluate(args)
    else:
        summarize(TASK)


if __name__ == "__main__":
    main()
