"""Shared by the three probe tasks: the on-disk layout, the shape primitives of the counting and arrow scenes, the
SFT text helpers and the per-task summary (accuracy per run and stratum, paired McNemar tests between formats).
"""

import itertools
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
from PIL import ImageFont

from grounded_reasoning import DATA, RESULTS, RUNS

# --- layout. A run is one reasoning format, suffixed `_labeled` when its images print a label on every object.
FORMATS = ("direct", "text", "grounded")


def run_name(fmt, labeled):
    return f"{fmt}_labeled" if labeled else fmt


def data_dir(task):
    return DATA / "probes" / task


def sft_path(task, fmt, labeled):
    return data_dir(task) / "sft" / f"{run_name(fmt, labeled)}.json"


def checkpoint_dir(task, fmt, labeled):
    return RUNS / "probes" / f"{task}_{run_name(fmt, labeled)}"


def results_dir(task):
    return RESULTS / "probes" / task


# --- shapes: star-shaped silhouettes stored as a radial signature r(phi) on an NSIG-point grid ("blob" = smooth
# Fourier contour, "poly" = straight-edged polygon with 4-9 corners). RNG call order is part of the generated data.
SS = 2  # supersample factor; renders are drawn at SS x and LANCZOS-downscaled
MARGIN = 36.0  # shape polygon to canvas edge (px)

K_HARM = (2, 3, 4, 5, 6, 7)  # blob Fourier harmonics; no k=1 term keeps the centroid at the centre
NSIG = 512
POLY_STORE = 120  # stored polygon points per shape
MIN_RADIUS = 0.34  # a shape is resampled if r(phi) dips below this
AMP_TOTAL = (0.30, 0.62)
DECAY = (0.4, 1.4)
DOM_BOOST = (1.5, 2.5)
N_CORNERS = (4, 9)  # inclusive
CORNER_JITTER = 0.70  # angular jitter as a fraction of the mean corner spacing
RADIUS_JITTER = (0.36, 1.0)  # per-corner radius multiplier

FILL = (172, 177, 184)
OUTLINE = (55, 60, 68)
QUERY_OUTLINE = (36, 88, 201)  # the blue ring on the query / referent shape
LABEL_INK = (20, 20, 24)

PHI = np.linspace(0.0, 2.0 * np.pi, NSIG, endpoint=False)
DPHI = 2.0 * np.pi / NSIG


def _poly_signature(ang, rad):
    """Sorted corner angles + radii -> exact r(phi) of the star-shaped polygon, normalised to mean 1.
    The ray at phi meets edge P_i->P_j at t = (P_i x d) / (u x d), d = P_j - P_i, u = (cos phi, sin phi)."""
    ang = np.asarray(ang, float)
    rad = np.asarray(rad, float)
    px, py = rad * np.cos(ang), rad * np.sin(ang)
    i = (np.searchsorted(ang, PHI, side="right") - 1) % len(ang)
    j = (i + 1) % len(ang)
    dx, dy = px[j] - px[i], py[j] - py[i]
    ux, uy = np.cos(PHI), np.sin(PHI)
    num = px[i] * dy - py[i] * dx
    den = ux * dy - uy * dx
    r = np.where(np.abs(den) < 1e-12, rad[i], num / np.where(np.abs(den) < 1e-12, 1.0, den))
    r = np.maximum(r, 1e-6)
    return r / r.mean()


def signature(shape, mode):
    """shape params -> r(phi) (NSIG,). blob: (amp, phase) per harmonic; poly: (corner angles, radii)."""
    a, b = shape
    if mode == "poly":
        return _poly_signature(a, b)
    return 1.0 + sum(w * np.cos(k * PHI + p) for w, p, k in zip(a, b, K_HARM))


def sample_shape(rng, mode):
    """-> shape params, rejection-sampled until the contour never dips below MIN_RADIUS."""
    if mode == "poly":
        for _ in range(200):
            n = int(rng.integers(N_CORNERS[0], N_CORNERS[1] + 1))
            step = 2.0 * np.pi / n
            jit = rng.uniform(-CORNER_JITTER * step / 2, CORNER_JITTER * step / 2, n)
            cand = np.sort((np.arange(n) * step + jit) % (2.0 * np.pi)), rng.uniform(*RADIUS_JITTER, n)
            sig = signature(cand, mode)
            if sig.min() / sig.mean() >= MIN_RADIUS:
                return cand
        raise RuntimeError("poly sampler starved")
    for _ in range(200):
        decay = rng.uniform(*DECAY)
        amp = np.array([rng.uniform(0.5, 1.5) / k**decay for k in K_HARM])
        amp[rng.integers(0, len(K_HARM))] *= rng.uniform(*DOM_BOOST)
        amp *= rng.uniform(*AMP_TOTAL) / amp.sum()
        ph = rng.uniform(0.0, 2.0 * np.pi, len(K_HARM))
        if signature((amp, ph), mode).min() >= MIN_RADIUS:
            return amp, ph
    raise RuntimeError("shape sampler starved")


def _min_shift_dist(u, v):
    """min over circular shifts of RMS(u - roll(v)), via FFT cross-correlation."""
    corr = np.fft.irfft(np.fft.rfft(u) * np.conj(np.fft.rfft(v)), n=len(u))
    d2 = (u * u).mean() + (v * v).mean() - 2.0 * corr.max() / len(u)
    return math.sqrt(max(d2, 0.0))


def shape_dist(su, sv):
    """Rotation- and mirror-invariant RMS distance between two mean-1 signatures."""
    return min(_min_shift_dist(su, sv), _min_shift_dist(su, sv[::-1]))


def polar_geom(shape, mode, area):
    """-> (scale giving exactly `area` px^2, perimeter px, max radius px, signature)."""
    sig = signature(shape, mode)
    s = math.sqrt(area / (0.5 * (sig**2).sum() * DPHI))
    dr = np.gradient(sig, DPHI)
    perim = s * (np.sqrt(sig**2 + dr**2).sum() * DPHI)
    return s, perim, s * sig.max(), sig


def world_poly(sig, scale, cx, cy):
    """Upright contour in px: (POLY_STORE, 2), vertex i at angle i * 2pi / POLY_STORE."""
    idx = np.linspace(0, NSIG, POLY_STORE, endpoint=False)
    base = np.interp(idx % NSIG, np.arange(NSIG), sig, period=NSIG)
    ang = idx * DPHI
    return np.stack([cx + scale * base * np.cos(ang), cy + scale * base * np.sin(ang)], axis=1)


def place(k, maxrs, rng, W, H, gap, tries=4000):
    """k centres, each >= MARGIN + its max radius from the edge and >= r_i + r_j + gap from every earlier
    centre (per-shape dart throwing); None if one shape cannot be placed."""
    pos = []
    for i in range(k):
        r_i = maxrs[i]
        for _ in range(tries):
            cx = rng.uniform(MARGIN + r_i, W - MARGIN - r_i)
            cy = rng.uniform(MARGIN + r_i, H - MARGIN - r_i)
            if all(math.hypot(cx - x, cy - y) >= r_i + maxrs[j] + gap for j, (x, y) in enumerate(pos)):
                pos.append((cx, cy))
                break
        else:
            return None
    return pos


def poly_min_gap(polys):
    """Exact min vertex-to-vertex distance over all polygon pairs (px). O(K^2 * POLY_STORE^2)."""
    best = float("inf")
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            d2 = ((polys[i][:, None, :] - polys[j][None, :, :]) ** 2).sum(-1)
            best = min(best, math.sqrt(d2.min()))
    return best


FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # the released data was rendered with this font
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def label_font(px):
    """-> (font, path) of the first font file found."""
    for cand in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, px), cand
        except OSError:
            continue
    return ImageFont.load_default(), "PIL_default"


def n1000(x, span):
    """px -> 0-1000 normalised coordinate, 1 decimal."""
    return round(1000.0 * float(x) / span, 1)


# --- SFT text of the counting and arrow traces: a position is "[x, y]" (0-1000 object centre, style `ground`) or a
# 3x3 image cell plus a within-cell qualifier (style `cot`).
THINK_HEAD = (
    " Think step by step inside <think> </think> tags, then give your final "
    "answer in \\boxed{} format. While thinking, first think about what you need "
    "to locate, then "
)
SUFFIXES = {
    "direct": " Answer directly with the point in \\boxed{} format, without any explanation.",
    "cot": THINK_HEAD + "describe where each relevant object is and what it looks like in words, then analyze those "
    "descriptions step-by-step.",
    "ground": THINK_HEAD + "locate each relevant object with its [x, y] point coordinate "
    "(integers normalized to 0-1000), then analyze the coordinates step-by-step.",
}

# Words, not digits, so the cot trace stays digit-free.
ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
    "eighteenth", "nineteenth", "twentieth",
]  # fmt: skip
ROWS = ("top", "middle", "bottom")
COLS = ("left", "centre", "right")


def cell_phrase(center):
    """[x, y] in 0-1000 -> its 3x3 cell phrase, e.g. "the top left of the image"."""
    r = 0 if center[1] < 333 else 1 if center[1] < 667 else 2
    c = 0 if center[0] < 333 else 1 if center[0] < 667 else 2
    if (r, c) == (1, 1):
        return "the centre of the image"
    return f"the {ROWS[r]} {COLS[c]} of the image"


def ipt(p):
    """[x, y] floats -> "[x, y]" rounded integers."""
    return f"[{p[0]:.0f}, {p[1]:.0f}]"


def build_refs(objs, ground):
    """{bid: unique position phrase} for objects with "bid" and "center" (0-1000).
    ground -> "[346, 761]"; cot -> "the bottom centre of the image (the left-hand one)": the cell phrase plus,
    when several objects share a cell, a qualifier ordering them along the cell's longer spread axis."""
    if ground:
        return {b["bid"]: ipt(b["center"]) for b in objs}
    refs, groups = {}, {}
    for b in objs:
        groups.setdefault(cell_phrase(b["center"]), []).append(b)
    for phrase, grp in groups.items():
        if len(grp) == 1:
            refs[grp[0]["bid"]] = phrase
            continue
        xs = max(b["center"][0] for b in grp) - min(b["center"][0] for b in grp)
        ys = max(b["center"][1] for b in grp) - min(b["center"][1] for b in grp)
        horiz = xs >= ys
        order = sorted(grp, key=lambda b: b["center"][0] if horiz else b["center"][1])
        if len(order) == 2:
            quals = ["the left-hand one", "the right-hand one"] if horiz else ["the upper one", "the lower one"]
        else:
            quals = [f"the {ORDINALS[i]} from the {'left' if horiz else 'top'}" for i in range(len(order))]
        for b, q in zip(order, quals):
            refs[b["bid"]] = f"{phrase} ({q})"
    return refs


def write_sft(path, rows):
    """Shuffle with seed 17 (every style built from one split gets the same row order) and write the json list."""
    random.Random(17).shuffle(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f)


# --- summary of $GR_RESULTS/probes/<task>/<eval set>/<run>.jsonl
STRATA = ("X", "K", "M", "H_HOPS")


def load(path):
    """rows jsonl -> {scene: row}."""
    with open(path) as f:
        return {r["scene"]: r for r in map(json.loads, f)}


def acc(rows):
    return sum(r["hit"] for r in rows) / len(rows)


def mcnemar_p(gained, lost):
    """Two-sided exact McNemar p from the discordant counts."""
    n, k = gained + lost, min(gained, lost)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n)


def run_stats(rows):
    """{scene: row} -> n, correct, accuracy, and per stratum key that varies: {value: {n, accuracy}}."""
    rs = list(rows.values())
    by_stratum = {}
    for key in STRATA:
        by = defaultdict(list)
        for r in rs:
            if key in r:
                by[r[key]].append(r)
        if len(by) > 1:
            by_stratum[key] = {str(v): {"n": len(by[v]), "accuracy": acc(by[v])} for v in sorted(by)}
    return {"n": len(rs), "correct": sum(r["hit"] for r in rs), "accuracy": acc(rs), "by_stratum": by_stratum}


def paired(rows_a, rows_b):
    """b vs a on their shared scenes: gained = b right and a wrong, lost = the reverse."""
    shared = sorted(set(rows_a) & set(rows_b))
    gained = sum(rows_b[s]["hit"] and not rows_a[s]["hit"] for s in shared)
    lost = sum(rows_a[s]["hit"] and not rows_b[s]["hit"] for s in shared)
    return {
        "n": len(shared),
        "gained": gained,
        "lost": lost,
        "delta": (gained - lost) / len(shared),
        "p": mcnemar_p(gained, lost),
    }


def summarize_set(set_dir):
    """One eval-set directory of <run>.jsonl files -> {accuracy, runs, paired}."""
    out = {"accuracy": {}, "runs": {}, "paired": []}
    for images, labeled in (("plain", False), ("labeled", True)):
        rows = {
            fmt: load(set_dir / f"{run_name(fmt, labeled)}.jsonl")
            for fmt in FORMATS
            if (set_dir / f"{run_name(fmt, labeled)}.jsonl").exists()
        }
        if not rows:
            continue
        for fmt, r in rows.items():
            out["runs"][run_name(fmt, labeled)] = run_stats(r)
        out["accuracy"][images] = {fmt: out["runs"][run_name(fmt, labeled)]["accuracy"] for fmt in rows}
        for a, b in itertools.combinations(rows, 2):
            out["paired"].append({"images": images, "a": a, "b": b, **paired(rows[a], rows[b])})
    return out


def table(name, summary):
    """-> printed lines for one eval set."""
    lines = [f"[{name}]"]
    for run, s in summary["runs"].items():
        parts = [f"{run:18s} n={s['n']:4d} acc={s['accuracy']:.4f}"]
        for key, by in s["by_stratum"].items():
            parts.append(" ".join(f"{key}{v}={c['accuracy']:.3f}" for v, c in by.items()))
        lines.append("  ".join(parts))
    for c in summary["paired"]:
        lines.append(
            f"{c['images']:7s} {c['b']} vs {c['a']}: n={c['n']} delta={100 * c['delta']:+.1f}pt "
            f"gained={c['gained']} lost={c['lost']} p={c['p']:.3g}"
        )
    return lines


def summarize(task):
    """Print the table and write $GR_RESULTS/probes/<task>/summary.json; the probe figure reads
    ["eval_sets"]["heldout"]["accuracy"][images][format]."""
    root = results_dir(task)
    sets = {d.name: summarize_set(d) for d in sorted(root.iterdir()) if d.is_dir()}
    summary = {"task": task, "eval_sets": {name: s for name, s in sets.items() if s["runs"]}}
    for name, s in summary["eval_sets"].items():
        print("\n".join(table(name, s)))
    with open(root / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {root / 'summary.json'}")
