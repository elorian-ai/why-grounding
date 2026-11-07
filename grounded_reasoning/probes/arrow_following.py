"""Arrow following: 48 gray blobs, each with one arrow pointing at another; start at the blue-outlined blob, follow the
arrows for H hops and point at the landing blob.
Data: $GR_DATA/probes/arrow_following/, results: $GR_RESULTS/probes/arrow_following/.

  python -m grounded_reasoning.probes.arrow_following generate | sft | summarize
  python -m grounded_reasoning.probes.arrow_following eval --format F [--labeled]
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

from grounded_reasoning import DATA
from grounded_reasoning.probes import common as S
from grounded_reasoning.probes.common import (
    FORMATS,
    THINK_HEAD,
    build_refs,
    checkpoint_dir,
    data_dir,
    ipt,
    results_dir,
    run_name,
    sft_path,
    summarize,
    write_sft,
)
from grounded_reasoning.probes.common import SUFFIXES as SHARED_SUFFIXES

TASK = "arrow_following"
OUT = data_dir(TASK)
PX = np.array([0.96, 0.64])  # 0-1000 units -> 3:2 pixel aspect, so bearings are true on the image

# ---------------------------------------------------------------- generate
# Arrows are angularly unambiguous and never mutual. One scene -> items PT_L0 (plain render), PT_L1 (same question,
# labeled render: a 2-digit number on every blob) and MCQ_L1. 90 shards (shards/h{H}_{i}) -> train (50,000 scenes,
# H = 2/4/6) + heldout (500).
MODE = "blob"
K = 48
PRESET = {  # recorded verbatim in meta.json
    "w": 1440,
    "h": 960,
    "area": 2000.0,
    "gap": 12.0,
    "arrow_len": 24.0,
    "arrow_sep": 8.0,
    "behind_ang": 5.0,
    "lbl_pt": 17,
    "lbl_off": 15.0,
    "poly_gap": 8.0,
    "max_hop": None,
    "repair": True,
    "lbl_inside": True,
    "arrow_clear": 3.5,
}
W, H = PRESET["w"], PRESET["h"]
AREA, GAP = PRESET["area"], PRESET["gap"]
ARROW_LEN, ARROW_SEP = PRESET["arrow_len"], PRESET["arrow_sep"]
BEHIND_ANG = PRESET["behind_ang"]
LBL_PT = PRESET["lbl_pt"]  # labels sit at the blob centre on a light plate
MIN_POLY_GAP = PRESET["poly_gap"]
ARROW_CLEAR = 5.0  # arrow-to-polygon clearance (px); the generator uses this, not PRESET["arrow_clear"]
MAX_HOP = 2.24 * math.sqrt((W - 2 * S.MARGIN) * (H - 2 * S.MARGIN) / K)  # 2.24 x the mean blob spacing
MIN_SHAPE_D = 0.055  # pairwise contour-distance floor
MIN_ANG = 16.0  # deg: near competitors must be at least this far off every source->target ray
BEHIND_RATIO = 1.25  # a competitor farther than ratio * d(target) is "behind" and owes only BEHIND_ANG
ARROW_W = 3
ARROW_INK = (10, 10, 12)

QUESTION_PT = (
    "This image shows {k} gray blobs. Each blob has one short arrow pointing at "
    "exactly one other blob. Start at the blue-outlined blob and follow the arrows "
    "hop by hop. Point to the blob you land on after {h} hops."
)
QUESTION_MCQ = (
    "This image shows {k} gray blobs, each labeled with a number. Each blob has "
    "one short arrow pointing at exactly one other blob. Start at the blue-outlined "
    "blob and follow the arrows hop by hop. Which blob do you land on after "
    "{h} hops?"
)
HOP_PROBE = "Point at the blob that the blue-outlined blob's arrow points at. Answer with [x, y] normalized to 0-1000."
ENUM_PROBE = "Point at every blob in the image, one point per blob, as [x, y] normalized to 0-1000."
QUERY_PROBE = "Point at the blob outlined in blue. Answer with [x, y] normalized to 0-1000."

# (dir, sid prefix, H, seed, scenes) per shard.
K48_SHARDS = [
    (f"shards/h{h}_{i:02d}", f"k48h{h}_{i:02d}_", h, 20260901 + 997 * h + i, 563) for h in (2, 4, 6) for i in range(30)
]
N_TRAIN, N_HO = 50000, 500


def ang_of(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def succ_matrix(pos):
    """-> per blob, the legal successor blobs: within MAX_HOP and no competitor nearer than MIN_ANG (or
    BEHIND_ANG when behind) to the ray. O(K^3) numpy."""
    P = np.asarray(pos, float)
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    a = np.arctan2(P[None, :, 1] - P[:, None, 1], P[None, :, 0] - P[:, None, 0])
    near, behind = math.radians(MIN_ANG), math.radians(BEHIND_ANG)
    out = []
    for b in range(len(pos)):
        A, D = a[b], d[b]
        ad = np.abs(A[:, None] - A[None, :]) % (2 * math.pi)  # (t, c)
        ad = np.minimum(ad, 2 * math.pi - ad)
        lim = np.where(D[None, :] <= D[:, None] * BEHIND_RATIO, near, behind)
        bad = ad < lim
        np.fill_diagonal(bad, False)  # c == t
        bad[:, b] = False  # c == b
        ok = (~bad.any(axis=1)) & (D <= MAX_HOP)
        ok[b] = False
        out.append(np.flatnonzero(ok).tolist())
    return out


def _dart(i, maxrs, pos, rng, tries=600):
    """Re-throw blob i against the current positions of the others (same constraints as S.place)."""
    r_i = maxrs[i]
    for _ in range(tries):
        cx = rng.uniform(S.MARGIN + r_i, W - S.MARGIN - r_i)
        cy = rng.uniform(S.MARGIN + r_i, H - S.MARGIN - r_i)
        if all(math.hypot(cx - x, cy - y) >= r_i + maxrs[j] + GAP for j, (x, y) in enumerate(pos) if j != i):
            return (cx, cy)
    return None


def place_succ(k, maxrs, rng, rounds=14):
    """-> (positions, successor lists) with every blob having >= 1 legal successor, or (None, None).
    Blobs without a successor are re-thrown and legality is recomputed each round."""
    pos = S.place(k, maxrs, rng, W, H, GAP)
    if pos is None:
        return None, None
    for _ in range(rounds):
        val = succ_matrix(pos)
        stuck = [b for b in range(k) if not val[b]]
        if not stuck:
            return pos, val
        for b in stuck:
            p = _dart(b, maxrs, pos, rng)
            if p is None:
                return None, None
            pos[b] = p
    return None, None


def radial_at(geo, ang):
    """World radius of a blob at angle `ang` (upright rendering, uniform NSIG grid)."""
    scale, _, _, sig = geo
    idx = (ang % (2 * math.pi)) / S.DPHI
    return scale * float(np.interp(idx % S.NSIG, np.arange(S.NSIG), sig, period=S.NSIG))


def make_arrow(b, t, pos, geos):
    """Arrow from b's boundary toward t's centre: (base, tip) in px, or None if off-canvas."""
    a = ang_of(pos[b], pos[t])
    r = radial_at(geos[b], a) + 3.0
    base = (pos[b][0] + r * math.cos(a), pos[b][1] + r * math.sin(a))
    tip = (base[0] + ARROW_LEN * math.cos(a), base[1] + ARROW_LEN * math.sin(a))
    for p in (base, tip):
        if not (8 <= p[0] <= W - 8 and 8 <= p[1] <= H - 8):
            return None
    return base, tip


def seg_points(base, tip, n=9):
    return [(base[0] + (tip[0] - base[0]) * i / (n - 1), base[1] + (tip[1] - base[1]) * i / (n - 1)) for i in range(n)]


def seg_clear(arrow, polys, skip):
    """Arrow segment keeps >= ARROW_CLEAR px from every polygon except its source `skip`."""
    pts = np.array(seg_points(*arrow))
    for j, poly in enumerate(polys):
        if j == skip:
            continue
        d2 = ((pts[:, None, :] - poly[None, :, :]) ** 2).sum(-1)
        if math.sqrt(d2.min()) < ARROW_CLEAR:
            return False
    return True


def arrows_apart(arrow, others):
    """Arrow segment keeps >= ARROW_SEP px from every already-placed arrow."""
    pts = np.array(seg_points(*arrow))
    for o in others:
        q = np.array(seg_points(*o))
        d2 = ((pts[:, None, :] - q[None, :, :]) ** 2).sum(-1)
        if math.sqrt(d2.min()) < ARROW_SEP:
            return False
    return True


def sample_shapes(k, rng, tries=400):
    """k mutually dissimilar blob shapes, or None."""
    shapes, sigs = [], []
    for _ in range(k):
        for _t in range(tries):
            cand = S.sample_shape(rng, MODE)
            sig = S.polar_geom(cand, MODE, AREA)[3]
            if all(S.shape_dist(s, sig) >= MIN_SHAPE_D for s in sigs):
                shapes.append(cand)
                sigs.append(sig)
                break
        else:
            return None
    return shapes


def build_chain(q, hops, val, rng):
    """Distinct node sequence chain[0..hops+1] from q along legal successors: DFS with backtracking
    (successor order randomised), capped at 20000 expansions. None if no chain was found."""
    need, budget = hops + 2, [20000]

    def dfs(path):
        if len(path) == need:
            return path
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        for t in rng.permutation(val[path[-1]]):
            t = int(t)
            if t in path:
                continue
            got = dfs(path + [t])
            if got:
                return got
        return None

    return dfs([q])


def build_scene(k, hops, rng, tries=60):
    for _ in range(tries):
        shapes = sample_shapes(k, rng)
        if shapes is None:
            continue
        geos = [S.polar_geom(sh, MODE, AREA) for sh in shapes]
        pos, val = place_succ(k, [g[2] for g in geos], rng)
        if pos is None:
            continue
        polys = [S.world_poly(g[3], g[0], pos[i][0], pos[i][1]) for i, g in enumerate(geos)]
        if S.poly_min_gap(polys) < MIN_POLY_GAP:
            continue

        q = int(rng.integers(0, k))
        chain = build_chain(q, hops, val, rng)
        if chain is None:
            continue

        # chain arrows are mandatory (rebuild the scene if one fails geometry); every other blob gets
        # any legal successor whose arrow clears the geometry gates
        succ, arrows, ok = {}, {}, True
        for i in range(hops + 1):
            b, t = chain[i], chain[i + 1]
            ar = make_arrow(b, t, pos, geos)
            if ar is None or not seg_clear(ar, polys, b) or not arrows_apart(ar, arrows.values()):
                ok = False
                break
            succ[b], arrows[b] = t, ar
        if not ok:
            continue
        for b in range(k):
            if b in succ:
                continue
            for t in rng.permutation(val[b]):
                t = int(t)
                if t == b or succ.get(t) == b:  # no mutual pairs
                    continue
                ar = make_arrow(b, t, pos, geos)
                if ar is not None and seg_clear(ar, polys, b) and arrows_apart(ar, arrows.values()):
                    succ[b], arrows[b] = t, ar
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        if any(succ[succ[b]] == b for b in range(k)):
            continue

        # bid = reading order (y, then x)
        reading = sorted(range(k), key=lambda i: (pos[i][1], pos[i][0]))
        old2new = {o: n for n, o in enumerate(reading)}
        sc = {"k": k, "hops": hops}
        sc["pos"] = [pos[i] for i in reading]
        sc["polys"] = [polys[i] for i in reading]
        sc["succ"] = {old2new[b]: old2new[t] for b, t in succ.items()}
        sc["arrows"] = {old2new[b]: a for b, a in arrows.items()}
        sc["chain"] = [old2new[b] for b in chain]
        sc["query"], sc["gold"] = sc["chain"][0], sc["chain"][hops]
        sc["labels"] = [str(v) for v in rng.choice(np.arange(10, 99), size=k, replace=False)]
        sc["label_pos"] = list(sc["pos"])  # labels sit at the blob centre
        return sc
    return None


def render(sc, labeled, path):
    img = Image.new("RGB", (W * S.SS, H * S.SS), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    for b in range(sc["k"]):
        pts = [(x * S.SS, y * S.SS) for x, y in sc["polys"][b]]
        dr.polygon(pts, fill=S.FILL)
        edge = S.QUERY_OUTLINE if b == sc["query"] else S.OUTLINE
        dr.line(pts + [pts[0]], fill=edge, width=(4 if b == sc["query"] else 3) * S.SS, joint="curve")
    for base, tip in sc["arrows"].values():
        a = math.atan2(tip[1] - base[1], tip[0] - base[0])
        # shaft stops short of the tip so the solid head triangle owns the point
        sx = tip[0] - 7.0 * math.cos(a)
        sy = tip[1] - 7.0 * math.sin(a)
        dr.line([(base[0] * S.SS, base[1] * S.SS), (sx * S.SS, sy * S.SS)], fill=ARROW_INK, width=ARROW_W * S.SS)
        barb = 11.0
        pts = [(tip[0] * S.SS, tip[1] * S.SS)]
        for da in (math.radians(152), -math.radians(152)):
            pts.append(((tip[0] + barb * math.cos(a + da)) * S.SS, (tip[1] + barb * math.sin(a + da)) * S.SS))
        dr.polygon(pts, fill=ARROW_INK)
    if labeled:
        font, _ = S.label_font(LBL_PT * S.SS)
        pad = (13 * LBL_PT / 21, 10 * LBL_PT / 21)
        for b in range(sc["k"]):
            lx, ly = sc["label_pos"][b]
            # a light plate so dark ink stays legible on the gray fill
            dr.rounded_rectangle(
                [((lx - pad[0]) * S.SS, (ly - pad[1]) * S.SS), ((lx + pad[0]) * S.SS, (ly + pad[1]) * S.SS)],
                radius=3 * S.SS,
                fill=(238, 240, 243),
            )
            dr.text((lx * S.SS, ly * S.SS), sc["labels"][b], font=font, fill=S.LABEL_INK, anchor="mm")
    img = img.resize((W, H), Image.LANCZOS)
    img.save(path)


def blob_gt(sc):
    out = []
    for b in range(sc["k"]):
        base, tip = sc["arrows"][b]
        cx, cy = sc["pos"][b]
        out.append(
            {
                "bid": b,
                "label": sc["labels"][b],
                "succ_bid": sc["succ"][b],
                "center": [S.n1000(cx, W), S.n1000(cy, H)],
                "polygon": [[S.n1000(x, W), S.n1000(y, H)] for x, y in sc["polys"][b]],
                "arrow_base": [S.n1000(base[0], W), S.n1000(base[1], H)],
                "arrow_tip": [S.n1000(tip[0], W), S.n1000(tip[1], H)],
                "label_xy": [S.n1000(sc["label_pos"][b][0], W), S.n1000(sc["label_pos"][b][1], H)],
            }
        )
    return out


def mcq_options(sc, bal, rng):
    """gold + foils (hop 1, hop H-1, hop H+1, then random non-query blobs), gold slot balanced."""
    hops, chain, gold_bid = sc["hops"], sc["chain"], sc["gold"]
    foils = []
    for cand in (chain[1], chain[hops - 1], chain[hops + 1]):
        if cand != gold_bid and cand not in foils:
            foils.append(cand)
    pool = [b for b in range(sc["k"]) if b not in foils and b != gold_bid and b != chain[0]]
    while len(foils) < 3:
        pick = int(rng.choice(pool))
        pool.remove(pick)
        foils.append(pick)
    gold = f"blob {sc['labels'][gold_bid]}"
    opts = [f"blob {sc['labels'][b]}" for b in foils]
    slot = min(range(4), key=lambda r: bal[("mcqpos", r)])
    bal[("mcqpos", slot)] += 1
    rng.shuffle(opts)
    return opts[:slot] + [gold] + opts[slot:], gold, foils


def pack(sc, sid, idx0, bal, rng):
    shared = {
        "scene": sid,
        "A": f"K{sc['k']}H{sc['hops']}",
        "K": sc["k"],
        "H_HOPS": sc["hops"],
        "query_bid": sc["query"],
        "gold_bid": sc["gold"],
        "chain_bids": sc["chain"],
        "hop1_bid": sc["chain"][1],
        "query_label": sc["labels"][sc["query"]],
        "gold_label": sc["labels"][sc["gold"]],
        "blobs": blob_gt(sc),
        "hop_probe": HOP_PROBE,
        "enum_probe": ENUM_PROBE,
        "query_probe": QUERY_PROBE,
    }
    q_pt = QUESTION_PT.format(k=sc["k"], h=sc["hops"])
    items = [
        dict(
            idx=idx0,
            image=f"{sid}_plain.png",
            B="PT_L0",
            template="chasepoint",
            question=q_pt,
            options=None,
            answer=None,
            **shared,
        ),
        dict(
            idx=idx0 + 1,
            image=f"{sid}_labeled.png",
            B="PT_L1",
            template="chasepoint",
            question=q_pt,
            options=None,
            answer=None,
            **shared,
        ),
    ]
    opts, gold, foil_bids = mcq_options(sc, bal, rng)
    items.append(
        dict(
            idx=idx0 + 2,
            image=f"{sid}_labeled.png",
            B="MCQ_L1",
            template="chasemcq",
            question=QUESTION_MCQ.format(k=sc["k"], h=sc["hops"]),
            options=opts,
            answer=gold,
            foil_bids=foil_bids,
            **shared,
        )
    )
    return items


def _to_px(p):
    return np.asarray(p, float) * PX


def _ang_deg(x, y):
    d = abs(x - y) % (2 * math.pi)
    return math.degrees(min(d, 2 * math.pi - d))


def _bearing(a, b):
    d = _to_px(b) - _to_px(a)
    return math.atan2(d[1], d[0])


def ambiguous(it):
    """True if some arrow, re-resolved from its STORED base/tip and blob centres, misses its declared successor by
    > 3 deg, or a competitor sits closer to the ray than it owes (1 deg tolerance)."""
    blobs = it["blobs"]
    for b in blobs:
        d = _to_px(b["arrow_tip"]) - _to_px(b["arrow_base"])
        a_arrow = math.atan2(d[1], d[0])
        dev, t = min(
            (_ang_deg(a_arrow, _bearing(b["center"], o["center"])), o["bid"]) for o in blobs if o["bid"] != b["bid"]
        )
        if t != b["succ_bid"] or dev > 3.0:
            return True
        d_t = float(np.hypot(*(_to_px(blobs[t]["center"]) - _to_px(b["center"]))))
        for c in blobs:
            if c["bid"] in (b["bid"], t):
                continue
            d_c = float(np.hypot(*(_to_px(c["center"]) - _to_px(b["center"]))))
            need = MIN_ANG if d_c <= d_t * BEHIND_RATIO else BEHIND_ANG
            if _ang_deg(a_arrow, _bearing(b["center"], c["center"])) < need - 1.0:
                return True
    return False


def make_shard(job):
    """One shard (items.jsonl, images/, meta.json) under OUT; ambiguous scenes are dropped."""
    out, prefix, hops, seed, n = job
    out = os.path.join(OUT, out)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    rng = np.random.default_rng(seed)
    bal = Counter()
    scenes, items = [], []
    while len(scenes) < n:
        sc = build_scene(K, hops, rng)
        if sc is not None:
            sid = f"{prefix}{len(scenes):05d}"
            scenes.append((sid, sc))
            items.extend(pack(sc, sid, len(items), bal, rng))
    bad = sorted({it["scene"] for it in items if it["B"] == "PT_L0" and ambiguous(it)})
    items = [it for it in items if it["scene"] not in bad]
    for i, it in enumerate(items):
        it["idx"] = i
    for sid, sc in scenes:
        if sid not in bad:
            render(sc, False, os.path.join(out, "images", f"{sid}_plain.png"))
            render(sc, True, os.path.join(out, "images", f"{sid}_labeled.png"))
    meta = {
        "seed": seed,
        "per_cell": n,
        "n_scenes": n - len(bad),
        "n_items": len(items),
        "W": W,
        "H": H,
        "k": K,
        "h_levels": [2, 4, 6],
        "preset": PRESET,
        "area": AREA,
        "gap": GAP,
        "poly_gap": MIN_POLY_GAP,
        "max_hop": MAX_HOP,
        "min_shape_d": MIN_SHAPE_D,
        "min_ang_deg": MIN_ANG,
        "behind_ang_deg": BEHIND_ANG,
        "behind_ratio": BEHIND_RATIO,
        "label_pt": LBL_PT,
        "arrow_len": ARROW_LEN,
        "arrow_clear": PRESET["arrow_clear"],
        "arrow_sep": ARROW_SEP,
        "arrow_ink": ARROW_INK,
        "question_pt": QUESTION_PT,
        "question_mcq": QUESTION_MCQ,
    }
    if bad:
        meta.update(pruned=bad, prune_reason="ambiguous arrow geometry after rounding, see generate.ambiguous")
    write_dir(out, items, meta)
    return out


def write_dir(d, items, meta):
    with open(os.path.join(d, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def merge(shards):
    """-> (items sorted by (scene, protocol), meta of the first shard + counts, {scene: shard dir})."""
    items, src = [], {}
    for sh in shards:
        with open(os.path.join(sh, "items.jsonl")) as f:
            for it in map(json.loads, f):
                items.append(it)
                src[it["scene"]] = sh
    items.sort(key=lambda it: (it["scene"], it["B"]))
    for i, it in enumerate(items):
        it["idx"] = i
    with open(os.path.join(shards[0], "meta.json")) as f:
        meta = json.load(f)
    meta.update(n_scenes=len(src), n_items=len(items), merged_from=[os.path.relpath(sh, OUT) for sh in shards])
    return items, meta, src


def write_split(d, items, meta, src):
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    for i, it in enumerate(items):
        it["idx"] = i
        if it["B"] == "PT_L0":
            for suff in ("_plain.png", "_labeled.png"):
                name = it["scene"] + suff
                if not os.path.exists(os.path.join(d, "images", name)):
                    os.link(os.path.join(src[it["scene"]], "images", name), os.path.join(d, "images", name))
    write_dir(d, items, meta)


def assemble(k48):
    """Merge the K=48 shards and split them per H by sorted scene id (the first N_TRAIN/3 scenes of each H
    train, the next N_HO/3 held out)."""
    items, meta, src = merge(k48)
    by_scene = defaultdict(list)
    for it in items:
        by_scene[it["scene"]].append(it)
    by_cell = defaultdict(list)
    for sid in sorted(by_scene):
        by_cell[by_scene[sid][0]["A"]].append(sid)
    tr, ho = [], []
    for ci, cell in enumerate(sorted(by_cell)):
        n_tr = N_TRAIN // len(by_cell) + (ci < N_TRAIN % len(by_cell))
        n_ho = N_HO // len(by_cell) + (ci < N_HO % len(by_cell))
        tr += by_cell[cell][:n_tr]
        ho += by_cell[cell][n_tr : n_tr + n_ho]
    for name, sids in (("train", tr), ("heldout", ho)):
        rows = [it for sid in sids for it in by_scene[sid]]
        write_split(os.path.join(OUT, name), rows, dict(meta, split=name, n_scenes=len(sids), n_items=len(rows)), src)


def generate():
    """All shards on every core, then train / heldout."""
    with Pool(len(os.sched_getaffinity(0))) as pool:
        assemble(pool.map(make_shard, K48_SHARDS, 1))


# ---------------------------------------------------------------- sft
# The chain-of-thought trace is a hop ledger:
#   Hop one: the arrow at {POS} points down and to the left, landing on the blob at {POS'}. I am now at {POS'}.
# Styles: direct (\boxed{[x, y]} of the landing blob only), cot (3x3 cells + within-cell qualifiers), ground (blob
# centres [x, y]), mark (the printed label, "blob 37").
SUFFIXES = {
    **SHARED_SUFFIXES,
    "mark": THINK_HEAD + "refer to each relevant blob by the number printed on it, then analyze those labels "
    "step-by-step.",
}
STYLES = {  # reasoning format -> trace style, on plain (False) and labeled (True) images
    False: {"direct": "direct", "text": "cot", "grounded": "ground"},
    True: {"direct": "direct", "text": "mark", "grounded": "ground"},
}
PROTOCOL = {False: "PT_L0", True: "PT_L1"}  # item "B" field: same question on the plain / labeled render
STEP_LINE = {
    "cot": "Let's analyze these descriptions step-by-step.",
    "ground": "Let's analyze these coordinates step-by-step.",
    "mark": "Let's analyze these labels step-by-step.",
}
OPENING = (
    "Let me think about what I need to locate — I need to locate the blue-outlined "
    "starting blob, then follow the arrows one hop at a time, keeping track of where "
    "I am after each hop."
)
HOPWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
DIR8 = [  # y grows downward, index 0 = +x
    "to the right", "down and to the right", "downward", "down and to the left",
    "to the left", "up and to the left", "upward", "up and to the right",
]  # fmt: skip


def prompt(it, style):
    """Text prompt, shared by SFT and eval."""
    return it["question"] + SUFFIXES[style]


def dir_word(a, b):
    d = (np.asarray(b, float) - np.asarray(a, float)) * PX
    return DIR8[int((math.atan2(d[1], d[0]) / (2 * math.pi) * 8) + 8.5) % 8]


def trace(it, style):
    blobs = {b["bid"]: b for b in it["blobs"]}
    chain, hops = it["chain_bids"], it["H_HOPS"]
    if style == "mark":
        name = lambda bid: f"blob {blobs[bid]['label']}"
        at = name
    else:
        name = build_refs(it["blobs"], style == "ground").__getitem__
        at = lambda bid: f"the blob at {name(bid)}"
    L = [OPENING, "", f"The start is {at(chain[0])}.", "", STEP_LINE[style], ""]
    for i in range(hops):
        prev, nxt = chain[i], chain[i + 1]
        dw = dir_word(blobs[prev]["center"], blobs[nxt]["center"])
        src = f"the arrow of {name(prev)}" if style == "mark" else f"the arrow at {name(prev)}"
        L.append(f"Hop {HOPWORD[i + 1]}: {src} points {dw}, landing on {at(nxt)}. I am now at {name(nxt)}.")
    L += ["", f"After {HOPWORD[hops]} hops I land on {at(chain[hops])}."]
    if style == "ground":
        L.append(f"So the answer is the point {ipt(blobs[it['gold_bid']]['center'])}.")
    else:
        L.append(f"So the answer is {at(chain[hops])}.")
    return "\n".join(L)


def row(it, style, image_root):
    gold = next(b for b in it["blobs"] if b["bid"] == it["gold_bid"])
    box = f"\\boxed{{{ipt(gold['center'])}}}"
    target = box if style == "direct" else f"<think>\n{trace(it, style)}\n</think>\n\n{box}"
    return {
        "image": os.path.join(image_root, it["image"]),
        "chase_arm": style,
        "chase_scene": it["scene"],
        "chase_cell": it["A"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt(it, style)}"},
            {"from": "gpt", "value": target},
        ],
    }


def build_sft(data, protocol, out_paths):
    """Items of one render (protocol) of a split -> one json per trace style ({style: path}); image paths relative
    to $GR_DATA."""
    items = []
    with open(os.path.join(data, "items.jsonl")) as f:  # streamed: the 50k split is ~15 GB
        for line in f:
            if f'"B": "{protocol}"' in line[:400]:
                items.append(json.loads(line))
    image_root = os.path.relpath(os.path.join(data, "images"), DATA)
    for style, path in out_paths.items():
        write_sft(path, [row(it, style, image_root) for it in items])
        print(f"{len(items)} rows -> {path}")


def prep_sft():
    for labeled in (False, True):
        build_sft(
            data_dir(TASK) / "train",
            PROTOCOL[labeled],
            {style: sft_path(TASK, fmt, labeled) for fmt, style in STYLES[labeled].items()},
        )


# ---------------------------------------------------------------- eval
# Greedy decoding on the plain (PT_L0) or labeled (PT_L1) render of heldout (500 scenes, H = 2/4/6). The first [x, y]
# after \boxed{ (else the first in the text) is hit-tested against the blob polygons.
BATCH_SIZE = 8
MAX_NEW_TOKENS = 2048
PT = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
BOXED = re.compile(r"\\boxed\{")


def point_in_poly(x, y, poly):
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def score(it, gen):
    m = BOXED.search(gen)
    pts = PT.findall(gen[m.end() :] if m else gen) or PT.findall(gen)
    if not pts:
        return {"pred": None, "hit": False}
    x, y = float(pts[0][0]), float(pts[0][1])
    landed = next((b for b in it["blobs"] if point_in_poly(x, y, b["polygon"])), None)
    return {"pred": [x, y], "hit": landed is not None and landed["bid"] == it["gold_bid"]}


def evaluate(args):
    """-> one row per item (scene, H_HOPS, pred, hit, resp) in $GR_RESULTS/probes/arrow_following/heldout/<run>.jsonl."""
    style = STYLES[args.labeled][args.format]
    data = data_dir(TASK) / "heldout"
    model_dir = str(args.model or checkpoint_dir(TASK, args.format, args.labeled))
    out_path = args.out or results_dir(TASK) / "heldout" / f"{run_name(args.format, args.labeled)}.jsonl"

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    with open(data / "items.jsonl") as f:
        items = [it for it in map(json.loads, f) if it["B"] == PROTOCOL[args.labeled]]
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", trust_remote_code=True
    ).eval()
    torch.manual_seed(0)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    hits = []
    with open(out_path, "w") as rf, torch.no_grad():
        for lo in range(0, len(items), BATCH_SIZE):
            chunk = items[lo : lo + BATCH_SIZE]
            images = [Image.open(data / "images" / it["image"]).convert("RGB") for it in chunk]
            msgs = [
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(it, style)}]}]
                for it in chunk
            ]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            inp = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
            out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            gens = processor.tokenizer.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)
            for it, g in zip(chunk, gens):
                rec = {"scene": it["scene"], "H_HOPS": it["H_HOPS"], **score(it, g), "resp": g}
                hits.append(rec["hit"])
                rf.write(json.dumps(rec) + "\n")
    print(f"{out_path}: n={len(hits)} acc={sum(hits) / len(hits):.4f}")


def main():
    ap = argparse.ArgumentParser(prog="python -m grounded_reasoning.probes.arrow_following")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="$GR_DATA/probes/arrow_following/{train,heldout}/ (CPU, every core)")
    sub.add_parser("sft", help="$GR_DATA/probes/arrow_following/sft/<run>.json")
    ev = sub.add_parser("eval", help="greedy eval of one checkpoint (1 GPU)")
    ev.add_argument("--format", required=True, choices=FORMATS)
    ev.add_argument("--labeled", action="store_true", help="labeled images (a number printed on every blob)")
    ev.add_argument("--model", help="default: $GR_RUNS/probes/arrow_following_<run>")
    ev.add_argument("--out", help="default: $GR_RESULTS/probes/arrow_following/heldout/<run>.jsonl")
    sub.add_parser("summarize", help="$GR_RESULTS/probes/arrow_following/summary.json")
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
