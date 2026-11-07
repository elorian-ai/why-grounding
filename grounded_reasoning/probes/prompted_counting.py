"""Visually prompted counting: K in {29, 35, 41} gray shapes, one outlined in blue; how many of the others are exact
copies of it (M in {9..12})? Data: $GR_DATA/probes/prompted_counting/, results: $GR_RESULTS/probes/prompted_counting/.

  python -m grounded_reasoning.probes.prompted_counting generate | sft | summarize
  python -m grounded_reasoning.probes.prompted_counting eval --format F [--labeled]
"""

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

from grounded_reasoning import DATA
from grounded_reasoning.probes import common as S
from grounded_reasoning.probes.common import (
    FORMATS,
    SUFFIXES,
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

TASK = "prompted_counting"
OUT = data_dir(TASK)

# ---------------------------------------------------------------- generate
# Decoy groups of identical non-referent shapes can outnumber the matches; every non-match is >= FAR_D away in
# contour distance. The _labeled twin prints a 2-character mark next to every shape.
MODE = "poly"
W, H = 1600, 1120
AREA = 3000.0  # px^2 per shape
GAP = 18.0  # min centre-line boundary gap at placement
MIN_POLY_GAP = 10.0  # exact polygon-to-polygon floor
K_LEVELS = (29, 35, 41)
M_LEVELS = (9, 10, 11, 12)
FAR_D = 0.16  # every non-match at least this far from the referent
PAIR_D = 0.08  # distinct non-match shapes at least this far from each other
N_DECOY_GROUPS = (1, 2)
DECOY_OVERSHOOT_P = 0.80  # P(one decoy group has m + 2 members, more than the referent and its m copies)
PER_M = 1376  # scenes per answer level, spread evenly over K
SHARD_ORDER = (10, 11, 12, 9)  # one RNG stream per M (seed 20260950 + M); scene ids run in this order
N_TRAIN, N_HO = 5000, 500

QUESTION = (
    "This image shows {k} gray shapes. One of them is outlined in blue. How many of the "
    "OTHER shapes have exactly the same shape as the blue-outlined one? Answer with a "
    "number."
)
ENUM_PROBE = (
    "Point at every shape that has exactly the same shape as the blue-outlined one, "
    "one point per shape, as [x, y] normalized to 0-1000."
)

# labeled render: marks A1..A9, B1..B9, ... (first K) assigned in random order, placed next to their shape
LETTERS, DIGITS = "ABCDEFGH", "123456789"
LABEL_PT = 21  # font px at 1x
FONT_WH = (32, 26)  # 2-char label bbox at 1x (slight over-estimate)
TIERS = (22.0, 14.0, 30.0)  # label-centre offsets beyond the object's max radius
N_ANG = 20
CLEAR_PX = 8.0  # min label-box corner distance to any other polygon
OWN_MARGIN = 4.0  # own polygon must be this much closer than any other
LABEL_SEED = 20260801


def pick_distinct(n, ref_sig, rng):
    """n shapes >= FAR_D from the referent and >= PAIR_D from each other, or None."""
    distinct, sigs = [], []
    for _ in range(n):
        for _t in range(400):
            cand = S.sample_shape(rng, MODE)
            sig = S.polar_geom(cand, MODE, AREA)[3]
            if S.shape_dist(ref_sig, sig) >= FAR_D and all(S.shape_dist(s, sig) >= PAIR_D for s in sigs):
                distinct.append(cand)
                sigs.append(sig)
                break
        else:
            return None
    return distinct


def build_scene(k, m, rng, tries=200):
    """1 referent + m exact copies + (k - 1 - m) non-matches (some in identical decoy groups), or None."""
    for _ in range(tries):
        ref = S.sample_shape(rng, MODE)
        ref_sig = S.polar_geom(ref, MODE, AREA)[3]
        n_groups = int(rng.integers(N_DECOY_GROUPS[0], N_DECOY_GROUPS[1] + 1))
        sizes, budget = [], k - 1 - m
        force_big = rng.random() < DECOY_OVERSHOOT_P
        for gi in range(n_groups):
            hi = min(budget, m + 2)
            if hi < 2:
                break
            lo = m + 2 if (force_big and gi == 0 and budget >= m + 2) else 2
            sz = int(rng.integers(lo, hi + 1))
            sizes.append(sz)
            budget -= sz
        distinct = pick_distinct(len(sizes) + budget, ref_sig, rng)  # one shape per decoy group + the singles
        if distinct is None:
            continue
        shapes = [ref] + [ref] * m
        roles = ["referent"] + ["match"] * m
        for i, sh in enumerate(distinct):
            n = sizes[i] if i < len(sizes) else 1
            shapes += [sh] * n
            roles += ["decoy_group" if i < len(sizes) else "filler"] * n
        if len(shapes) != k:
            continue
        order = list(rng.permutation(k))
        shapes = [shapes[i] for i in order]
        roles = [roles[i] for i in order]
        geos = [S.polar_geom(s, MODE, AREA) for s in shapes]
        pos = S.place(k, [g[2] for g in geos], rng, W, H, GAP)
        if pos is None:
            continue
        polys = [S.world_poly(g[3], g[0], pos[i][0], pos[i][1]) for i, g in enumerate(geos)]
        if S.poly_min_gap(polys) < MIN_POLY_GAP:
            continue
        reading = sorted(range(k), key=lambda i: (round(pos[i][1] / 180), pos[i][0]))  # 180 px row bands, then x
        roles = [roles[i] for i in reading]
        return {
            "k": k,
            "m": m,
            "decoy_sizes": sorted(sizes, reverse=True),
            "pos": [pos[i] for i in reading],
            "polys": [polys[i] for i in reading],
            "roles": roles,
            "ref": roles.index("referent"),
            "matches": [i for i, r in enumerate(roles) if r == "match"],
        }
    return None


def gen_shard(m):
    """All PER_M scenes of answer level m, in generation order (spread evenly over the feasible K)."""
    ks = [k for k in K_LEVELS if k - 1 - m >= m + 2]  # a decoy group larger than 1 + m must fit
    base, rem = divmod(PER_M, len(ks))
    rng = np.random.default_rng(20260950 + m)
    scenes = []
    for j, k in enumerate(ks):
        made = 0
        while made < base + (j < rem):
            sc = build_scene(k, m, rng)
            if sc is not None:
                scenes.append(sc)
                made += 1
    return scenes


def draw_shapes(polys, ref):
    img = Image.new("RGB", (W * S.SS, H * S.SS), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    for b, poly in enumerate(polys):
        pts = [(x * S.SS, y * S.SS) for x, y in poly]
        dr.polygon(pts, fill=S.FILL)
        is_ref = b == ref
        dr.line(
            pts + [pts[0]],
            fill=S.QUERY_OUTLINE if is_ref else S.OUTLINE,
            width=(4 if is_ref else 3) * S.SS,
            joint="curve",
        )
    return img, dr


def render_plain(job):
    sc, path = job
    img, _ = draw_shapes(sc["polys"], sc["ref"])
    img.resize((W, H), Image.LANCZOS).save(path)


def pack(sc, sid):
    objs = [
        {
            "bid": b,
            "role": sc["roles"][b],
            "is_referent": b == sc["ref"],
            "is_match": sc["roles"][b] == "match",
            "center": [S.n1000(sc["pos"][b][0], W), S.n1000(sc["pos"][b][1], H)],
            "polygon": [[S.n1000(x, W), S.n1000(y, H)] for x, y in sc["polys"][b]],
        }
        for b in range(sc["k"])
    ]
    return {
        "idx": 0,
        "image": f"{sid}.png",
        "B": "COUNT",
        "template": "countmatch",
        "question": QUESTION.format(k=sc["k"]),
        "options": None,
        "answer": str(sc["m"]),
        "scene": sid,
        "A": f"K{sc['k']}M{sc['m']}",
        "K": sc["k"],
        "M": sc["m"],
        "decoy_group_sizes": sc["decoy_sizes"],
        "largest_decoy_group": (max(sc["decoy_sizes"]) if sc["decoy_sizes"] else 0),
        "referent_bid": sc["ref"],
        "match_bids": sc["matches"],
        "match_points": [objs[b]["center"] for b in sc["matches"]],
        "objects": objs,
        "enum_probe": ENUM_PROBE,
    }


def split(items):
    """Stratified by (K, M) cell: a held-out quota per cell, surplus back to train, then shuffled."""
    by_cell = collections.defaultdict(list)
    for it in items:
        by_cell[it["A"]].append(it)
    frac = N_HO / (N_TRAIN + N_HO)
    tr, ho = [], []
    for c in sorted(by_cell):
        n_ho = max(1, round(len(by_cell[c]) * frac))
        ho += by_cell[c][:n_ho]
        tr += by_cell[c][n_ho:]
    if len(ho) > N_HO:
        tr += ho[N_HO:]
        ho = ho[:N_HO]
    random.Random(0).shuffle(tr)
    random.Random(1).shuffle(ho)
    return tr[:N_TRAIN], ho[:N_HO]


def scene_rng(scene):
    h = int.from_bytes(hashlib.sha1(scene.encode()).digest()[:8], "big")
    return np.random.default_rng((h ^ (LABEL_SEED * 0x9E3779B97F4A7C15)) & ((1 << 63) - 1))


def poly_dist(pt, poly):
    """Min distance from a point to the polygon's vertices (px)."""
    d = poly - np.asarray(pt, float)
    return math.sqrt(float((d**2).sum(axis=1).min()))


def in_poly(pt, poly):
    x, y = pt
    j, inside = len(poly) - 1, False
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def label_spot(bi, polys, centers, maxrs, taken, rng):
    """Label anchor (px) for object bi: in-canvas, no box corner inside any polygon, >= 2 px from its own
    and >= CLEAR_PX from every other polygon, OWN_MARGIN closer to its own, no overlap with placed labels.
    Candidates are snapped to the stored 0.1-unit grid before they are checked."""
    fw, fh = FONT_WH
    cx, cy = centers[bi]
    for tier in TIERS:
        for ang in rng.permutation(np.linspace(0, 2 * np.pi, N_ANG, endpoint=False)):
            lx = cx + (maxrs[bi] + tier) * math.cos(ang)
            ly = cy + (maxrs[bi] + tier) * math.sin(ang)
            lx = round(1000.0 * lx / W, 1) * W / 1000.0
            ly = round(1000.0 * ly / H, 1) * H / 1000.0
            x0, y0, x1, y1 = lx - fw / 2 - 3, ly - fh / 2 - 2, lx + fw / 2 + 3, ly + fh / 2 + 2
            if x0 < 6 or y0 < 6 or x1 > W - 6 or y1 > H - 6:
                continue
            corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1], [lx, ly]])
            if any(in_poly(c, polys[j]) for c in corners for j in range(len(polys))):
                continue
            own = min(poly_dist(c, polys[bi]) for c in corners)
            if own < 2.0:
                continue
            ok = True
            for j, poly in enumerate(polys):
                if j == bi:
                    continue
                dj = min(poly_dist(c, poly) for c in corners)
                if dj < CLEAR_PX or dj < own + OWN_MARGIN:
                    ok = False
                    break
            if ok and not any(abs(lx - tx) < fw + 8 and abs(ly - ty) < fh + 8 for tx, ty in taken):
                return lx, ly
    return None


def label_scene(job):
    """(item, image dir) -> labeled item (render written), or None if some object has no legal label spot."""
    it, img_dir = job
    rng = scene_rng(it["scene"])
    k = it["K"]
    roster = [f"{LETTERS[i // 9]}{DIGITS[i % 9]}" for i in range(k)]
    marks = list(rng.permutation(roster))
    polys = [np.asarray(o["polygon"], float) * [W / 1000.0, H / 1000.0] for o in it["objects"]]
    cents = [[o["center"][0] * W / 1000.0, o["center"][1] * H / 1000.0] for o in it["objects"]]
    maxrs = [float(np.hypot(*(p - c).T).max()) for p, c in zip(polys, cents)]
    taken = []
    for b in range(k):
        sp = label_spot(b, polys, cents, maxrs, taken, rng)
        if sp is None:
            return None
        taken.append(sp)
    img, dr = draw_shapes(polys, it["referent_bid"])
    font, _ = S.label_font(LABEL_PT * S.SS)
    for b, (lx, ly) in enumerate(taken):
        dr.text((lx * S.SS, ly * S.SS), marks[b], font=font, fill=S.LABEL_INK, anchor="mm")
    img.resize((W, H), Image.LANCZOS).save(os.path.join(img_dir, it["image"]))
    rec = dict(it)
    rec["objects"] = [
        dict(o, mark=marks[b], label_pos=[S.n1000(taken[b][0], W), S.n1000(taken[b][1], H)])
        for b, o in enumerate(it["objects"])
    ]
    rec["mark_roster"] = roster
    rec["referent_mark"] = marks[it["referent_bid"]]
    rec["match_marks"] = [marks[b] for b in it["match_bids"]]
    return rec


def write_dir(d, items, meta):
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    with open(os.path.join(d, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def generate(out=OUT):
    """train (5,000 scenes) and heldout (500), each with its _labeled twin, on every core."""
    with Pool(len(os.sched_getaffinity(0))) as pool:
        shards = pool.map(gen_shard, SHARD_ORDER)
        scenes = [sc for shard in shards for sc in shard]
        items = [pack(sc, f"cmh_{i:06d}") for i, sc in enumerate(scenes)]  # ids seed the label RNG: keep them
        by_scene = dict(zip((it["scene"] for it in items), scenes))
        pool_meta = {  # the merged pool keeps the header of its first shard (M = SHARD_ORDER[0])
            "per_m": PER_M,
            "n_scenes": len(items),
            "n_items": len(items),
            "cells": [f"K{k}M{SHARD_ORDER[0]}" for k in K_LEVELS],
            "W": W,
            "H": H,
            "area": AREA,
            "shape_mode": MODE,
            "n_corners": list(S.N_CORNERS),
            "far_d": FAR_D,
            "pair_d": PAIR_D,
            "n_decoy_groups": list(N_DECOY_GROUPS),
            "decoy_overshoot_p": DECOY_OVERSHOOT_P,
            "rotate": False,
            "k_levels": list(K_LEVELS),
            "m_levels": list(M_LEVELS),
            "min_poly_gap": MIN_POLY_GAP,
            "question": QUESTION,
            "shard_seeds": {f"M{m}": 20260950 + m for m in SHARD_ORDER},
            "merged_from": [f"M{m}" for m in SHARD_ORDER],
        }
        for name, rows in zip(("train", "heldout"), split(items)):
            src = os.path.join(out, name)
            for i, it in enumerate(rows):
                it["idx"] = i
            meta = dict(pool_meta, split=name, n_scenes=len(rows), n_items=len(rows))
            write_dir(src, rows, meta)
            pool.map(render_plain, [(by_scene[it["scene"]], os.path.join(src, "images", it["image"])) for it in rows])

            dst = os.path.join(out, f"{name}_labeled")
            os.makedirs(os.path.join(dst, "images"), exist_ok=True)
            labeled = [r for r in pool.map(label_scene, [(it, os.path.join(dst, "images")) for it in rows], 16) if r]
            meta.update(
                build="labeled",
                src=name,
                seed=LABEL_SEED,
                label_pt=LABEL_PT,
                label_font=S.label_font(LABEL_PT)[1],
                font_wh=list(FONT_WH),
                mark_scheme="letters ABCDEFGH x digits 1-9, dense prefix, random assignment",
                n_scenes=len(labeled),
                n_items=len(labeled),
            )
            write_dir(dst, labeled, meta)
            print(f"{name}: {len(rows)} scenes, {len(labeled)} labeled -> {src}, {dst}")


# ---------------------------------------------------------------- sft
# direct: \boxed{M} only. cot / ground visit the K-1 other shapes in reading order, compare each with the referent
# and keep a running tally, positions as 3x3 cells / [x, y]; key / keyg do the same over the printed marks in roster
# order, "mark A1" / "mark A1 [x, y]".
STYLES = {  # reasoning format -> trace style, on plain (False) and labeled (True) images
    False: {"direct": "direct", "text": "cot", "grounded": "ground"},
    True: {"direct": "direct", "text": "key", "grounded": "keyg"},
}
NUMW = {
    0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen",
}  # fmt: skip
SUFFIX_OF = {"direct": "direct", "cot": "cot", "ground": "ground", "key": "cot", "keyg": "ground"}  # style -> prompt


def prompt(it, style):
    """Text prompt, shared by SFT and eval."""
    return it["question"] + SUFFIXES[SUFFIX_OF[style]]


def trace(it, style):
    """<think> body: locate the referent, one comparison line per visited shape, the final count."""
    if style in ("key", "keyg"):
        ref = it["objects"][it["referent_bid"]]
        by_mark = {o["mark"]: o for o in it["objects"]}
        walk = [by_mark[m] for m in it["mark_roster"] if m != ref["mark"]]
        refs = {
            o["bid"]: f"mark {o['mark']}" + (f" {ipt(o['center'])}" if style == "keyg" else "") for o in it["objects"]
        }
    else:
        walk = [o for o in it["objects"] if o["bid"] != it["referent_bid"]]
        refs = build_refs(it["objects"], style == "ground")
    ref_pos = refs[it["referent_bid"]]
    lines, seen = [f"I find the reference shape at {ref_pos}.", ""], 0
    for o in walk:
        if o["is_match"]:
            seen += 1
            lines.append(
                f"Looking at {refs[o['bid']]} and comparing to {ref_pos}, the shapes are the same — that makes {NUMW[seen]}."
            )
        else:
            lines.append(f"Looking at {refs[o['bid']]} and comparing to {ref_pos}, the shapes are different.")
    lines += ["", f"So {NUMW[seen]} of the other shapes match the reference shape."]
    return "\n".join(lines)


def row(it, style, image_root):
    box = f"\\boxed{{{it['M']}}}"
    target = box if style == "direct" else f"<think>\n{trace(it, style)}\n</think>\n\n{box}"
    return {
        "image": os.path.join(image_root, it["image"]),
        "cm_arm": style,
        "cm_scene": it["scene"],
        "cm_cell": it["A"],
        "cm_M": it["M"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt(it, style)}"},
            {"from": "gpt", "value": target},
        ],
    }


def build_sft(data, out_paths):
    """Items of one split -> one json per trace style ({style: path}); image paths relative to $GR_DATA."""
    with open(os.path.join(data, "items.jsonl")) as f:
        items = [json.loads(line) for line in f]
    image_root = os.path.relpath(os.path.join(data, "images"), DATA)
    for style, path in out_paths.items():
        write_sft(path, [row(it, style, image_root) for it in items])
        print(f"{len(items)} rows -> {path}")


def prep_sft():
    for labeled in (False, True):
        build_sft(
            data_dir(TASK) / ("train_labeled" if labeled else "train"),
            {style: sft_path(TASK, fmt, labeled) for fmt, style in STYLES[labeled].items()},
        )


# ---------------------------------------------------------------- eval
# Greedy decoding on the 500 held-out scenes; the answer is the last \boxed{N}, else the final
# "So N of the other shapes match".
BATCH_SIZE = 4
MAX_NEW_TOKENS = 6144
BOXED = re.compile(r"\\boxed\{\s*(\d+)\s*\}")
FINALW = re.compile(r"So (\w+) of the other shapes match|So none of the other shapes match")
WORD2N = {v: k for k, v in NUMW.items()}


def score(it, gen):
    m = BOXED.findall(gen)
    f = FINALW.search(gen)
    if m:
        pred = int(m[-1])
    elif f:
        pred = WORD2N.get(f.group(1), 0) if f.group(1) else 0
    else:
        pred = None
    return {"pred": pred, "hit": pred == it["M"]}


def evaluate(args):
    """-> one row per item (scene, K, M, pred, hit, resp) in $GR_RESULTS/probes/prompted_counting/heldout/<run>.jsonl"""
    style = STYLES[args.labeled][args.format]
    data = data_dir(TASK) / ("heldout_labeled" if args.labeled else "heldout")
    model_dir = str(args.model or checkpoint_dir(TASK, args.format, args.labeled))
    out_path = args.out or results_dir(TASK) / "heldout" / f"{run_name(args.format, args.labeled)}.jsonl"

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    with open(data / "items.jsonl") as f:
        items = [json.loads(line) for line in f]
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", trust_remote_code=True
    ).eval()
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
                rec = {"scene": it["scene"], "K": it["K"], "M": it["M"], **score(it, g), "resp": g}
                hits.append(rec["hit"])
                rf.write(json.dumps(rec) + "\n")
    print(f"{out_path}: n={len(hits)} acc={sum(hits) / len(hits):.4f}")


def main():
    ap = argparse.ArgumentParser(prog="python -m grounded_reasoning.probes.prompted_counting")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="$GR_DATA/probes/prompted_counting/{train,heldout}[_labeled]/ (CPU, every core)")
    sub.add_parser("sft", help="$GR_DATA/probes/prompted_counting/sft/<run>.json")
    ev = sub.add_parser("eval", help="greedy held-out eval of one checkpoint (1 GPU)")
    ev.add_argument("--format", required=True, choices=FORMATS)
    ev.add_argument("--labeled", action="store_true", help="labeled images (a printed mark next to every shape)")
    ev.add_argument("--model", help="default: $GR_RUNS/probes/prompted_counting_<run>")
    ev.add_argument("--out", help="default: $GR_RESULTS/probes/prompted_counting/heldout/<run>.jsonl")
    sub.add_parser("summarize", help="$GR_RESULTS/probes/prompted_counting/summary.json")
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
