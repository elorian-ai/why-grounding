"""The four BrickVision generators: seeded brick scenes -> <out>/{samples/*.png, masks/*.png, items.jsonl,
scenes.jsonl, meta.json} (Counting has no masks; its answer is an integer).

  python -m brickvision.generate {crop_id,counting,duplicate,unpaired} --seed S --out DIR [--n-items N (counting)]
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from brickvision import render as G

N_ITEMS = 250
PALETTE = tuple(G.COLORS)
OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))  # 2x2 middle brick on the 3x3 base
GEOM_ATTRS = ("middle_offset", "bar_orientation", "bar_slot", "top_slot")


def write_outputs(out, items, scenes, meta):
    (out / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    (out / "scenes.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scenes))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------------------------------------------------
# Crop ID: ten five-piece structures share one color per role and differ only in the end marker's offset (3) and the
# cap's slot (8). The inset shows the target with 2-3 pieces blacked out, one outlined in magenta. 50 items per role.

CROP_FAMILY = "croppiece_geo"
CROP_PER_ROLE = 50
CROP_ROLES = ("foundation", "pedestal", "candidate", "side_marker", "cap")
MARKER_OFFSETS = (0, 1, 2)  # along the pedestal's 3-stud side
CAP_POSITIONS = [(cx, cy) for cx in range(4) for cy in range(2)]  # on the 4x2 candidate
CHARCOAL, SEAM, MAGENTA = (45, 45, 45), (120, 120, 120), (255, 0, 220)
BACKGROUND = (247, 247, 250)
CROP_QUESTION = (
    "The inset on the right shows one structure from the scene, enlarged. In the inset, "
    "some pieces are blacked out and appear only as dark silhouettes; the other pieces "
    "keep their colors. Exactly one of the blacked-out pieces is outlined in magenta. "
    "Using the visible colors together with the shapes and positions of all the pieces, "
    "locate this exact structure in the full scene on the left, then point to the piece "
    "in the full scene that corresponds to the outlined blacked-out piece. Point in the "
    "full scene, not in the inset. Return exactly one point and no other points."
    '\n\nReturn only a JSON list in this exact form: [{"point": [y, x]}]. Gemini '
    "coordinate order is [y, x], NOT [x, y]. Integer coordinates are normalized to "
    "0-1000 of image height and width, origin at top-left. No labels, prose, or Markdown."
)


def build_crop_scene(seed, global_index, attempt):
    """Scene + task dict: 10 pods whose target (offset, cap) pair is unique, its offset shared by >= 2 other pods
    and its cap slot by >= 1."""
    rng = random.Random(seed * 9_000_011 + global_index * 1_009 + attempt * 7_717)
    themes = list(G.PLATE_THEMES)
    scene = G.blank_scene(seed, global_index, CROP_FAMILY, themes[global_index % len(themes)], None)
    palette = list(G.COLORS)
    rng.shuffle(palette)
    colors = dict(zip(CROP_ROLES, palette[:5]))  # one color per role, scene-wide
    combos = [(mo, cp) for mo in MARKER_OFFSETS for cp in CAP_POSITIONS]
    target = rng.choice(combos)
    same_offset = [c for c in combos if c[0] == target[0] and c != target]
    same_cap = [c for c in combos if c[1] == target[1] and c != target]
    chosen = {target, *rng.sample(same_offset, 2), *rng.sample(same_cap, 1)}
    chosen |= set(rng.sample([c for c in combos if c not in chosen], 10 - len(chosen)))
    designs = list(chosen)
    rng.shuffle(designs)
    records = []
    for rank, ((x, y), (mo, (cx, cy))) in enumerate(zip(G.origins(rng, 10, 5, 4), designs)):
        roles = {
            "foundation": G.add_brick(scene, x, y, 0, 5, 4, colors["foundation"], "foundation", rank, True),
            "pedestal": G.add_brick(scene, x, y, 1, 5, 3, colors["pedestal"], "pedestal", rank, True),
            "candidate": G.add_brick(scene, x, y, 2, 4, 2, colors["candidate"], "candidate", rank, True),
            "side_marker": G.add_brick(scene, x + 4, y + mo, 2, 1, 1, colors["side_marker"], "side_marker", rank, True),
            "cap": G.add_brick(scene, x + cx, y + cy, 3, 1, 1, colors["cap"], "cap", rank, True),
        }
        records.append({"pod_index": rank, "features": {"marker_offset": mo, "cap_pos": [cx, cy]}, **roles})
    crop_rank = designs.index(target)
    task = {
        "records": records,
        "crop_rank": crop_rank,
        "cue_roles": list(CROP_ROLES),
        "scene_colors": colors,
        "target_features": records[crop_rank]["features"],
        "crop_mode": "geometry_only",
    }
    return scene, task


def pod_signature(scene, record, visible_roles):
    """What the inset shows: every role's geometry relative to the foundation + colors of visible roles."""
    by_id = {brick["id"]: brick for brick in scene["bricks"]}
    base = by_id[record["foundation"]]
    return tuple(
        (
            role,
            b["x"] - base["x"],
            b["y"] - base["y"],
            b["z"],
            b["lx"],
            b["ly"],
            b["color"] if role in visible_roles else None,
        )
        for role in CROP_ROLES
        for b in (by_id[record[role]],)
    )


def choose_masked(scene, task, highlight_role, rng):
    """Sorted masked-role set containing highlight_role that keeps the target pod's visible signature unique."""
    records = task["records"]
    target = records[task["crop_rank"]]
    others = [role for role in CROP_ROLES if role != highlight_role]
    options = [{highlight_role, *rng.sample(others, n - 1)} for n in (2, 3) for _ in range(12)]
    rng.shuffle(options)
    for masked in options:
        visible = set(CROP_ROLES) - masked
        target_sig = pod_signature(scene, target, visible)
        if sum(pod_signature(scene, record, visible) == target_sig for record in records) == 1:
            return sorted(masked)
    raise AssertionError("no masked set keeps the target unique")


def compose(image, id_mask, scene, task, projector, highlight_role, masked_roles):
    """Scene + 2x inset of the target pod (masked pieces charcoal, a lighter seam between two masked pieces, the
    highlighted piece outlined in magenta) -> (canvas, crop metadata)."""
    record = task["records"][task["crop_rank"]]
    role_ids = {role: record[role] + 1 for role in CROP_ROLES}
    masked_ids = {role_ids[role] for role in masked_roles}
    highlight_id = role_ids[highlight_role]
    by_id = {brick["id"]: brick for brick in scene["bricks"]}
    points = []
    for role in CROP_ROLES:
        b = by_id[record[role]]
        for xx in (b["x"], b["x"] + b["lx"]):
            for yy in (b["y"], b["y"] + b["ly"]):
                for zz in (b["z"], b["z"] + 1):
                    points.append(projector(xx, yy, zz))
    xs, ys = zip(*points)
    box = (
        max(0, int(min(xs)) - 5),
        max(0, int(min(ys)) - 5),
        min(image.width, int(max(xs)) + 6),
        min(image.height, int(max(ys)) + 6),
    )
    inset_mask = np.asarray(id_mask.crop(box))
    inset_arr = np.asarray(image.crop(box)).copy()
    inset_arr[~np.isin(inset_mask, sorted(role_ids.values()))] = BACKGROUND
    is_masked = np.isin(inset_mask, sorted(masked_ids))
    inset_arr[is_masked] = CHARCOAL
    seam = np.zeros_like(is_masked)
    h, w = inset_mask.shape
    for dy, dx in ((1, 0), (0, 1)):
        a, b = inset_mask[dy:, dx:], inset_mask[: h - dy, : w - dx]
        diff = (a != b) & np.isin(a, sorted(masked_ids)) & np.isin(b, sorted(masked_ids))
        seam[dy:, dx:] |= diff
        seam[: h - dy, : w - dx] |= diff
    inset_arr[seam] = SEAM
    retained = Counter(inset_mask.ravel().tolist())
    for role in CROP_ROLES:
        assert retained[role_ids[role]] >= G.MIN_PIXELS, role
    inset = Image.fromarray(inset_arr).resize((w * 2, h * 2), Image.Resampling.NEAREST)
    highlight2x = np.kron(inset_mask == highlight_id, np.ones((2, 2), bool))
    inset_arr2x = np.asarray(inset).copy()
    inset_arr2x[G.dilate(highlight2x, 3) & ~highlight2x] = MAGENTA
    inset = Image.fromarray(inset_arr2x)

    pad, label_h = 14, 22
    width = image.width + inset.width + 3 * pad
    height = max(image.height, inset.height + label_h + 2 * pad)
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    full_offset = (0, (height - image.height) // 2)
    canvas.paste(image, full_offset)
    inset_x = image.width + 2 * pad
    inset_y = (height - inset.height - label_h) // 2 + label_h
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    draw.text((inset_x, inset_y - label_h), "close-up (outlined piece is blacked out):", fill=(60, 60, 70), font=font)
    canvas.paste(inset, (inset_x, inset_y))
    frame = [inset_x - 2, inset_y - 2, inset_x + inset.width + 1, inset_y + inset.height + 1]
    draw.rectangle(frame, outline=(60, 60, 70), width=2)
    return canvas, {
        "crop_box": list(box),
        "highlight_role": highlight_role,
        "highlight_brick": record[highlight_role],
        "masked_roles": list(masked_roles),
        "visible_roles": [role for role in CROP_ROLES if role not in masked_roles],
        "masked_bricks": sorted(i - 1 for i in masked_ids),
        "outline_rgb": list(MAGENTA),
        "blank_rgb": list(CHARCOAL),
        "seam_rgb": list(SEAM),
        "full_offset": list(full_offset),
        "frame_wh": [width, height],
        "inset_box": [inset_x, inset_y, inset_x + inset.width, inset_y + inset.height],
    }


def render_crop_item(out, seed, global_index, highlight_role):
    """Render + write one Crop ID item (retrying construction attempts) -> (serialized scene, item)."""
    for attempt in range(500):
        try:
            scene, task = build_crop_scene(seed, global_index, attempt)
            rng = random.Random(seed * 7_000_003 + global_index * 613 + attempt * 31)
            masked_roles = choose_masked(scene, task, highlight_role, rng)
            base, id_mask, visibility, projector = G.render_scene(scene)
            base = G.recolor_plate(base, scene["plate_theme"])
            min_vis, min_px = G.visibility_gate(scene, id_mask, visibility)
            display, crop = compose(base, id_mask, scene, task, projector, highlight_role, masked_roles)
            width, height = crop["frame_wh"]
            off_x, off_y = crop["full_offset"]
            idm = np.asarray(id_mask)
            label = np.zeros((height, width), dtype=np.uint8)
            region = label[off_y : off_y + idm.shape[0], off_x : off_x + idm.shape[1]]
            region[idm == crop["highlight_brick"] + 1] = 1
            assert int((label == 1).sum()) >= G.MIN_PIXELS, "target too small"
            break
        except (AssertionError, RuntimeError):
            continue
    else:
        raise RuntimeError((global_index, highlight_role))
    payload = G.png_bytes(display)
    scene_id = f"cpgeo_s{seed}_{global_index:06d}"
    image_rel, mask_rel = f"samples/{scene_id}.png", f"masks/{scene_id}.png"
    (out / image_rel).write_bytes(payload)
    Image.fromarray(label, mode="L").save(out / mask_rel)
    anchor_x, anchor_y = G.interior_anchor(label == 1)
    point = [min(1000, max(0, round(1000 * anchor_x / width))), min(1000, max(0, round(1000 * anchor_y / height)))]
    tf = task["target_features"]
    n_px = int((label == 1).sum())
    item = {
        "scene_id": scene_id,
        "family": CROP_FAMILY,
        "image": image_rel,
        "mask": mask_rel,
        "question": CROP_QUESTION,
        "answer": highlight_role,
        "target_labels": [1],
        "target_count": 1,
        "answer_points": [
            {"point_2d": point, "instance_label": 1, "target_brick_id": crop["highlight_brick"], "visible_pixels": n_px}
        ],
        "image_wh": [width, height],
        "meta": {
            "highlight_role": highlight_role,
            "relation": highlight_role,
            "masked_roles": crop["masked_roles"],
            "visible_roles": crop["visible_roles"],
            "n_masked": len(crop["masked_roles"]),
            "target_marker_offset": tf["marker_offset"],
            "target_cap_pos": tf["cap_pos"],
            "scene_colors": task["scene_colors"],
            "plate_theme": scene["plate_theme"],
            "image_wh": [width, height],
            "coordinate_order_in_question": "yx",
            "coordinate_scale": 1000,
            "min_visibility": min_vis,
            "min_visible_pixels": min_px,
        },
    }
    serial = {k: v for k, v in scene.items() if k != "occ"}
    serial.update(scene_id=scene_id, image=image_rel, mask=mask_rel, sha256=hashlib.sha256(payload).hexdigest())
    serial.update(plate=G.PLATE, task=task, crop=crop, item=item, min_visibility=min_vis, min_visible_pixels=min_px)
    return serial, item


def generate_crop_id(seed, out):
    (out / "samples").mkdir(parents=True)
    (out / "masks").mkdir()
    roles = (r for r in CROP_ROLES for _ in range(CROP_PER_ROLE))
    pairs = [render_crop_item(out, seed, i, role) for i, role in enumerate(roles)]
    scenes, items = [s for s, _ in pairs], [i for _, i in pairs]
    meta = {
        "family": CROP_FAMILY,
        "seed": seed,
        "n_items": len(items),
        "highlight_roles": dict(Counter(i["meta"]["highlight_role"] for i in items)),
        "n_masked_hist": dict(sorted(Counter(i["meta"]["n_masked"] for i in items).items())),
        "target_count": 1,
        "scoring": "point-in-exact-brick-mask, label 1",
        "contract": {
            "outlined_piece_is_masked": True,
            "scene_wide_colors_per_role": True,
            "geometry_axes": {"marker_offset": list(MARKER_OFFSETS), "cap_pos": [list(c) for c in CAP_POSITIONS]},
            "target_offset_shared_by": ">=2 other pods",
            "target_cap_shared_by": ">=1 other pod",
        },
        "min_visibility": min(s["min_visibility"] for s in scenes),
    }
    write_outputs(out, items, scenes, meta)


# ---------------------------------------------------------------------------------------------------------------------
# Fine-grained counting: count the 2x4 bricks that satisfy a geometric predicate. 10 query pods (support slab + 2x4
# candidate + optional 1x1s) and two wrong-shape pods that also satisfy it; golds rotate 2..7; 84 scenes per predicate
# = 252 scenes, of which the first n_items are written (bench: 250, SFT shards: 252).

COUNT_FAMILY = "geomcount_relation"
PER_PREDICATE = 84
QUERY_PODS = 10
GOLD_VALUES = (2, 3, 4, 5, 6, 7)
PREDICATES = ("top_presence", "short_end", "perpendicular")
COUNT_QUESTION = {
    "top_presence": "How many 2x4 bricks have a 1x1 brick resting directly on top of them?",
    "short_end": "How many 2x4 bricks touch a 1x1 brick at a short end?",
    "perpendicular": "How many 2x4 bricks rest on a brick whose long side runs crosswise (perpendicular) to their own long side?",
}
WRONG_DIMS = {  # two wrong-shape candidates per scene, both satisfy the predicate
    "top_presence": ((6, 2), (2, 2)),
    "short_end": ((4, 1), (3, 2)),
    "perpendicular": ((4, 1), (3, 2)),
}


def build_pod(scene, colors, predicate, x, y, match, variant, dims, is_query):
    """Place one pod; brick colors come from a rotation independent of predicate truth."""
    lx, ly = dims
    pod = len([b for b in scene["bricks"] if b["construction_role"] == "candidate"])

    def add(x, y, z, lx, ly, role):
        color = PALETTE[colors["i"] % len(PALETTE)]
        colors["i"] += 3  # coprime stride keeps neighbors varied
        return G.add_brick(scene, x, y, z, lx, ly, color, role, pod, is_query)

    if predicate == "top_presence":
        sup_lx = min(lx + 1, 6)
        add(x, y, 0, sup_lx, 3, "support")
        add(x, y, 1, lx, ly, "candidate")
        if match:
            add(x + lx // 2, y, 2, 1, 1, "top_1x1")
        elif variant % 2 == 0 and sup_lx > lx:
            add(x + lx, y, 1, 1, 1, "beside_1x1")
    elif predicate == "short_end":
        add(x, y, 0, 6, 3, "support")
        add(x, y, 1, lx, ly, "candidate")
        if match:
            add(x + lx, y, 1, 1, 1, "end_1x1")
        elif variant % 2 == 0 and lx <= 4:
            add(x + lx + 1, y, 1, 1, 1, "gap_1x1")
        elif ly + 1 <= 3:
            add(x + 1, y + ly, 1, 1, 1, "side_1x1")
    else:  # perpendicular: a match lies crosswise to its oblong 6x4 slab
        add(x, y, 0, 6, 4, "support")
        if match:
            add(x + 1, y, 1, ly, lx, "candidate")
        else:
            add(x + 1, y + 1, 1, lx, ly, "candidate")


def cells(brick):
    """Set of (x, y) stud cells under a brick."""
    return {(brick["x"] + dx, brick["y"] + dy) for dx in range(brick["lx"]) for dy in range(brick["ly"])}


def rederive(bricks, cand, predicate):
    """Predicate truth for one candidate brick from geometry alone."""
    foot = cells(cand)
    others = [b for b in bricks if b["id"] != cand["id"]]
    if predicate == "top_presence":
        return any(b["z"] == cand["z"] + 1 and sorted((b["lx"], b["ly"])) == [1, 1] and cells(b) & foot for b in others)
    if predicate == "short_end":
        if cand["lx"] >= cand["ly"]:
            ends = {(cand["x"] - 1, cand["y"] + dy) for dy in range(cand["ly"])}
            ends |= {(cand["x"] + cand["lx"], cand["y"] + dy) for dy in range(cand["ly"])}
        else:
            ends = {(cand["x"] + dx, cand["y"] - 1) for dx in range(cand["lx"])}
            ends |= {(cand["x"] + dx, cand["y"] + cand["ly"]) for dx in range(cand["lx"])}
        return any(b["z"] == cand["z"] and sorted((b["lx"], b["ly"])) == [1, 1] and cells(b) & ends for b in others)
    supports = [b for b in others if b["z"] == cand["z"] - 1 and cells(b) & foot]
    assert len(supports) == 1
    return (cand["lx"] >= cand["ly"]) != (supports[0]["lx"] >= supports[0]["ly"])


def build_count_scene(seed, predicate, gold, scene_index, attempt):
    """Scene holding exactly `gold` matching 2x4 bricks; asserts reject bad draws."""
    rng = random.Random(seed * 17_000_003 + scene_index * 170_003 + attempt * 17109)
    colors = {"i": rng.randrange(len(PALETTE))}
    scene = G.blank_scene(seed, scene_index, COUNT_FAMILY, G.THEME_ORDER[scene_index % len(G.THEME_ORDER)], "none")
    placements = G.origins(rng, QUERY_PODS + 2, 6, 4)
    flags = [True] * gold + [False] * (QUERY_PODS - gold)
    rng.shuffle(flags)
    specs = [("query", (4, 2), flag) for flag in flags] + [("wrong", dims, True) for dims in WRONG_DIMS[predicate]]
    rng.shuffle(specs)
    for variant, ((kind, dims, match), (x, y)) in enumerate(zip(specs, placements)):
        build_pod(scene, colors, predicate, x, y, match, variant, dims, kind == "query")
    bricks = scene["bricks"]
    candidates = [b for b in bricks if b["construction_role"] == "candidate"]
    query = [b for b in candidates if sorted((b["lx"], b["ly"])) == [2, 4]]
    truth = [rederive(bricks, b, predicate) for b in query]
    assert sum(truth) == gold
    assert all(rederive(bricks, b, predicate) for b in candidates if sorted((b["lx"], b["ly"])) != [2, 4])
    match_colors = {b["color"] for b, t in zip(query, truth) if t}
    assert match_colors & {b["color"] for b, t in zip(query, truth) if not t}, "candidate colors must overlap"
    return scene


def generate_counting(seed, out, n_items=N_ITEMS):
    (out / "samples").mkdir(parents=True)
    work = [(p, GOLD_VALUES[j % len(GOLD_VALUES)]) for p in PREDICATES for j in range(PER_PREDICATE)][:n_items]
    scenes, items = [], []
    for scene_index, (predicate, gold) in enumerate(work):
        for attempt in range(1000):
            try:
                scene = build_count_scene(seed, predicate, gold, scene_index, attempt)
                display, id_mask, visibility, _ = G.render_scene(scene)
                display = G.recolor_plate(display, scene["plate_theme"])
                min_vis, min_px = G.visibility_gate(scene, id_mask, visibility)
                break
            except (AssertionError, RuntimeError):
                continue
        else:
            raise RuntimeError((predicate, gold, scene_index))
        payload = G.png_bytes(display)
        scene_id, image_rel = f"geomcount_s{seed}_{scene_index:06d}", f"samples/scene_{scene_index:06d}.png"
        (out / image_rel).write_bytes(payload)
        item = {
            "family": COUNT_FAMILY,
            "question": f"{COUNT_QUESTION[predicate]} Answer with one number.",
            "answer": str(gold),
            "meta": {"relation": predicate, "repeat_count": gold, "predicate_color": None},
            "scene_id": scene_id,
            "image": image_rel,
        }
        item["meta"].update(plate_theme=scene["plate_theme"], min_visibility=min_vis, min_visible_pixels=min_px)
        serial = {k: v for k, v in scene.items() if k != "occ"}
        serial.update(scene_id=scene_id, image=image_rel, sha256=hashlib.sha256(payload).hexdigest(), item=item)
        serial.update(task={"predicate": predicate, "predicate_color": None, "gold": gold})
        serial.update(min_visibility=min_vis, min_visible_pixels=min_px)
        scenes.append(serial)
        items.append(item)
    meta = {
        "family": COUNT_FAMILY,
        "seed": seed,
        "n_items": len(items),
        "predicates": dict(Counter(i["meta"]["relation"] for i in items)),
        "golds": dict(Counter(i["answer"] for i in items)),
        "query_pods": QUERY_PODS,
        "wrong_shape_pods": 2,
        "answer_mode": "open_only_no_options",
        "min_visibility": min(s["min_visibility"] for s in scenes),
        "min_visible_pixels": min(s["min_visible_pixels"] for s in scenes),
    }
    write_outputs(out, items, scenes, meta)


# ---------------------------------------------------------------------------------------------------------------------
# Stacks (Duplicate and Unpaired): four bricks, 3x3 base, 2x2 middle, 1x2 bar, 1x1 top. A design is 4 colors + the
# 4 GEOM_ATTRS; all stacks of a scene share one color tuple.


def mutate(design, attr, rng):
    """Copy of design with one geometry attribute changed to a different value."""
    out = json.loads(json.dumps(design))
    if attr == "middle_offset":
        out[attr] = list(rng.choice([v for v in OFFSETS if list(v) != design[attr]]))
    elif attr == "bar_orientation":
        out[attr] = "y" if design[attr] == "x" else "x"
    else:
        out[attr] = 1 - design[attr]
    return out


def stack_parts(x, y, design):
    """Base origin (x, y) + design -> [(role, x, y, z, lx, ly, color)] for base, middle, bar, top."""
    mx, my = x + design["middle_offset"][0], y + design["middle_offset"][1]
    if design["bar_orientation"] == "x":
        bar_x, bar_y, bar_lx, bar_ly = mx, my + design["bar_slot"], 2, 1
        top_x, top_y = bar_x + design["top_slot"], bar_y
    else:
        bar_x, bar_y, bar_lx, bar_ly = mx + design["bar_slot"], my, 1, 2
        top_x, top_y = bar_x, bar_y + design["top_slot"]
    return [
        ("base", x, y, 0, 3, 3, design["base_color"]),
        ("middle", mx, my, 1, 2, 2, design["middle_color"]),
        ("bar", bar_x, bar_y, 2, bar_lx, bar_ly, design["bar_color"]),
        ("top", top_x, top_y, 3, 1, 1, design["top_color"]),
    ]


# ---------------------------------------------------------------------------------------------------------------------
# Duplicate: K in {10, 11, 12} stacks; exactly two share a design. Distractors differ from the twins only in geometry
# (Hamming distance 1 while unused single mutants remain, then 2). Masks hold stack index + 1.

DUP_FAMILY = "duplicate_pair_pointing"
K_VALUES = [10, 11, 12]  # K cycles over these
MIN_TARGET_PIXELS = 80
DUP_MIN_VIS, DUP_MIN_PIXELS = 0.34, 55  # this task's own (looser) visibility gate
QUERY_ROLES = ("base", "middle", "bar", "top")
DUP_QUESTION = (
    "Exactly two stacks in this scene share a repeated design: they match in every "
    "brick color, size, orientation, and offset from bottom to top. Every other stack "
    "differs from that design and from one another. Find the two matching stacks."
    "\n\nInstead of describing them, point to both matching stacks. Return exactly two "
    "points, one on each of the two matching stacks, and no other points.\n"
    'Return only a JSON list in this exact form: [{"point": [y, x]}, {"point": [y, x]}]. '
    "Gemini coordinate order is [y, x], NOT [x, y]. Integer coordinates are normalized "
    "to 0-1000 of image height and width, origin at top-left. No labels, prose, or Markdown."
)


def trace_bin(brick, projector, width, height):
    """Coarse 5x3 screen-location bin of a brick center."""
    px, py = projector(brick["x"] + brick["lx"] / 2, brick["y"] + brick["ly"] / 2, brick["z"] + 0.5)
    return (min(4, max(0, int(5 * px / width))), min(2, max(0, int(3 * py / height))))


def sample_centers(rng, k_value):
    """k stack centers with >= 2 in one coarse screen bin -> (centers, sorted twin pair from that bin, bins)."""
    projector, width, height = G.make_projector(4)
    for _layout in range(3000):
        centers = []
        for _ in range(k_value):
            for _try in range(1000):
                cx, cy = rng.randint(2, G.PLATE - 3), rng.randint(2, G.PLATE - 3)
                if all(
                    cx + 2 <= ox - 1 or ox + 2 <= cx - 1 or cy + 2 <= oy - 1 or oy + 2 <= cy - 1 for ox, oy in centers
                ):
                    centers.append((cx, cy))
                    break
            else:
                break
        if len(centers) != k_value:
            continue
        bins = [trace_bin({"x": cx, "y": cy, "z": 3, "lx": 1, "ly": 1}, projector, width, height) for cx, cy in centers]
        pairs = [(i, j) for i in range(k_value) for j in range(i + 1, k_value) if bins[i] == bins[j]]
        if not pairs:
            continue
        return centers, sorted(rng.choice(pairs)), [list(value) for value in bins]
    raise RuntimeError("could not place stacks")


def design_signature(d):
    colors = (d["base_color"], d["middle_color"], d["bar_color"], d["top_color"])
    return (*colors, tuple(d["middle_offset"]), d["bar_orientation"], d["bar_slot"], d["top_slot"])


def mutate_design(target, attribute, rng, used):
    """Copy of target with one geometry attribute changed to a signature not in `used` (added to it)."""
    for _ in range(100):
        design = mutate(target, attribute, rng)
        signature = design_signature(design)
        if signature not in used:
            used.add(signature)
            return design
    raise RuntimeError((attribute, "mutation collision"))


def mutate_design_pair(target, rng, used):
    """Hamming-2 mutant not in `used` (only once every single mutant is used)."""
    for _ in range(200):
        first, second = rng.sample(list(GEOM_ATTRS), 2)
        design = mutate_design(mutate_design(target, first, rng, set()), second, rng, set())
        signature = design_signature(design)
        if sum(a != b for a, b in zip(signature, design_signature(target))) == 2 and signature not in used:
            used.add(signature)
            return design
    raise RuntimeError("pair mutation collision")


def build_dup_scene(scene_index, seed, attempt, k_value):
    """One scene with a twin pair and K - 2 distinct geometry distractors."""
    rng = random.Random(seed * 1_000_003 + scene_index * 7_919 + attempt * 97_409)
    centers, group, coarse_bins = sample_centers(rng, k_value)
    # question_type / query_role / answer are stored scene fields the task does not use; they fix the colors below
    question_type = "count" if scene_index % 2 else "attribute"
    design_role = QUERY_ROLES[(scene_index // 2) % len(QUERY_ROLES)]
    color_answer = list(G.COLORS)[(scene_index // 2 + seed) % len(G.COLORS)]
    colors = [color for color in G.COLORS if color != color_answer]
    rng.shuffle(colors)
    target = dict(zip(("base_color", "middle_color", "bar_color", "top_color"), colors))
    target.update(
        middle_offset=list(rng.choice(OFFSETS)),
        bar_orientation=rng.choice(("x", "y")),
        bar_slot=rng.randrange(2),
        top_slot=rng.randrange(2),
    )
    target[f"{design_role}_color"] = color_answer
    designs = [None] * k_value
    for index in group:
        designs[index] = json.loads(json.dumps(target))
    used = {design_signature(target)}
    distractors = [index for index in range(k_value) if index not in group]
    rng.shuffle(distractors)
    for ordinal, index in enumerate(distractors):
        # a binary attribute admits one distinct 1-mutant: fall through to the next one, then to Hamming-2
        for shift in range(len(GEOM_ATTRS)):
            try:
                designs[index] = mutate_design(target, GEOM_ATTRS[(ordinal + shift) % len(GEOM_ATTRS)], rng, used)
                break
            except RuntimeError:
                continue
        else:
            designs[index] = mutate_design_pair(target, rng, used)

    bricks, occupied, stacks = [], {}, []
    for index, ((cx, cy), design) in enumerate(zip(centers, designs)):
        ids = {}
        for role, x, y, z, lx, ly, color in stack_parts(cx - 1, cy - 1, design):
            ids[role] = brick_id = len(bricks)
            bricks.append({"id": brick_id, "x": x, "y": y, "z": z, "lx": lx, "ly": ly, "color": color})
            bricks[-1].update(construction_role=role, stack_index=index)
            for dx in range(lx):
                for dy in range(ly):
                    assert (x + dx, y + dy, z) not in occupied
                    occupied[(x + dx, y + dy, z)] = brick_id
        stacks.append(
            {
                "index": index,
                "center": [cx, cy],
                "design": design,
                "brick_ids": ids,
                "signature": list(design_signature(design)),
            }
        )
    return {
        "plate": G.PLATE,
        "family": "duplicate_stack",
        "mode": "hard",
        "K": k_value,
        "question_type": question_type,
        "query_role": None if question_type == "count" else design_role,
        "answer": "2" if question_type == "count" else color_answer,
        "repeat_count": 2,
        "duplicate_group": group,
        "coarse_bins": coarse_bins,
        "stacks": stacks,
        "mutations": list(GEOM_ATTRS),
        "max_hamming": 2,
        "bricks": bricks,
        "occ": occupied,
    }


def raster_gate(scene, image, id_mask, visibility, projector):
    """Assert visibility and that the twins' top bricks share a coarse screen bin; return item meta stats."""
    pixel_counts = Counter(id_mask.getdata())
    min_vis = min(visibility[b["id"]] for b in scene["bricks"])
    min_pixels = min(pixel_counts.get(b["id"] + 1, 0) for b in scene["bricks"])
    assert min_vis >= DUP_MIN_VIS and min_pixels >= DUP_MIN_PIXELS
    tops = [scene["bricks"][row["brick_ids"]["top"]] for row in scene["stacks"]]
    bins = [list(trace_bin(b, projector, image.width, image.height)) for b in tops]
    group_bins = [tuple(bins[index]) for index in scene["duplicate_group"]]
    assert len(group_bins) != len(set(group_bins))
    stats = {"min_visibility": round(min_vis, 4), "min_visible_pixels": min_pixels, "trace_bins": bins}
    return {**stats, "repeated_group_text_collision": True}


def generate_duplicate(seed, out):
    (out / "samples").mkdir(parents=True)
    (out / "masks").mkdir()
    items, scenes, hashes = [], [], set()
    for scene_index in range(N_ITEMS):
        for attempt in range(200):
            scene = build_dup_scene(scene_index, seed, attempt, K_VALUES[scene_index % len(K_VALUES)])
            image, id_mask, visibility, projector = G.render_scene(scene)
            try:
                raster = raster_gate(scene, image, id_mask, visibility, projector)
            except AssertionError:
                continue
            idm = np.asarray(id_mask)
            stack_mask = np.zeros(idm.shape, dtype=np.uint8)
            for brick in scene["bricks"]:
                stack_mask[idm == brick["id"] + 1] = brick["stack_index"] + 1
            group = scene["duplicate_group"]
            if any(int((stack_mask == g + 1).sum()) < MIN_TARGET_PIXELS for g in group):
                continue
            png = G.png_bytes(image)
            digest = hashlib.sha256(png).hexdigest()
            if digest not in hashes:
                break
        else:
            raise RuntimeError(scene_index)
        hashes.add(digest)
        image_rel, mask_rel = f"samples/scene_{scene_index:05d}.png", f"masks/scene_{scene_index:05d}.png"
        (out / image_rel).write_bytes(png)
        Image.fromarray(stack_mask).save(out / mask_rel)
        scene_id = f"duppair_s{seed}_{scene_index:05d}"
        bboxes = []
        for g in group:
            ys, xs = np.nonzero(stack_mask == g + 1)
            bboxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        twin_sig = scene["stacks"][group[0]]["signature"]
        hamming = Counter(
            sum(a != b for a, b in zip(st["signature"], twin_sig)) for st in scene["stacks"] if st["index"] not in group
        )
        items.append(
            {
                "scene_id": scene_id,
                "family": DUP_FAMILY,
                "image": image_rel,
                "mask": mask_rel,
                "question": DUP_QUESTION,
                "answer": json.dumps([g + 1 for g in group]),
                "target_stacks": group,
                "target_labels": [g + 1 for g in group],
                "target_bboxes": bboxes,
                "image_wh": [image.width, image.height],
                "meta": {
                    "K": scene["K"],
                    "total_stacks": scene["K"],
                    "mode": "hard",
                    "repeat_count": 2,
                    "duplicate_group": group,
                    "geometry_only": True,
                    "distractor_hamming": sorted(hamming.items()),
                    "source_scene_index": scene_index,
                    **raster,
                },
            }
        )
        serial = {key: value for key, value in scene.items() if key != "occ"}
        head = {"scene_id": scene_id, "image": image_rel, "mask": mask_rel, "img_w": image.width, "img_h": image.height}
        scenes.append({**head, "sha256": digest, "source_scene_index": scene_index, **serial})
    meta = {  # the field values of the released meta.json
        "family": DUP_FAMILY,
        "seed": seed,
        "n_items": len(items),
        "geometry_only": True,
        "K_counts": dict(sorted(Counter(i["meta"]["K"] for i in items).items())),
        "repeat_count": 2,
        "min_visibility": min(i["meta"]["min_visibility"] for i in items),
        "min_visible_pixels": min(i["meta"]["min_visible_pixels"] for i in items),
        "contract": {
            "pointing": True,
            "target_points": 2,
            "scoring": "exactly two points, one inside each target stack's exact mask",
            "scene_builder": "brickvision.generate.duplicate.build_scene (mode hard) at repeat_count == 2 indices",
            "distractors_differ_from_pair_in": "the orientation or position of one or two bricks",
            "K_values": K_VALUES,
        },
    }
    write_outputs(out, items, scenes, meta)


# ---------------------------------------------------------------------------------------------------------------------
# Unpaired: 3-4 groups of 2-4 identical stacks (9-12 stacks) plus one singleton. Each group mutates one geometry
# attribute of a base design; the singleton is group A's design with one value borrowed from B. Masks: stack index + 1.

UNPAIRED_FAMILY = "singleton_design_pointing_geo"
COLOR_ATTRS = ("base_color", "middle_color", "bar_color", "top_color")
UNPAIRED_QUESTION = (
    "Every stack in this scene has an exact twin — another stack whose design "
    "matches in every brick color, shape, orientation, and offset from bottom "
    "to top — except exactly one stack, whose complete design appears nowhere "
    "else. Find that one-of-a-kind stack."
    "\n\nInstead of naming any color, point to that one-of-a-kind stack. "
    "Return exactly one point.\n"
    'Return only a JSON list in this exact form: [{"point": [y, x]}]. '
    "Gemini coordinate order is [y, x], NOT [x, y]. Integer coordinates are "
    "normalized to 0-1000 of image height and width, origin at top-left. "
    "No labels, prose, or Markdown."
)


def unpaired_signature(design):
    return tuple(json.dumps(design[a], sort_keys=True) for a in COLOR_ATTRS + GEOM_ATTRS)


def random_design(rng):
    colors = rng.sample(PALETTE, 4)
    return {
        "base_color": colors[0],
        "middle_color": colors[1],
        "bar_color": colors[2],
        "top_color": colors[3],
        "middle_offset": list(rng.choice(OFFSETS)),
        "bar_orientation": rng.choice(("x", "y")),
        "bar_slot": rng.randrange(2),
        "top_slot": rng.randrange(2),
    }


def build_unpaired_scene(seed, global_index, attempt):
    """Scene + task dict naming the singleton stack."""
    rng = random.Random(seed * 13_000_027 + global_index * 89 + attempt * 11_003)
    base = random_design(rng)  # one color tuple for the WHOLE scene
    group_count = rng.choice((3, 4))
    mutated_attrs = rng.sample(GEOM_ATTRS, group_count)
    group_designs = [mutate(base, attr, rng) for attr in mutated_attrs]
    sizes = [rng.choice((2, 3, 4)) for _ in range(group_count)]
    while not 9 <= sum(sizes) <= 12:
        sizes[rng.randrange(group_count)] = rng.choice((2, 3, 4))

    a, b = rng.sample(range(group_count), 2)
    singleton = json.loads(json.dumps(group_designs[a]))
    singleton[mutated_attrs[b]] = group_designs[b][mutated_attrs[b]]

    designs = []
    for group_index, size in enumerate(sizes):
        designs += [group_designs[group_index]] * size
    designs.append(singleton)
    rng.shuffle(designs)
    total = len(designs)

    scene = G.blank_scene(seed, global_index, UNPAIRED_FAMILY, G.THEME_ORDER[global_index % len(G.THEME_ORDER)], None)
    placements = G.origins(rng, total, 5, 5)
    stacks = []
    for stack_index, ((x, y), design) in enumerate(zip(placements, designs)):
        ids = {
            role: G.add_brick(scene, bx, by, z, lx, ly, color, role, stack_index, True)
            for role, bx, by, z, lx, ly, color in stack_parts(x, y, design)
        }
        stacks.append(
            {
                "index": stack_index,
                "origin": [x, y],
                "design": design,
                "brick_ids": ids,
                "signature": list(unpaired_signature(design)),
            }
        )
    scene["stacks"] = stacks

    task = {
        "singleton_stack": next(s["index"] for s in stacks if s["design"] is singleton),
        "group_sizes": sizes,
        "group_count": group_count,
        "mutated_attrs": list(mutated_attrs),
        "singleton_from": [a, b],
        "total_stacks": total,
        "scene_colors": sorted(base[c] for c in COLOR_ATTRS),
    }
    return scene, task


def generate_unpaired(seed, out):
    (out / "samples").mkdir(parents=True)
    (out / "masks").mkdir()
    items, scenes = [], []
    for index in range(N_ITEMS):
        for attempt in range(2000):
            try:
                scene, task = build_unpaired_scene(seed, index, attempt)
                base, id_mask, visibility, _ = G.render_scene(scene)
                base = G.recolor_plate(base, scene["plate_theme"])
                min_visibility, min_pixels = G.visibility_gate(scene, id_mask, visibility)
                break
            except (AssertionError, RuntimeError):
                continue
        else:
            raise RuntimeError(index)
        payload = G.png_bytes(base)
        scene_id = f"sgeo_s{seed}_{index:06d}"
        image_rel, mask_rel = f"samples/scene_{index:06d}.png", f"masks/scene_{index:06d}.png"
        (out / image_rel).write_bytes(payload)
        idm = np.asarray(id_mask)
        stack_mask = np.zeros_like(idm)
        for brick in scene["bricks"]:
            stack_mask[idm == brick["id"] + 1] = brick["pod_index"] + 1
        Image.fromarray(stack_mask.astype(np.uint8)).save(out / mask_rel)
        singleton = task["singleton_stack"]
        ys, xs = np.nonzero(stack_mask == singleton + 1)
        items.append(
            {
                "scene_id": scene_id,
                "family": UNPAIRED_FAMILY,
                "image": image_rel,
                "question": UNPAIRED_QUESTION,
                "answer": str(singleton + 1),
                "mask": mask_rel,
                "target_stack": singleton,
                "target_label": singleton + 1,
                "target_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "image_wh": list(base.size),
                "meta": {
                    "total_stacks": task["total_stacks"],
                    "group_count": task["group_count"],
                    "mutated_attrs": task["mutated_attrs"],
                    "scene_colors": task["scene_colors"],
                    "plate_theme": scene["plate_theme"],
                    "min_visibility": min_visibility,
                    "min_visible_pixels": min_pixels,
                },
            }
        )
        serial = {key: value for key, value in scene.items() if key != "occ"}
        serial.update(scene_id=scene_id, image=image_rel, sha256=hashlib.sha256(payload).hexdigest(), task=task)
        serial.update(min_visibility=min_visibility, min_visible_pixels=min_pixels)
        scenes.append(serial)
    meta = {
        "family": UNPAIRED_FAMILY,
        "seed": seed,
        "n_items": len(items),
        "variant": "identical scene-wide colors; geometry-only differences",
        "stack_counts": dict(Counter(i["meta"]["total_stacks"] for i in items)),
    }
    write_outputs(out, items, scenes, meta)


GENERATORS = {
    "crop_id": generate_crop_id,
    "counting": generate_counting,
    "duplicate": generate_duplicate,
    "unpaired": generate_unpaired,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate one BrickVision task.")
    sub = ap.add_subparsers(dest="task", required=True)
    for task in GENERATORS:
        p = sub.add_parser(task, help=f"write the {task} items to --out")
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--out", type=Path, required=True)
        if task == "counting":
            p.add_argument("--n-items", type=int, default=N_ITEMS, help="first N of the 252 scenes (bench: 250)")
    args = vars(ap.parse_args())
    GENERATORS[args.pop("task")](**args)


if __name__ == "__main__":
    main()
