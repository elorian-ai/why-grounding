"""Build the paired BrickVision SFT sets: $GR_DATA/brickvision/sft/scenes/<task>_seed<S>/ (generator shards) ->
$GR_DATA/brickvision/sft/{direct,text,grounded}.json, 2,500 scenes per task. Prompt and answer are shared; the
reasoning is empty (direct), a narrative re-derived from the scene geometry (text), or that narrative with a
{"point_2d": [x, y]} after every object it names (grounded).   python -m brickvision.build_sft
"""

import argparse
import json

import numpy as np
from PIL import Image

from brickvision import DATA, TASKS, read_jsonl
from brickvision import render as G
from brickvision.evaluate import xy_question
from brickvision.generate import cells, rederive

PER_TASK = 2500
# Screen directions. Projector: screen x ~ (x - y), screen y ~ (x + y) - z, so world +x points to the lower right
# and world +y to the lower left.
MIDDLE_CORNER = {(0, 0): "top", (1, 0): "right", (0, 1): "left", (1, 1): "bottom"}
BAR_DIR = {"x": "toward the lower right", "y": "toward the lower left"}
BAR_SIDE = {("x", 0): "upper-right", ("x", 1): "lower-left", ("y", 0): "upper-left", ("y", 1): "lower-right"}
CAP_END = {("x", 0): "upper-left", ("x", 1): "lower-right", ("y", 0): "upper-right", ("y", 1): "lower-left"}
MARKER_STUD = {0: "upper-right", 1: "middle", 2: "lower-left"}  # along the tier's lower-right end
CAP_ROW = {0: "upper-right", 1: "lower-left"}
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
CP_ROLE = {
    "foundation": "bottom 5x4 plate",
    "pedestal": "5x3 tier",
    "candidate": "2x4 brick",
    "side_marker": "1x1 end marker",
    "cap": "1x1 cap",
}


def anchor_1000(mask, wh):
    """Interior point of a bool (H, W) object mask, 0-1000 of the full frame (W, H)."""
    ax, ay = G.interior_anchor(mask)
    return (min(1000, max(0, round(1000 * ax / wh[0]))), min(1000, max(0, round(1000 * ay / wh[1]))))


def pt(point):
    return f'{{"point_2d": [{point[0]}, {point[1]}]}}'


def answer_json(points):
    return "[" + ", ".join(pt(p) for p in points) + "]"


def left_to_right(points):
    order = sorted(points, key=lambda k: (points[k][0], points[k][1]))
    return order, {k: r + 1 for r, k in enumerate(order)}


def traces_stacks(item, scene, data, family):
    """Duplicate / Unpaired -> (prompt, narrative(grounded) -> str, answer)."""
    mask = np.asarray(Image.open(data / item["mask"]))
    wh = tuple(item["image_wh"])
    stacks = scene["stacks"]
    points = {s["index"]: anchor_1000(mask == s["index"] + 1, wh) for s in stacks}
    order, ordinal = left_to_right(points)
    words = {}
    groups = {}
    for s in stacks:
        d, o = s["design"], s["design"]["bar_orientation"]
        words[s["index"]] = (
            f"middle 2x2 on the {MIDDLE_CORNER[tuple(d['middle_offset'])]} corner of the base, "
            f"bar running {BAR_DIR[o]} on the middle's {BAR_SIDE[(o, d['bar_slot'])]} side, "
            f"cap at the bar's {CAP_END[(o, d['top_slot'])]} end"
        )
        groups.setdefault((tuple(d["middle_offset"]), o, d["bar_slot"], d["top_slot"]), []).append(s["index"])
    if family == "duplicate":
        targets = sorted(scene["duplicate_group"], key=ordinal.get)
    else:
        targets = [scene["task"]["singleton_stack"]]

    def narrative(grounded):
        def name(i):
            return f"Stack {ordinal[i]} at {pt(points[i])}" if grounded else f"Stack {ordinal[i]}"

        lines = [
            f"All {len(stacks)} stacks wear the same four colors, so only the arrangement can "
            "tell them apart. I number the stacks from image-left to image-right and read each one."
        ]
        lines += [f"{name(i)}: {words[i]}." for i in order]
        if family == "duplicate":
            a, b = targets
            lines.append(
                f"Stacks {ordinal[a]} and {ordinal[b]} have exactly the same arrangement; "
                "every other arrangement appears only once."
            )
            lines.append(f"The matching pair is {name(a)} and {name(b)}.")
        else:
            for members in sorted(groups.values(), key=lambda m: min(ordinal[i] for i in m)):
                if len(members) > 1:
                    ms = sorted(members, key=ordinal.get)
                    lines.append(
                        f"Stacks {', '.join(str(ordinal[i]) for i in ms[:-1])} and {ordinal[ms[-1]]} share one arrangement."
                    )
            lines.append(f"{name(targets[0])} matches no other stack, so it is the one-of-a-kind stack.")
        return "\n".join(lines)

    return xy_question(item, len(targets)), narrative, answer_json([points[t] for t in targets])


def traces_crop_id(item, scene, data):
    """Crop ID -> (prompt, narrative(grounded) -> str, answer)."""
    task, crop = scene["task"], scene["crop"]
    records = task["records"]
    target = records[task["crop_rank"]]
    role = crop["highlight_role"]
    wh = tuple(item["image_wh"])
    idm = G.render_id_mask(scene)
    off_x, off_y = crop["full_offset"]
    frame = np.zeros((wh[1], wh[0]), dtype=np.int32)
    frame[off_y : off_y + idm.shape[0], off_x : off_x + idm.shape[1]] = idm
    points = {r["pod_index"]: anchor_1000(np.isin(frame, [r[k] + 1 for k in CP_ROLE]), wh) for r in records}
    order, ordinal = left_to_right(points)
    by_pod = {r["pod_index"]: r for r in records}
    tf = target["features"]

    def short(f):
        return (
            f"marker on the {MARKER_STUD[f['marker_offset']]} stud, cap on the "
            f"{ORD[f['cap_pos'][0] + 1]} stud, {CAP_ROW[f['cap_pos'][1]]} row"
        )

    def narrative(grounded):
        def name(p):
            return f"Structure {ordinal[p]} at {pt(points[p])}" if grounded else f"Structure {ordinal[p]}"

        cx, cy = tf["cap_pos"]
        lines = [
            f"The outlined piece is the {CP_ROLE[role]}. In the close-up, its structure has the end marker "
            f"on the {MARKER_STUD[tf['marker_offset']]} stud of the tier's lower-right end and the cap on "
            f"the {ORD[cx + 1]} stud of the 2x4, {CAP_ROW[cy]} row — {short(tf)}.",
            "All structures share the same colors, so I read these two cues on every structure, "
            "numbering structures from image-left to image-right.",
        ]
        lines += [f"{name(p)}: {short(by_pod[p]['features'])}." for p in order]
        match = target["pod_index"]
        lines.append(f"Only Structure {ordinal[match]} reads {short(tf)}, matching the close-up.")
        lines.append(f"I point at the {CP_ROLE[role]} of {name(match)}.")
        return "\n".join(lines)

    return xy_question(item, 1), narrative, answer_json([tuple(item["answer_points"][0]["point_2d"])])


def count_phrase(bricks, cand, predicate, ok):
    """Counting-trace phrase for one 2x4 and its geometric verdict."""
    foot = cells(cand)
    pod_1x1s = [
        b
        for b in bricks
        if b["id"] != cand["id"]
        and b["pod_index"] == cand["pod_index"]
        and b["z"] == cand["z"]
        and sorted((b["lx"], b["ly"])) == [1, 1]
    ]
    edge = {(x + dx, y + dy) for (x, y) in foot for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))} - foot
    beside = any(cells(b) & edge for b in pod_1x1s)
    if predicate == "top_presence":
        if ok:
            return "a 1x1 sits directly on top — counts"
        return "a 1x1 sits beside it on the tier, not on top" if beside else "nothing sits on top"
    if predicate == "short_end":
        if ok:
            return "a 1x1 touches its short end — counts"
        if beside:
            return "a 1x1 touches its long side, not a short end"
        return "the nearest 1x1 sits a stud away from its short end" if pod_1x1s else "no 1x1 sits at its short ends"
    return "it lies crosswise to its slab — counts" if ok else "it lies the same way as its slab"


def traces_counting(item, scene):
    """Counting -> (prompt, narrative(grounded) -> str, answer)."""
    bricks = scene["bricks"]
    predicate = scene["task"]["predicate"]
    idm = G.render_id_mask(scene)
    wh = (idm.shape[1], idm.shape[0])
    candidates = [b for b in bricks if b["construction_role"] == "candidate"]
    points = {b["id"]: anchor_1000(idm == b["id"] + 1, wh) for b in candidates}

    def key(b):
        return points[b["id"]][0], points[b["id"]][1]

    query = sorted((b for b in candidates if sorted((b["lx"], b["ly"])) == [2, 4]), key=key)
    w1, w2 = sorted((b for b in candidates if sorted((b["lx"], b["ly"])) != [2, 4]), key=key)
    verdicts = [(b, rederive(bricks, b, predicate)) for b in query]
    count = sum(ok for _, ok in verdicts)

    def dims(b):
        return f"{max(b['lx'], b['ly'])}x{min(b['lx'], b['ly'])}"

    def narrative(grounded):
        def at(b):
            return f" at {pt(points[b['id']])}" if grounded else ""

        lines = [
            f"Only 2x4 bricks count. The {dims(w1)} brick{at(w1)} and the {dims(w2)} "
            f"brick{at(w2)} are other sizes, so I skip them.",
            "I check the ten 2x4 bricks from image-left to image-right.",
        ]
        lines += [
            f"Brick {r}{at(b)}: {count_phrase(bricks, b, predicate, ok)}." for r, (b, ok) in enumerate(verdicts, 1)
        ]
        lines.append(f"{count} of the ten 2x4 bricks count.")
        return "\n".join(lines)

    return item["question"], narrative, str(count)


def main() -> None:
    argparse.ArgumentParser(description="Build sft/{direct,text,grounded}.json from sft/scenes/.").parse_args()
    rows = []
    for task in TASKS:
        task_rows = []
        for shard in sorted((DATA / "sft" / "scenes").glob(f"{task}_seed*")):
            scenes = {s["scene_id"]: s for s in read_jsonl(shard / "scenes.jsonl")}
            task_rows += [(item, scenes[item["scene_id"]], shard) for item in read_jsonl(shard / "items.jsonl")]
        if task == "counting":  # 10 x 252 -> 2,500, round-robin over (predicate, gold) cells
            by_cell = {}
            for r in task_rows:
                by_cell.setdefault((r[1]["task"]["predicate"], r[1]["task"]["gold"]), []).append(r)
            cell_keys = sorted(by_cell)
            picked, k = [], 0
            while len(picked) < PER_TASK:
                cell = by_cell[cell_keys[k % len(cell_keys)]]
                if cell:
                    picked.append(cell.pop(0))
                k += 1
            task_rows = picked
        for item, scene, shard in task_rows[:PER_TASK]:
            if task == "crop_id":
                prompt, narrative, answer = traces_crop_id(item, scene, shard)
            elif task == "counting":
                prompt, narrative, answer = traces_counting(item, scene)
            else:
                prompt, narrative, answer = traces_stacks(item, scene, shard, task)
            reasoning = {"direct": "", "text": narrative(False), "grounded": narrative(True)}
            targets = {arm: f"<think>\n{text}\n</think>\n\n{answer}" for arm, text in reasoning.items()}
            image = (shard / item["image"]).relative_to(DATA.parent)  # relative to $GR_DATA
            rows.append((task, item["scene_id"], str(image), prompt, targets))
    for arm in ("direct", "text", "grounded"):
        data = [
            {
                "image": image,
                "source": f"brickvision_{task}",
                "family": task,
                "scene_id": scene_id,
                "arm": arm,
                "conversations": [
                    {"from": "human", "value": "<image>\n" + prompt},
                    {"from": "gpt", "value": targets[arm]},
                ],
            }
            for task, scene_id, image, prompt, targets in rows
        ]
        (DATA / "sft" / f"{arm}.json").write_text(json.dumps(data))


if __name__ == "__main__":
    main()
