"""Figure 5 (brickvision): one restyled BrickVision item per task in a 2x2 grid, a gold point on each target and the
question below. $GR_DATA/brickvision/{synthetic,restyled}/<task>/ -> $GR_RESULTS/paper/brickvision.{png,pdf}
    python paper/figures/brickvision_examples.py
"""

import matplotlib.pyplot as plt
import numpy as np
from common import first_where, save, setup_sans
from matplotlib import font_manager
from matplotlib.patches import Circle
from matplotlib.textpath import TextPath
from PIL import Image

from brickvision import render
from brickvision.generate import rederive
from grounded_reasoning import DATA

BRICKVISION = DATA / "brickvision"
FIG_W = 5.85  # in; 5.5 = \textwidth, widened so the a/c questions stay on one line (scale to \linewidth in LaTeX)
COL_GAP, ROW_GAP = 0.14, 0.16  # in
TITLE_H, TEXT_PAD = 0.27, 0.04  # in
FS_TITLE, FS_BODY = 9.5, 8.5  # pt
PITCH_IN = FS_BODY * 1.35 / 72  # body line pitch, inches (lines are placed one by one)
ACCENT = "#FF2D95"  # gold-point colour (no brick or plate uses this hue)
INK = "#1A1A1A"
POINT_R = 13  # px: gold-point marker radius
POINT_RING = 1.6  # pt: white ring width; a thin dark halo sits outside it

# one panel per task, covering the four restyle templates once each (studio / playroom / window desk / lamp);
# question None = the item's own question minus the answer-format suffix
PANELS = [
    {"task": task, "scene_id": scene_id, "title": title, "question": question}
    for task, scene_id, title, question in [
        ("duplicate", "duppair_s64004_00220", "Duplicate", "Point to the two stacks that have the same design."),
        ("unpaired", "sgeo_s56002_000003", "Unpaired", "Point to the stack that is not duplicated."),
        ("crop_id", "cpgeo_s66002_000109", "Crop ID", "Point to the piece corresponding to the outline in the crop."),
        ("counting", "geomcount_s61002_000008", "Fine-grained counting", None),
    ]
]
ROWS = [PANELS[:2], PANELS[2:]]  # 2x2 grid


def read_jsonl_row(path, scene_id):
    return first_where(path, lambda row: row.get("scene_id") == scene_id)


def load(panel):
    """-> (synthetic dataset root, synthetic item, restyled image) of one panel."""
    root = BRICKVISION / "synthetic" / panel["task"]
    image = Image.open(BRICKVISION / "restyled" / panel["task"] / "samples" / f"{panel['scene_id']}.png")
    return root, read_jsonl_row(root / "items.jsonl", panel["scene_id"]), image


def gold_points(panel, item, dataset_root):
    """-> gold point markers [(x, y)] in image pixels."""
    if panel["task"] != "counting":
        labels = np.array(Image.open(dataset_root / item["mask"]))
        wanted = item.get("target_labels") or [item["target_label"]]
        points = []
        for label in wanted:  # centroid, snapped onto the mask for thin/concave targets
            ys, xs = np.nonzero(labels == label)
            nearest = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
            points.append((float(xs[nearest]), float(ys[nearest])))
        return points

    scene = read_jsonl_row(dataset_root / "scenes.jsonl", item["scene_id"])
    bricks = scene["bricks"]
    render.PLATE = scene["plate"]  # the projector reads the plate size from this module global
    projector = render.make_projector(max(b["z"] + 1 for b in bricks))[0]
    predicate = scene["task"]["predicate"]
    candidates = [b for b in bricks if b["construction_role"] == "candidate" and sorted((b["lx"], b["ly"])) == [2, 4]]
    gold = [b for b in candidates if rederive(bricks, b, predicate)]
    # a point on each counted brick's top face, on the exposed stud cell farthest from the 1x1 resting on it
    occupied = {(b["x"] + i, b["y"] + j, b["z"]) for b in bricks for i in range(b["lx"]) for j in range(b["ly"])}
    points = []
    for brick in gold:
        cells = [(brick["x"] + i, brick["y"] + j) for i in range(brick["lx"]) for j in range(brick["ly"])]
        covered = [c for c in cells if (c[0], c[1], brick["z"] + 1) in occupied]
        exposed = [c for c in cells if (c[0], c[1], brick["z"] + 1) not in occupied]
        anchor = np.mean(covered, axis=0) if covered else np.mean(cells, axis=0)
        far = max(exposed, key=lambda c: (c[0] - anchor[0]) ** 2 + (c[1] - anchor[1]) ** 2)
        points.append(tuple(projector(far[0] + 0.5, far[1] + 0.5, brick["z"] + 1)))
    return points


def question_text(panel, item):
    question = panel["question"] or item["question"].replace(" Answer with one number.", "")
    return question.replace("2x4", "2×4").replace("1x1", "1×1")


def wrap_lines(text, width_in):
    """Greedy word wrap against the measured glyph widths of the active typeface."""
    prop = font_manager.FontProperties(family=plt.rcParams["font.family"], weight=300)
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and TextPath((0, 0), trial, size=FS_BODY, prop=prop).get_extents().width / 72 > width_in:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def draw_panel(fig, panel, x0, y0, width, text_h, fig_h, bold):
    """Lay out one panel whose bottom-left corner is (x0, y0) in inches."""
    dataset_root, item, image = load(panel)
    img_w, img_h = image.size
    img_h_in = width * img_h / img_w
    points = gold_points(panel, item, dataset_root)

    def axes(y_bottom, height):
        return fig.add_axes([x0 / FIG_W, y_bottom / fig_h, width / FIG_W, height / fig_h])

    ax_t = axes(y0 + text_h + img_h_in, TITLE_H)
    ax_t.set_axis_off()
    title = {"fontsize": FS_TITLE, "weight": "bold", "fontfamily": bold}
    ax_t.text(0, 0.22, panel["title"], **title, va="bottom", ha="left", color=INK)

    ax_i = axes(y0 + text_h, img_h_in)
    ax_i.imshow(image, interpolation="lanczos")
    ax_i.set(xlim=(0, img_w), ylim=(img_h, 0))
    ax_i.set_axis_off()
    for px, py in points:  # magenta disc, white ring, faint dark halo: reads on any background
        ax_i.add_patch(Circle((px, py), POINT_R + 3, facecolor="none", edgecolor="black", linewidth=0.6, alpha=0.55))
        ax_i.add_patch(Circle((px, py), POINT_R, facecolor=ACCENT, edgecolor="white", linewidth=POINT_RING))

    # question: one text object per line at a fixed pitch
    ax_q = axes(y0, text_h)
    ax_q.set_axis_off()
    for row, line in enumerate(wrap_lines(question_text(panel, item), width)):
        y = 1 - (TEXT_PAD + row * PITCH_IN) / text_h
        ax_q.text(0, y, line, fontsize=FS_BODY, color=INK, va="top", ha="left")


def main():
    _, bold = setup_sans()
    plt.rcParams["text.color"] = INK
    # Row geometry: panels in a row share one image height, widths follow aspect, the text block fits the longest
    # wrapped question of the row.
    row_geoms = []
    for row in ROWS:
        loaded = [load(panel) for panel in row]
        aspects = [image.size[0] / image.size[1] for _, _, image in loaded]
        img_h = (FIG_W - COL_GAP * (len(row) - 1)) / sum(aspects)
        widths = [a * img_h for a in aspects]
        n_lines = [len(wrap_lines(question_text(p, it), w)) for p, (_, it, _), w in zip(row, loaded, widths)]
        text_h = TEXT_PAD + max(n_lines) * PITCH_IN + TEXT_PAD
        row_geoms.append((widths, img_h, text_h))

    fig_h = sum(TITLE_H + h + t for _, h, t in row_geoms) + ROW_GAP * (len(ROWS) - 1)
    fig = plt.figure(figsize=(FIG_W, fig_h))
    fig.patch.set_facecolor("white")

    y_top = fig_h
    for row, (widths, img_h, text_h) in zip(ROWS, row_geoms):
        y0 = y_top - (TITLE_H + img_h + text_h)
        x0 = 0.0
        for panel, width in zip(row, widths):
            draw_panel(fig, panel, x0, y0, width, text_h, fig_h, bold)
            x0 += width + COL_GAP
        y_top = y0 - ROW_GAP

    save(fig, "brickvision")


if __name__ == "__main__":
    main()
