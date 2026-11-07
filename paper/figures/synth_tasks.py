"""Figure 4 (synth_tasks): each probe task as a Base and a Text Labels scene, above held-out accuracy bars (Base
hatched, Text Labels solid) for Direct / Text CoT / Grounded CoT.
One held-out scene per task under $GR_DATA/probes/<task>/ and $GR_RESULTS/probes/<task>/summary.json
-> $GR_RESULTS/paper/synth_tasks.{png,pdf}
    python paper/figures/synth_tasks.py
"""

import json

import matplotlib.pyplot as plt
from common import first_where, save, setup_sans
from matplotlib.patches import Patch
from PIL import Image

from grounded_reasoning import DATA, RESULTS

PROBES = DATA / "probes"
PROBE_RESULTS = RESULTS / "probes"
TASKS = [("Wire", "wire_tracing"), ("Counting", "prompted_counting"), ("Arrow", "arrow_following")]
METHODS = ["direct", "text", "grounded"]
ROWS = ["Base", "Text Labels"]
CANVAS = (1440, 960)  # every scene resampled to one pixel size -> identical rendered size


def probe_accuracy():
    """-> {"Base"|"Text Labels": [[direct, text, grounded] per task]}, rounded to 3 decimals like the paper's table."""
    acc = {"Base": [], "Text Labels": []}
    for _, task in TASKS:
        with open(PROBE_RESULTS / task / "summary.json") as f:
            heldout = json.load(f)["eval_sets"]["heldout"]["accuracy"]
        for row, images in zip(ROWS, ("plain", "labeled")):
            acc[row].append([round(heldout[images][m], 3) for m in METHODS])
    return acc


def scene_paths():
    """-> [(column title, base image path, text-labels image path)]; one fixed held-out scene per task."""
    wire_dir, count_dir, arrow_dir = PROBES / "wire_tracing", PROBES / "prompted_counting", PROBES / "arrow_following"
    wire = first_where(wire_dir / "heldout/items.jsonl", lambda r: r["B"] == "B0" and r["X"] == "X1")
    cnt = first_where(count_dir / "heldout/items.jsonl", lambda r: r["K"] == 35)
    arrow_items = arrow_dir / "heldout/items.jsonl"
    arrow = first_where(arrow_items, lambda r: r["B"] == "PT_L0" and r["H_HOPS"] == 4)
    arrow_l = first_where(arrow_items, lambda r: r["B"] == "MCQ_L1" and r["scene"] == arrow["scene"])
    return [
        ("Wire Tracing", *(wire_dir / d / "images" / wire["image"] for d in ("heldout", "heldout_labeled"))),
        (
            "Visually-Prompted Counting",
            *(count_dir / d / "images" / cnt["image"] for d in ("heldout", "heldout_labeled")),
        ),
        ("Arrow Following", *(arrow_dir / "heldout/images" / r["image"] for r in (arrow, arrow_l))),
    ]


def pad_to(im, ratio=1.5):
    """White-pad a PIL image to width/height == ratio (all scenes have white grounds)."""
    w, h = im.size
    W, H = (w, round(w / ratio)) if w / h < ratio else (round(h * ratio), h)
    W, H = max(W, w), max(H, h)
    out = Image.new("RGB", (W, H), (255, 255, 255))
    out.paste(im, ((W - w) // 2, (H - h) // 2))
    return out


def main():
    _, bold = setup_sans(base_size=9)
    panels = scene_paths()
    acc = probe_accuracy()

    # layout in inches
    fig_w = 8.0
    label_w, col_gap, right = 0.62, 0.12, 0.04
    col_w = (fig_w - label_w - right - 2 * col_gap) / 3
    img_h = col_w / 1.5
    top, row_gap, block_gap, bar_h, bottom = 0.30, 0.09, 0.52, 1.55, 0.62  # bottom holds x ticks + legend
    fig_h = top + 2 * img_h + row_gap + block_gap + bar_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    def add_axes(x_in, y_top_in, w_in, h_in):
        """Axes from a top-left corner in inches (y measured down from the figure top)."""
        return fig.add_axes([x_in / fig_w, 1 - (y_top_in + h_in) / fig_h, w_in / fig_w, h_in / fig_h])

    col_x = [label_w + c * (col_w + col_gap) for c in range(3)]

    # top: example scenes; bottom: accuracy bars
    for r, row in enumerate(ROWS):
        y_top = top + r * (img_h + row_gap)
        x, y = label_w / 2 / fig_w, 1 - (y_top + img_h / 2) / fig_h
        head = {"fontsize": 10.5, "fontfamily": bold, "weight": "bold"}
        fig.text(x, y, row.replace(" ", "\n"), ha="center", va="center", **head, color="#222", linespacing=1.1)
        for c, (title, plain, labeled) in enumerate(panels):
            ax = add_axes(col_x[c], y_top, col_w, img_h)
            im = pad_to(Image.open(plain if r == 0 else labeled).convert("RGB")).resize(CANVAS, Image.LANCZOS)
            ax.imshow(im, interpolation="lanczos", aspect="auto")
            ax.set(xticks=[], yticks=[])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
                sp.set_color("#9a9a9a")
            if r == 0:
                ax.set_title(title, **head, pad=5)

    fills, edges = ["#66C2A5", "#FC8D62", "#8DA0CB"], ["#3E9E7E", "#D9633A", "#6577A8"]
    bar_top = top + 2 * img_h + row_gap + block_gap
    bar_axes = []
    for task_index, (task, _) in enumerate(TASKS):
        ax = add_axes(col_x[task_index], bar_top, col_w, bar_h)
        bar_axes.append(ax)
        for method_index, (fill, edge) in enumerate(zip(fills, edges)):
            for condition, offset in (("Base", -0.205), ("Text Labels", 0.205)):
                accuracy = acc[condition][task_index][method_index] * 100
                hatch = "///" if condition == "Base" else None
                x = method_index + offset
                ax.bar(x, accuracy, width=0.35, color=fill, hatch=hatch, edgecolor=edge, linewidth=0.8, zorder=3)
                ax.text(x, accuracy + 2, f"{accuracy:.1f}", ha="center", va="bottom", fontsize=7.5, color="#333")
        ax.set_title(task, fontsize=11, fontfamily=bold, weight="bold", pad=7)
        ax.set_xticks([0, 1, 2], ["Direct", "Text", "Grounded"], fontsize=8.5)
        ax.set(xlim=(-0.55, 2.55), ylim=(0, 112), yticks=[0, 25, 50, 75, 100])
        if task_index:
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="both", length=0, pad=5)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#E4E6EA", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color("#8C96A8")
            spine.set_linewidth(0.6)
    bar_axes[0].set_ylabel("Accuracy (%)", labelpad=6)
    grey = {"facecolor": "#999999", "edgecolor": "#707070"}
    handles = [Patch(**grey, hatch="///", label="Base"), Patch(**grey, label="Text Labels")]
    place = {"loc": "upper center", "bbox_to_anchor": (0.5, -0.19), "ncol": 2, "frameon": False, "fontsize": 10}
    bar_axes[1].legend(handles=handles, **place, handlelength=1.2, columnspacing=2)
    save(fig, "synth_tasks")


if __name__ == "__main__":
    main()
