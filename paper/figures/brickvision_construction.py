"""Figure 9 (brickvision_construction): one BrickVision scene from synthetic render, box map and segmentation mask
through the image generator to the restyled image with masks.
$GR_DATA/brickvision/{synthetic,restyled}/unpaired/ -> $GR_RESULTS/paper/brickvision_construction.{png,pdf}
    python paper/figures/brickvision_construction.py
"""

import matplotlib.pyplot as plt
import numpy as np
from brickvision_examples import BRICKVISION, INK, read_jsonl_row
from common import save, setup_sans
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from brickvision.restyle import draw_boxes, render_base, structure_boxes

SCENE = {"task": "unpaired", "scene_id": "sgeo_s56002_000003", "group_key": "pod_index"}

# layout (inches); wider than the 5.5 in text width, LaTeX scales it to \linewidth
FIG_W = 6.7
W_LABEL, W_IN = 0.98, 1.15  # left label gutter, input images
W_BOX, W_OUT = 0.80, 1.40  # generator box, restyled / final panels
GAP_X, GAP_Y = 0.25, 0.10  # between columns / between stacked inputs
PAD, HEAD = 0.08, 0.26  # border padding, header strip for the group title
FS_LABEL, FS_TITLE, FS_BOX = 9.0, 9.5, 9.0
ARROW = {"arrowstyle": "-|>", "mutation_scale": 8, "linewidth": 0.9, "color": "black"}
BOX_FACE, BOX_EDGE = "#CFE6F3", "black"
GROUP_EDGE = "black"
MASK_ALPHA, MASK_EDGE = 0.55, 0.6
# fmt: off
MASK_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#8C510A", "#7570B3",
                "#66A61E", "#E7298A", "#1B9E77"]
# fmt: on
LABELS = ["Synthetic\nRender", "Box Map", "Segmentation\nMask"]


def load_scene():
    dataset_root = BRICKVISION / "synthetic" / SCENE["task"]
    item = read_jsonl_row(dataset_root / "items.jsonl", SCENE["scene_id"])
    scene = read_jsonl_row(dataset_root / "scenes.jsonl", SCENE["scene_id"])
    synthetic = Image.open(dataset_root / item["image"]).convert("RGB")
    restyled = Image.open(BRICKVISION / "restyled" / SCENE["task"] / "samples" / f"{SCENE['scene_id']}.png")
    restyled = restyled.convert("RGB")
    labels = np.array(Image.open(dataset_root / item["mask"]))
    _, id_mask = render_base(scene)
    boxed = draw_boxes(synthetic, structure_boxes(id_mask, scene["bricks"], SCENE["group_key"]))
    return synthetic, boxed, restyled, labels


def masks(labels, background_rgb):
    """-> (solid colour per stack on the render's background colour, RGBA overlay, {label: rgb})."""
    ids = sorted(set(np.unique(labels)) - {0})
    colours = {lab: to_rgb(MASK_COLOURS[i % len(MASK_COLOURS)]) for i, lab in enumerate(ids)}
    solid = np.tile(np.array(background_rgb, dtype=float) / 255, labels.shape + (1,))
    overlay = np.zeros(labels.shape + (4,))
    for lab, rgb in colours.items():
        solid[labels == lab] = rgb
        overlay[labels == lab] = (*rgb, MASK_ALPHA)
    return solid, overlay, colours


def main():
    _, bold = setup_sans()
    plt.rcParams["text.color"] = INK
    synthetic, boxed, restyled, labels = load_scene()
    solid, overlay, colours = masks(labels, synthetic.getpixel((0, 0)))
    aspect = synthetic.size[0] / synthetic.size[1]
    h_in, h_out = W_IN / aspect, W_OUT / aspect
    group_w = PAD + W_LABEL + W_IN + PAD
    group_h = fig_h = PAD + HEAD + 3 * h_in + 2 * GAP_Y + PAD
    fig = plt.figure(figsize=(FIG_W, fig_h))
    fig.patch.set_facecolor("white")

    def ax_at(x, y, w, h):  # inches, bottom-left corner
        ax = fig.add_axes([x / FIG_W, y / fig_h, w / FIG_W, h / fig_h])
        ax.set_axis_off()
        return ax

    def show(ax, image, size):
        ax.imshow(image, interpolation="lanczos")
        ax.set(xlim=(0, size[0]), ylim=(size[1], 0))

    fx, fy = (lambda x: x / FIG_W), (lambda y: y / fig_h)

    # group border + title
    border = {"boxstyle": "round,pad=0,rounding_size=0.012", "facecolor": "none", "edgecolor": GROUP_EDGE}
    xy, w, h = (fx(0.01), fy(0.01)), fx(group_w - 0.02), fy(group_h - 0.02)
    fig.add_artist(FancyBboxPatch(xy, w, h, **border, transform=fig.transFigure, linewidth=0.9))
    bold_text = {"weight": "bold", "fontfamily": bold, "color": INK}
    x, y = fx(PAD + 0.04), fy(group_h - PAD - HEAD / 2)
    fig.text(x, y, "Procedural Generation", fontsize=FS_TITLE, **bold_text, va="center", ha="left")

    # column 1: labels + render / boxed / mask
    x_img = PAD + W_LABEL
    y_render = PAD + 2 * (h_in + GAP_Y)
    y_boxed = PAD + h_in + GAP_Y
    y_mask = PAD
    images = [
        (synthetic, synthetic.size),
        (boxed, boxed.size),
        (solid, synthetic.size),
    ]
    for (image, size), y, label in zip(images, (y_render, y_boxed, y_mask), LABELS):
        show(ax_at(x_img, y, W_IN, h_in), image, size)
        xl, yl = fx(PAD + W_LABEL / 2), fy(y + h_in / 2)
        kw = {"va": "center", "ha": "center", "multialignment": "center", "linespacing": 1.15}
        fig.text(xl, yl, label, fontsize=FS_LABEL, **bold_text, **kw)

    # generator box, centred between the render and boxed rows
    x_box = group_w + GAP_X
    y_mid = (y_render + y_boxed + h_in) / 2
    box_h = 2 * h_in + GAP_Y  # spans the render and box-map rows, so both input arrows are horizontal
    ax_box = ax_at(x_box, y_mid - box_h / 2, W_BOX, box_h)
    box = {"boxstyle": "round,pad=0,rounding_size=0.10", "facecolor": BOX_FACE, "edgecolor": BOX_EDGE}
    ax_box.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, **box, transform=ax_box.transAxes, linewidth=0.9))
    kw = {"ha": "center", "va": "center", "transform": ax_box.transAxes, "linespacing": 1.2}
    ax_box.text(0.5, 0.5, "Image\nGenerator", fontsize=FS_BOX, **bold_text, **kw)

    # restyled and final panels
    x_out = x_box + W_BOX + GAP_X
    x_fin = x_out + W_OUT + GAP_X
    y_out = y_mid - h_out / 2
    show(ax_at(x_out, y_out, W_OUT, h_out), restyled, restyled.size)
    ax_fin = ax_at(x_fin, y_out, W_OUT, h_out)
    show(ax_fin, restyled, restyled.size)
    ax_fin.imshow(overlay, interpolation="nearest")
    for lab in colours:
        ax_fin.contour((labels == lab).astype(float), levels=[0.5], colors=["white"], linewidths=MASK_EDGE)

    # arrows (figure coordinates); inputs leave from the group border's right edge
    gap = 0.04
    x_from = group_w + gap
    arrows = [
        ((x_from, y_render + h_in / 2), (x_box - gap, y_render + h_in / 2)),
        ((x_from, y_boxed + h_in / 2), (x_box - gap, y_boxed + h_in / 2)),
        ((x_box + W_BOX + gap, y_mid), (x_out - gap, y_mid)),
        ((x_out + W_OUT + gap, y_mid), (x_fin - gap, y_mid)),
    ]
    for (xa, ya), (xb, yb) in arrows:
        fig.add_artist(FancyArrowPatch((fx(xa), fy(ya)), (fx(xb), fy(yb)), transform=fig.transFigure, **ARROW))
    fig.add_artist(
        FancyArrowPatch(
            (fx(x_from), fy(y_mask + h_in / 2)),
            (fx(x_fin + W_OUT / 2), fy(y_out - gap)),
            transform=fig.transFigure,
            connectionstyle="angle,angleA=0,angleB=90,rad=4",
            **ARROW,
        )
    )

    save(fig, "brickvision_construction")


if __name__ == "__main__":
    main()
