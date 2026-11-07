"""Figure 6 (format_ablation): SpatialMQA pair_idx 1690 under Appended (GT points pasted after the first mention of
each object) and Reflection (two-stage grounded trace), coordinates coloured like the rings on the image.
$GR_DATA/sft/{appended,grounded}.json -> $GR_RESULTS/paper/format_ablation.{png,pdf}
    python paper/figures/format_ablation.py
"""

import json
import re

import matplotlib.pyplot as plt
from common import INK, draw_trace, save, setup_sans
from PIL import Image

from grounded_reasoning import DATA

SUBSET, PAIR_IDX = "spatialmqa", 1690
PAL = ["#5B6FA6", "#E0643A", "#3E9C80", "#C2559A"]  # darker Set2 tones
RULE = "#8C96A8"
PT = re.compile(r"\[\d+, \d+\]")
W = 5.5
FS, FS_HEAD, FS_Q = 7.0, 8.0, 7.0


def load():
    """-> (image path, question, appended trace, reflection trace); rows joined on (TVG subset, pair_idx)."""

    def pick(arm):
        with open(DATA / "sft" / f"{arm}.json") as f:
            rows = json.load(f)
        return next(r for r in rows if r["image"].startswith(f"tvg/{SUBSET}/") and r["pair_idx"] == PAIR_IDX)

    g, p = pick("grounded"), pick("appended")
    q = g["conversations"][0]["value"].replace("<image>\n", "").split(" Think step by step")[0]
    gt, pt = g["conversations"][1]["value"], p["conversations"][1]["value"]
    return DATA / g["image"], q, pt, gt


def to_markup(trace, order, bold_look):
    """verbatim trace -> draw_trace markup: coordinates coloured by referent; 'Looking at' (the reflection move) in bold."""
    t = trace.replace("<think>", "").replace("</think>", "")
    t = "\n".join(l.strip() for l in t.split("\n") if l.strip())
    t = PT.sub(lambda m: f"[[c{order.index(m.group())}|{m.group()}]]", t)
    if bold_look:
        t = t.replace("Looking at ", "[[b|Looking at]] ")
    return t


def build(H, data, bold):
    img_path, q, pasted, grounded = data
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.canvas.draw()
    fx = lambda xin: xin / W
    fy = lambda yin: 1 - yin / H
    iy = lambda f: (1 - f) * H
    order = list(dict.fromkeys(PT.findall(grounded)))
    styles = {f"c{i}": {"color": c, "weight": "bold", "family": bold, "nobreak": True} for i, c in enumerate(PAL)}
    styles["b"] = {"weight": "bold", "family": bold}
    # left column: image with a ring on every point, question below; right column: Appended above Reflection
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    img_w, img_h = 1.95, 1.95 * h / w
    iax = fig.add_axes([0, fy(img_h), fx(img_w), img_h / H])
    iax.imshow(im, interpolation="lanczos")
    iax.set_axis_off()
    for k, c in enumerate(order):
        x, y = json.loads(c)
        for ec, lw in (("white", 2.6), (PAL[k], 1.5)):
            iax.scatter(
                [x / 1000 * w], [y / 1000 * h], s=120, facecolors="none", edgecolors=[ec], linewidths=lw, zorder=5
            )
    iax.set(xlim=(0, w), ylim=(h, 0))
    q_mk = "[[b|Question]] " + q.split("\n")[0] + "\n" + " ".join(q.split("\n")[1:])
    text = lambda mk, x, y, width, size: draw_trace(ax, mk, styles, x=x, y=y, width=width, size=size, color=INK)
    left_bot = iy(text(q_mk, x=0, y=fy(img_h + 0.08), width=fx(img_w), size=FS_Q))
    x0, y, bots = img_w + 0.25, 0.0, [left_bot]
    col_w = W - x0
    for title, mk in [("Appended", to_markup(pasted, order, False)), ("Reflection", to_markup(grounded, order, True))]:
        y = iy(text(f"[[b|{title}]]", x=fx(x0), y=fy(y), width=fx(col_w), size=FS_HEAD)) + 0.02
        ax.plot([fx(x0), fx(W)], [fy(y)] * 2, color=RULE, lw=0.7, solid_capstyle="butt")
        y = iy(text(mk, x=fx(x0), y=fy(y + 0.05), width=fx(col_w), size=FS)) + 0.16
        bots.append(y - 0.16)
    return fig, max(bots)


if __name__ == "__main__":
    _, bold = setup_sans(base_size=7)
    data = load()
    fig, used = build(6.0, data, bold)  # pass 1 measures the height (layout is top-anchored)
    plt.close(fig)
    fig, _ = build(used + 0.02, data, bold)
    save(fig, "format_ablation")
