"""Figure 7 (vgr): three TreeVGR-SFT-35K rows with box errors (13976, 15678: wrong box; 20258: box = whole image),
each with its boxes, the question, a <think> excerpt (" … " = elided, the defect in orange) and the <answer>.
$GR_DATA/treevgr_sft/TreeVGR-SFT-35K.json -> $GR_RESULTS/paper/vgr.{png,pdf}
    python paper/figures/vgr.py
"""

import json
import re

import matplotlib.pyplot as plt
from common import draw_trace, save, setup_sans
from matplotlib.patches import Rectangle
from PIL import Image

from grounded_reasoning import DATA

TREEVGR_SFT = DATA / "treevgr_sft"
BOX_RE = re.compile(r"<box>\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]</box>")
EL = " … "

# (row, defect label, [verbatim trace pieces], [(anchor-before, span, style)]); the anchor picks which occurrence of
# `span` is highlighted. Rows 13976, 15678 are GQA, 20258 is OCR-VQA (its defect is the box itself = the whole image).
BAD_KEYBOARD = (
    "The other is in front of the right-side monitor, identifiable by its shape despite being partially covered"
)
# fmt: off
SAMPLES = [
    (13976, "Incorrect box", [
        "There are two keyboards. One is in front of the left-side monitor, which is a standard white QWERTY keyboard "
        "(<box>[0,420,480,576]</box>). The other is in front of the right-side monitor, identifiable by its shape despite "
        "being partially covered (<box>[536,330,800,468]</box>).\nFor computers, the left setup has a tower CPU next to "
        "the monitor (<box>[56,102,184,438]</box>), indicating a computer system."], [("", BAD_KEYBOARD, "bad")]),
    (15678, "Incorrect box", [
        "The question asks for the vehicle right of the <box>[140,69,470,258]</box> fence. The black metal fence is on the "
        "left-side of the image, separating the grassy area from the tracks. The <box>[140,69,470,258]</box> is on the "
        "tracks, and since the fence is on the left, the <box>[140,69,470,258]</box> is to the right of the fence."],
        [("[140,69,470,258]</box> ", "fence", "bad")]),
    (20258, "Box too large", [
        'The cover of the book titled "GREAT RAILWAY JOURNEYS" is shown.',
        "Considering the title <box>[0,0,299,475]</box>, the steam-train imagery <box>[0,0,299,475]</box>, the subtitle "
        "<box>[0,0,299,475]</box>, and the author's reputation, it is clear that the book is related to travel."], []),
]
# fmt: on

W = 5.5
IMG_W, IMG_HMAX = 1.75, 1.35  # image slot (in)
TX0 = IMG_W + 0.18  # text column x (in)
SIZE, TITLE_SIZE = 7.4, 9.0  # pt
INK = "#1A1A1A"
# box + coordinate-text colours: darker edge tones of seaborn Set2 hues
BOX_EDGE = ["#3E9C80", "#5B6FA6", "#C2559A"]
DEFECT = "#E0643A"  # darker Set2 orange: the span the title refers to


def load_rows():
    """-> {row index: TreeVGR-SFT-35K row} for the SAMPLES rows. O(35k)."""
    with open(TREEVGR_SFT / "TreeVGR-SFT-35K.json") as f:
        rows = json.load(f)
    return {idx: rows[idx] for idx, *_ in SAMPLES}


def to_markup(text, highlights, order):
    """verbatim excerpt -> draw_trace markup: highlight spans first (raw text), then boxes -> chips p{k}."""
    for anchor, span, style in highlights:
        s = text.find(anchor + span) + len(anchor)
        text = text[:s] + f"[[{style}|{span}]]" + text[s + len(span) :]

    def chip(m):
        c = tuple(map(int, m.groups()))
        return f"[[p{order.index(c)}|[{','.join(map(str, c))}]]]"

    return BOX_RE.sub(chip, text)


def build(H, rows, bold):
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.canvas.draw()
    fx = lambda xin: xin / W
    fy = lambda yin: 1 - yin / H
    iny = lambda yf: (1 - yf) * H
    styles = {f"p{i}": {"color": c, "weight": "bold", "family": bold, "nobreak": True} for i, c in enumerate(BOX_EDGE)}
    styles["b"] = {"weight": "bold", "family": bold}
    styles["bad"] = {"color": DEFECT, "weight": "bold", "family": bold}
    y = 0.0
    for n, (idx, label, pieces, hl) in enumerate(SAMPLES):
        question, trace = (turn["value"] for turn in rows[idx]["conversations"][:2])
        order = list(dict.fromkeys(tuple(map(int, m.groups())) for m in BOX_RE.finditer(trace)))  # colour index
        title = f"({'abc'[n]}) {label}"
        head = {"fontsize": TITLE_SIZE, "fontweight": "bold", "family": bold, "color": INK}
        ax.text(fx(0), fy(y), title, **head, va="top", ha="left", transform=ax.transAxes)
        y0 = y + 0.22
        im = Image.open(TREEVGR_SFT / rows[idx]["images"][0]).convert("RGB")
        w, h = im.size
        sc = min(IMG_W / w, IMG_HMAX / h)
        iw, ih = w * sc, h * sc
        iax = fig.add_axes([fx(0), fy(y0 + ih), iw / W, ih / H])
        iax.imshow(im, extent=(0, w, h, 0), aspect="auto", interpolation="lanczos")
        iax.set(xlim=(0, w), ylim=(h, 0))
        iax.set_axis_off()
        used = {tuple(map(int, m.groups())) for p in pieces for m in BOX_RE.finditer(p)}
        for k, c in enumerate(order):
            if c not in used:
                continue
            x1, y1, x2, y2 = c
            iax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=BOX_EDGE[k], lw=1.6, clip_on=False))
        q = question.replace("<image>\n", "").split("\n")[0]
        answer = re.search(r"<answer>(.*?)</answer>", trace, re.S).group(1).strip()
        body = EL.join(to_markup(p, [h_ for h_ in hl if (h_[0] + h_[1]) in p], order) for p in pieces)
        mk = f"[[b|Question]]  {q}\n\n[[b|Trace]]  {body}\n\n[[b|Answer]]  {answer}"
        tb = draw_trace(ax, mk, styles, fx(TX0), fy(y0 - 0.04), (W - TX0) / W, SIZE, INK, linespacing=1.3)
        y = max(y0 + ih, iny(tb)) + 0.26
    return fig, y - 0.26


if __name__ == "__main__":
    _, bold = setup_sans(base_size=7)
    rows = load_rows()
    fig, used = build(8.0, rows, bold)  # pass 1 measures the height (layout is top-anchored)
    plt.close(fig)
    fig, _ = build(used + 0.02, rows, bold)
    save(fig, "vgr")
