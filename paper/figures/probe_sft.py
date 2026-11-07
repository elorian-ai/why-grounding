"""Figure 8 (probe_sft): arrow-following SFT targets of one training scene (k48h4_27_00322) under the three CoT arms,
step-aligned (row k = line k of every trace; rows identical in all arms are drawn once).
$GR_DATA/probes/arrow_following/sft/{text,grounded,text_labeled}.json -> $GR_RESULTS/paper/probe_sft.{png,pdf}
    python paper/figures/probe_sft.py
"""

import json
import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from common import INK, save, setup_sans
from common import draw_trace as _draw_trace

from grounded_reasoning import DATA

W_IN, FS = 5.5, 6.4
FS_TITLE, FS_HEAD, FS_Q = 8.5, 7.4, 6.8
RULE, GRID = "#8C96A8", "#E4E6EA"
COORD = re.compile(r"\[\d+, \d+\]")
SCENE = "k48h4_27_00322"
QUESTION_END = re.compile(r" (?=Think step by step|Answer directly|Answer with the terminal label)")
# the only non-position differences between the grounded and text targets
SWAPS = [
    ("these coordinates step-by-step", "these descriptions step-by-step"),
    ("So the answer is the point [", "So the answer is the blob at ["),
]
# referent colours: darker edge tones of seaborn-Set2 hues; the image ring uses the same tone
PAL = ["#5B6FA6", "#E0643A", "#3E9C80", "#C2559A", "#D4A017", "#6E9E2B", "#9C7B4F", "#2B7BB9", "#8E5BA8", "#6B6B6B"]


def styles_for(bold):
    st = {"b": {"weight": "bold", "family": bold}}
    for i, c in enumerate(PAL):  # coordinate (never split across lines) / text-arm words / printed label
        coord = {"color": c, "weight": "bold", "family": bold, "nobreak": True}
        st[f"g{i}"], st[f"t{i}"], st[f"l{i}"] = coord, {"color": c}, dict(coord)
    return st


def sft(path, pred):
    """First row of an SFT JSON (list of {image, conversations}) matching pred -> (image, human, gpt)."""
    with open(path) as f:
        r = next(r for r in json.load(f) if pred(r))
    return r["image"], r["conversations"][0]["value"], r["conversations"][1]["value"]


def recover_spans(ground, text):
    """Grounded trace (pre-answer part) and text trace, identical outside the position spans -> [(start, end)] of the
    text span standing where each coordinate stands. O(len(text))."""
    pieces = COORD.split(ground)
    pos, out = len(pieces[0]), []
    for piece in pieces[1:]:
        j = text.find(piece, pos + 1)
        out.append((pos, j))
        pos = j + len(piece)
    return out


def markup(text, spans):
    """text + sorted non-overlapping [(start, end, style)] -> draw_trace markup."""
    out, pos = [], 0
    for s, e, st in sorted(spans):
        out.append(text[pos:s])
        out.append(f"[[{st}|{text[s:e]}]]" if st else text[s:e])
        pos = e
    return "".join(out) + text[pos:]


def split_answer(t):
    """gpt target -> (reasoning part, '\\boxed{...}' tail)."""
    i = t.rindex("\\boxed{")
    return t[:i], t[i:]


def trace_lines(m):
    """markup of a full target -> its non-empty lines, <think> tags dropped (spans never contain newlines)."""
    m = m.replace("<think>", "").replace("</think>", "")
    return [line for line in m.split("\n") if line.strip()]


def text_spans(ground, text, colour_of):
    """Text-arm spans aligned with the grounded coordinates; answer coordinates (identical in both) styled as coordinates."""
    g_body = split_answer(ground)[0]
    t_body, t_ans = split_answer(text)
    g_cmp = g_body
    for a, b in SWAPS:
        g_cmp = g_cmp.replace(a, b)
    coords = COORD.findall(g_cmp)
    spans = [(s, e, f"t{colour_of[c]}") for (s, e), c in zip(recover_spans(g_cmp, t_body), coords)]
    off = len(t_body)
    return spans + [(off + m.start(), off + m.end(), f"g{colour_of[m.group()]}") for m in COORD.finditer(t_ans)]


def load_spec():
    """-> dict(images, question, cols, rows) for build()."""
    sft_dir = DATA / "probes/arrow_following/sft"
    arms = {
        a: sft(sft_dir / f"{run}.json", lambda r: r["chase_scene"] == SCENE)
        for a, run in [("cot", "text"), ("ground", "grounded"), ("label", "text_labeled")]
    }
    g, t, l = (arms[a][2] for a in ("ground", "cot", "label"))
    g_coords = COORD.findall(split_answer(g)[0])
    lab2c = dict(zip(re.findall(r"blob (\d+)", split_answer(l)[0]), g_coords))  # label k <-> coordinate k
    colour_of = {c: i for i, c in enumerate(dict.fromkeys(g_coords))}  # first-appearance order in the grounded trace
    l_body, l_ans = split_answer(l)
    lab_sp = [(m.start(), m.end(), f"l{colour_of[lab2c[m.group(1)]]}") for m in re.finditer(r"blob (\d+)", l_body)]
    lab_sp += [
        (len(l_body) + m.start(), len(l_body) + m.end(), f"g{colour_of[m.group()]}") for m in COORD.finditer(l_ans)
    ]
    cols_lines = [
        trace_lines(markup(t, text_spans(g, t, colour_of))),
        trace_lines(markup(g, [(m.start(), m.end(), f"g{colour_of[m.group()]}") for m in COORD.finditer(g)])),
        trace_lines(markup(l, lab_sp)),
    ]
    grid = [cells[0] if len(set(cells)) == 1 else list(cells) for cells in zip(*cols_lines)]  # str = shared row
    marks = [(json.loads(c), PAL[i]) for c, i in colour_of.items()]
    return {
        "images": [(DATA / arms["ground"][0], "Plain Scene"), (DATA / arms["label"][0], "Labeled Scene")],
        "marks": marks,
        "question": "[[b|Question]] "
        + QUESTION_END.split(arms["ground"][1].replace("<image>\n", ""))[0].replace("\n", " "),
        "cols": ["Text CoT", "Grounded CoT", "Text CoT with labels"],
        "rows": grid,
    }


def build(H, spec, bold):
    """spec -> figure of height H (in); returns (fig, used height in inches)."""
    styles = styles_for(bold)
    draw_trace = lambda ax, mk, **kw: _draw_trace(ax, mk, styles, color=INK, **kw)
    fig = plt.figure(figsize=(W_IN, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.canvas.draw()
    fy = lambda yin: 1 - yin / H  # inches from top -> axes fraction
    iy = lambda f: (1 - f) * H
    fx = lambda xin: xin / W_IN
    y = 0.0
    # images, each under a short bold title, with a ring (white under colour) on every cited blob
    gap, title_h = 0.14, 0.2
    img_w = (W_IN - gap) / 2
    for j, (path, title) in enumerate(spec["images"]):
        img = mpimg.imread(path)
        ih, iw = img.shape[:2]
        img_h = img_w * ih / iw
        x = fx(j * (img_w + gap))
        ax.text(x, fy(y), title, fontsize=FS_TITLE, fontweight="bold", family=bold, va="top", ha="left")
        iax = fig.add_axes([x, fy(y + title_h + img_h), fx(img_w), img_h / H])
        iax.imshow(img, interpolation="lanczos")
        iax.set_axis_off()
        for px, col in spec["marks"]:
            X, Y = px[0] / 1000 * iw, px[1] / 1000 * ih
            iax.scatter([X], [Y], s=150, facecolors="none", edgecolors="white", linewidths=2.4, zorder=4)
            iax.scatter([X], [Y], s=150, facecolors="none", edgecolors=[col], linewidths=1.4, zorder=5)
        iax.set(xlim=(0, iw), ylim=(ih, 0))
    y += title_h + img_h + 0.1
    b = draw_trace(ax, spec["question"], x=0, y=fy(y), width=1.0, size=FS_Q)
    y = iy(b) + 0.14
    # step-aligned grid: one column per trace format, row k = line k of every trace; identical rows span the grid
    cols, cgap = spec["cols"], 0.16
    col_w = (W_IN - (len(cols) - 1) * cgap) / len(cols)
    col_x = [i * (col_w + cgap) for i in range(len(cols))]
    hb = [
        iy(draw_trace(ax, f"[[b|{title}]]", x=fx(x0), y=fy(y), width=fx(col_w), size=FS_HEAD))
        for title, x0 in zip(cols, col_x)
    ]
    y = max(hb) + 0.02
    for x0 in col_x:
        ax.plot([fx(x0), fx(x0 + col_w)], [fy(y)] * 2, color=RULE, lw=0.7, solid_capstyle="butt")
    seps = []
    for row in spec["rows"]:
        shared = isinstance(row, str)
        if shared:  # one cell across the grid
            y = iy(draw_trace(ax, row, x=0, y=fy(y + 0.035), width=1.0, size=FS)) + 0.03
        else:
            bots = [draw_trace(ax, m, x=fx(x0), y=fy(y + 0.035), width=fx(col_w), size=FS) for m, x0 in zip(row, col_x)]
            y = max(iy(v) for v in bots) + 0.03
        seps.append((shared, y))
    for (shared, s_), (next_shared, _) in zip(seps[:-1], seps[1:]):  # full-width rule next to a shared row
        spans = [(0, W_IN)] if (shared or next_shared) else [(x0, x0 + col_w) for x0 in col_x]
        for a_, b_ in spans:
            ax.plot([fx(a_), fx(b_)], [fy(s_)] * 2, color=GRID, lw=0.6, zorder=-4)
    return fig, y


def main():
    _, bold = setup_sans(base_size=7)
    spec = load_spec()
    fig, used = build(11.0, spec, bold)  # pass 1 measures the height (layout is top-anchored)
    plt.close(fig)
    fig, _ = build(used, spec, bold)
    save(fig, "probe_sft")


if __name__ == "__main__":
    main()
