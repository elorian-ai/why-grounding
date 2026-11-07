"""Shared helpers of the paper figures: matplotlib style (text width = 5.5 in), output, a JSONL reader and
word-wrapped mixed-style text for the trace figures. Figures are written to $GR_RESULTS/paper/<name>.{png,pdf}.
"""

import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from grounded_reasoning import RESULTS

FIG_OUT = RESULTS / "paper"
FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
INK, FRAME = "#222222", "#8C96A8"
SET2 = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]  # seaborn Set2


def setup_sans(base_size=10):
    """Helvetica Neue Light if in paper/fonts/ or installed, else Inter Light -> (body family, first bold family)."""
    # DejaVu (mono / mathtext) has no light face; matplotlib then falls back to 400 and logs it once per text.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    for path in sorted(FONT_DIR.glob("*.[ot]tf")):
        font_manager.fontManager.addfont(str(path))
    for family in ("Helvetica Neue Light", "Helvetica Neue", "HelveticaNeue-Light", "Inter"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family, weight=300), fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.weight": 300, "text.color": INK, "font.size": base_size})
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    heavy = {f.name for f in font_manager.fontManager.ttflist if isinstance(f.weight, int) and f.weight >= 600}
    bold = next((fam for fam in ("Helvetica Neue", family, "Inter") if fam in heavy), "DejaVu Sans")
    return family, bold


def save(fig, name):
    """Write $GR_RESULTS/paper/<name>.png (300 dpi) and <name>.pdf."""
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    print("wrote", FIG_OUT / f"{name}.png")


def first_where(path, pred):
    """First record of a JSONL file with pred(record) true."""
    with open(path) as f:
        for line in f:
            if line.strip() and pred(r := json.loads(line)):
                return r
    raise KeyError(path)


# Trace text. Markup: plain text with spans [[key|text]], key = a style in `styles` (color, weight, style, family,
# nobreak); spans may contain brackets ("[[c0|[488, 626]]]"). A nobreak span never wraps inside, punctuation glued to
# a span wraps with it, newlines are hard breaks.
_SPAN = re.compile(r"\[\[([A-Za-z0-9_]+)\|(.*?)\]\](?!\])", re.S)


def parse(markup):
    """markup str -> [(text, style_key or None)]."""
    parts, out = _SPAN.split(markup), []  # [plain, key, span text, plain, ..., plain]
    for i in range(0, len(parts), 3):
        out += [(parts[i], None)] if parts[i] else []
        out += [(parts[i + 2], parts[i + 1])] if i + 2 < len(parts) else []
    return out


def draw_trace(ax, markup, styles, x, y, width, size, color, linespacing=1.32):
    """Draw markup in ax (axes fraction) from top-left (x, y), wrapping at x + width -> the y below the last line.
    Glyphs are measured at the figure dpi: the figure must have its final size and the savefig dpi."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    ab = ax.get_window_extent(renderer)
    to_ax_x = lambda px: px / ab.width
    to_ax_y = lambda px: px / ab.height
    line_h = size * linespacing * fig.dpi / 72.0  # pixels
    x_max = x + width

    def kw_for(st):
        font = {"color": st.get("color", color), "fontweight": st.get("weight", "normal"), "family": st.get("family")}
        place = {"fontstyle": st.get("style", "normal"), "va": "baseline", "ha": "left", "transform": ax.transAxes}
        return {"fontsize": size, **font, **place}

    def width_px(s, kw):
        if not s:
            return 0.0
        t = ax.text(0, 0, s, **kw)
        w = t.get_window_extent(renderer).width
        t.remove()
        return w

    space_px = width_px("a a", kw_for({})) - width_px("aa", kw_for({}))

    # words = maximal space-free runs of (text, kw, is_nobreak) fragments, so "(" + "[56,102]" + ")," wraps as one
    # unit; hard newlines become None.
    words, cur, gap = [], [], False  # gap: a space preceded the current word
    gap_before = [False]

    def end_word():
        nonlocal cur
        if cur:
            words.append((gap_before[0], cur))
            cur = []

    for text, key in parse(markup):
        st = styles.get(key, {}) if key else {}
        kw = kw_for(st)
        if st.get("nobreak"):  # atomic: spaces inside a nobreak span never break it
            if not cur:
                gap_before[0] = gap
            cur.append((text, kw, True))
            gap = False
            continue
        for li, line in enumerate(text.split("\n")):
            if li > 0:
                end_word()
                words.append(None)
                gap = False
            for wi, piece in enumerate(line.split(" ")):
                if wi > 0:  # a space separated this piece from the previous one
                    end_word()
                    gap = True
                if piece:
                    if not cur:
                        gap_before[0] = gap
                    cur.append((piece, kw, False))
                    gap = False
    end_word()

    # Consecutive plain fragments of one style on a line are merged into one Text (natural spacing).
    cx, cy = x, y - to_ax_y(line_h * 0.8)
    pending = []  # [x, text, kw] plain runs on the current line

    def draw_pending():
        for px_, t, kw in pending:
            ax.text(px_, cy, t, **kw)
        pending.clear()

    for tok in words:
        if tok is None:
            draw_pending()
            cx = x
            cy -= to_ax_y(line_h)
            continue
        gap_b, frags = tok
        wpx = sum(width_px(t, kw) for t, kw, _ in frags)
        lead = to_ax_x(space_px) if (gap_b and cx > x + 1e-6) else 0.0
        if cx + lead + to_ax_x(wpx) > x_max and cx > x + 1e-6:
            draw_pending()
            cx = x
            cy -= to_ax_y(line_h)
            lead = 0.0
        cx += lead
        for t, kw, nobreak in frags:
            if nobreak:
                ax.text(cx, cy, t, **kw)
                cx += to_ax_x(width_px(t, kw))
                continue
            if pending and pending[-1][2] is kw and lead >= 0:  # extend the same-style run (re-measure the whole run)
                run = pending[-1]
                sep = " " if lead > 0 else ""
                run[1] = run[1] + sep + t
                cx = run[0] + to_ax_x(width_px(run[1], kw))
            else:
                pending.append([cx, t, kw])
                cx += to_ax_x(width_px(t, kw))
            lead = 0.0
        if frags[-1][2]:  # a nobreak span ends any mergeable run
            pending.append([cx, "", frags[-1][1]])
    draw_pending()
    return cy - to_ax_y(line_h * 0.45)
