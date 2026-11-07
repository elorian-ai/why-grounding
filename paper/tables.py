"""Tables 2 (Appended vs Reflection), 3 (SFT and RL, ungrounded vs grounded) and 6 (appendix, point vs box).
$GR_RESULTS/eval/{sft,rl}/<arm>_s<seed>/ -> $GR_RESULTS/paper/table_{sft_rl,appended_reflection,point_box}.tex, each
followed by %-comment lines with the per-seed-pair deltas and error terms of every cell.
    python paper/tables.py
"""

import math
import statistics as st
from collections import defaultdict

from grounded_reasoning import RESULTS
from grounded_reasoning.evaluate import clustered_se, load_run

SEEDS = (17, 18, 19)  # training seeds; run k of one arm is compared with run k of the other


def runs(stage, arm):
    return [f"{stage}/{arm}_s{seed}" for seed in SEEDS]


SFT_TEXT, SFT_GROUNDED = runs("sft", "text"), runs("sft", "grounded")
SFT_BOX, SFT_APPENDED = runs("sft", "grounded_box"), runs("sft", "appended")
RL_TEXT, RL_GROUNDED = runs("rl", "text"), runs("rl", "grounded")

# task -> row name; spatial and perception rows use the sampled protocol (temperature 1, 3 decode seeds), counting
# rows the greedy one
# fmt: off
NAME = {"gr_spatialmqa": "SMQA", "gr_multihopspatial": "MHS", "gr_embspatial": "EmbSpatial", "gr_vsr": "VSR",
        "gr_omnispatial": "OmniSpatial", "gr_erqa": "ERQA", "gr_vstar": r"V$^*$", "gr_mmerw": "MME-RW",
        "gr_treebench": "TreeBench", "gr_countqa": "CountQA", "gr_fsc147": "FSC-147"}
# fmt: on
GREEDY = {"gr_countqa", "gr_fsc147"}
SPATIAL = ["gr_spatialmqa", "gr_multihopspatial", "gr_embspatial", "gr_vsr", "gr_omnispatial", "gr_erqa"]
PERCEPTION = ["gr_vstar", "gr_mmerw", "gr_treebench"]
COUNTING = ["gr_countqa", "gr_fsc147"]
BLOCKS = [("Spatial Reasoning", SPATIAL), ("Fine-grained Perception", PERCEPTION), ("Counting", COUNTING)]


def compare(results, ref, arm, task):
    """results {run: {task: {idx: (score, cluster)}}}, matched-seed ref / arm runs -> mr / ma (mean run accuracy), d
    (mean pair delta, pp), dsd (SD of the pair deltas), se_item, ci (95% half-width, pp), pairs, n (items)."""
    k = len(ref)

    def acc(run):
        return st.mean(v[0] for v in results[run][task].values())

    pairs, avg = [], defaultdict(float)  # avg = run-averaged per-item difference
    for r, a in zip(ref, arm):
        shared = set(results[r][task]) & set(results[a][task])
        diff = {i: results[a][task][i][0] - results[r][task][i][0] for i in shared}
        pairs.append(100 * st.mean(diff.values()))
        for i, x in diff.items():
            avg[i] += x / k
    d, dsd = st.mean(pairs), st.stdev(pairs)
    se_item = 100 * clustered_se(avg, {i: results[ref[0]][task][i][1] for i in avg})
    ci = 1.96 * math.sqrt(se_item**2 + dsd**2 / k)
    means = {
        "mr": st.mean([acc(r) for r in ref]),
        "ma": st.mean([acc(r) for r in arm]),
        "n": len(results[ref[0]][task]),
    }
    return {**means, "d": d, "dsd": dsd, "se_item": se_item, "ci": ci, "pairs": pairs}


def significant(c):
    return abs(c["d"]) > c["ci"]


def shaded(c):  # the paper's cell macros: green / red when the interval excludes 0, grey otherwise
    macro = r"\dns" if not significant(c) else (r"\dpos" if c["d"] > 0 else r"\dneg")
    return rf"{macro}{{{c['d']:+.1f} \pm {c['ci']:.1f}}}"


def rows(results_by_protocol, block_header, comparisons, fmt_row, blocks=BLOCKS):
    """-> (latex lines, log lines): per block a block_header.format(title=...) line (none if block_header is None) +
    fmt_row(task, cells) per bench, one cell per comparison (label, ref runs, arm runs)."""
    lines, log = [], []
    for bi, (title, tasks) in enumerate(blocks):
        if block_header is not None:
            if bi:
                lines.append(r"\midrule")
            lines.append(block_header.format(title=title))
        for task in tasks:
            results = results_by_protocol["greedy" if task in GREEDY else "sampled"]
            cells = [compare(results, ref, arm, task) for _, ref, arm in comparisons]
            lines.append(" & ".join(fmt_row(task, cells)) + r" \\")
            for (label, _, _), c in zip(comparisons, cells):
                pairs = [round(x, 2) for x in c["pairs"]]
                line = f"% {task[3:]:15s} {label}: pairs {pairs} Δ {c['d']:+.2f}±SD {c['dsd']:.2f} se_item {c['se_item']:.2f}"
                log.append(f"{line} CI ±{c['ci']:.2f}")
    return lines, log


def table_sft_rl(results_by_protocol):
    def fmt_row(task, cells):
        return [NAME[task], str(cells[0]["n"])] + [
            x for c in cells for x in (f"{c['mr']:.3f}", f"{c['ma']:.3f}", shaded(c))
        ]

    header = r"\multicolumn{{8}}{{l}}{{\emph{{{title}}}}} \\"
    comparisons = [("SFT", SFT_TEXT, SFT_GROUNDED), ("RL", RL_TEXT, RL_GROUNDED)]
    blocks = [("Spatial reasoning", SPATIAL), ("Fine-grained perception", PERCEPTION), ("Counting", COUNTING)]
    body, log = rows(results_by_protocol, header, comparisons, fmt_row, blocks)
    head = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lc ccc ccc}
\toprule
 & & \multicolumn{3}{c}{SFT} & \multicolumn{3}{c}{RL} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
Benchmark & $n$ & Ungrounded & Grounded & $\Delta$ & Ungrounded & Grounded & $\Delta$ \\
\midrule"""
    caption = (
        r"\caption{\textbf{Training recipe comparison}. We report the performance of grounded reasoning and text-only "
        r"reasoning training recipes after SFT and RL training phases. Deltas that exclude 0 from the confidence interval "
        r"are highlighted. We find that grounding provides a statistically significant benefit on counting, but not on "
        r"spatial reasoning or perception.}"
    )
    tail = [r"\bottomrule", r"\end{tabular}", caption, r"\label{tab:sft-rl}", r"\end{table}"]
    return [head, *body, *tail], log


def table_point_box(results_by_protocol):
    def fmt_row(task, cells):
        c = cells[0]
        s = f"{c['d']:+.1f} \\pm {c['ci']:.1f}"
        delta = rf"$\mathbf{{{s}}}$" if significant(c) else rf"${s}$"
        return [NAME[task], str(c["n"]), f"{c['mr']:.3f}", f"{c['ma']:.3f}", delta]

    header = r"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{title}}}}} \\"
    body, log = rows(results_by_protocol, header, [("box-point", SFT_GROUNDED, SFT_BOX)], fmt_row)
    head = r"""\begin{table*}[h]
\centering
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lc cc c}
\toprule
Bench & $n$ & Point & Box & Box $-$ Point \\
\midrule"""
    tail = [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\caption{\textbf{Point versus box grounding after SFT.} We report the box minus point accuracy delta with 95\% "
        r"confidence intervals computed as in Appendix~\ref{app:stats}. Bold deltas exclude 0.}",
        r"\label{tab:sft-pointbox}",
        r"\end{table*}",
    ]
    return [head, *body, *tail], log


def table_appended_reflection(results_by_protocol):
    def fmt_row(task, cells):
        c = cells[0]
        return [NAME[task], f"{c['mr']:.3f}", f"{c['ma']:.3f}", shaded(c)]

    blocks = [("", ["gr_spatialmqa", "gr_multihopspatial", *PERCEPTION, *COUNTING])]
    comparisons = [("reflection-appended", SFT_APPENDED, SFT_GROUNDED)]
    body, log = rows(results_by_protocol, None, comparisons, fmt_row, blocks)
    head = r"""% needs \usepackage{wrapfig} in the preamble
\begin{wraptable}{r}{0.5\textwidth}
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lc cc}
\toprule
Benchmark & Appended & Reflection & $\Delta$ \\
\midrule"""
    caption = (
        r"\caption{\textbf{Appended vs reflection performance}. The reflection reasoning style yields significantly "
        r"better performance on counting.}"
    )
    tail = [r"\bottomrule", r"\end{tabular}}", caption, r"\label{tab:sft-paste}", r"\end{wraptable}"]
    return [head, *body, *tail], log


def main():
    names = sorted({*SFT_TEXT, *SFT_GROUNDED, *SFT_BOX, *SFT_APPENDED, *RL_TEXT, *RL_GROUNDED})
    results_by_protocol = {
        protocol: {name: load_run(RESULTS / "eval" / name, protocol) for name in names}
        for protocol in ("sampled", "greedy")
    }
    out = RESULTS / "paper"
    out.mkdir(parents=True, exist_ok=True)
    for build in (table_sft_rl, table_appended_reflection, table_point_box):
        lines, log = build(results_by_protocol)
        path = out / (build.__name__ + ".tex")
        path.write_text("\n".join(lines) + "\n" + "\n".join(log) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
