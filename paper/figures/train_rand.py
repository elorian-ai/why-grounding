"""Figure 3 (train_rand): training-coordinate randomization, 0/5/20/50/100% of the SFT trace coordinates randomized.
Left: accuracy of $GR_RESULTS/eval/sft/grounded{,_rand05,_rand20,_rand50,_rand100}_s17 (mean +- SD over decode seeds;
CountQA greedy). Right: MultihopSpatial randomize - replay (pp) +- SE from $GR_RESULTS/decode_randomization/report/.
-> $GR_RESULTS/paper/train_rand.{png,pdf}
    python paper/figures/train_rand.py
"""

import json
import statistics as st

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common import FRAME, INK, SET2, save, setup_sans

from grounded_reasoning import RESULTS
from grounded_reasoning.evaluate import cell, load_seeds

EVAL_RESULTS = RESULTS / "eval"
DECODE_REPORT = RESULTS / "decode_randomization" / "report" / "report.json"
ARMS = ["grounded", "grounded_rand05", "grounded_rand20", "grounded_rand50", "grounded_rand100"]
DOSE = [0, 5, 20, 50, 100]
MARK = {"ms": 5.5, "mec": "white", "mew": 0.6}
SAMPLED = [
    ("SMQA", "gr_spatialmqa", SET2[2], "o"),
    ("MHS", "gr_multihopspatial", SET2[0], "s"),
    ("TreeBench", "gr_treebench", SET2[1], "^"),
]


def task_accuracy():
    """-> {label: (means[5], sds[5], colour, marker)} for the left panel, in legend order."""
    runs = {arm: EVAL_RESULTS / "sft" / f"{arm}_s17" for arm in ARMS}
    sampled = {arm: load_seeds(run, "sampled") for arm, run in runs.items()}
    acc = {}
    for label, task, colour, marker in SAMPLED:
        per_arm = [[cell(hits)["acc_raw"] for hits in sampled[arm][task].values()] for arm in ARMS]
        acc[label] = ([st.mean(v) for v in per_arm], [st.stdev(v) for v in per_arm], colour, marker)
    greedy = [cell(load_seeds(runs[arm], "greedy")["gr_countqa"][0])["acc_raw"] for arm in ARMS]  # one pass, seed 0
    acc["CountQA"] = (greedy, [0] * 5, SET2[3], "D")
    return acc


def usage_effects():
    """-> (means[5], ses[5]) of the MultihopSpatial randomize - replay delta (pp) per arm."""
    with open(DECODE_REPORT) as f:
        results = json.load(f)["results"]
    effects = [results[arm]["multihopspatial"]["effect"] for arm in ARMS]
    return [e["mean"] for e in effects], [e["se"] for e in effects]


def main():

    sns.set_theme(style="whitegrid", palette="Set2")
    _, bold = setup_sans(base_size=10)
    plt.rcParams.update(
        {"axes.edgecolor": FRAME, "axes.linewidth": 0.8, "grid.color": "#E6E6E6", "grid.linewidth": 0.7}
    )
    plt.rcParams.update({"xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK})
    xs = np.arange(5)
    xt = [f"{d}%" for d in DOSE]

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.0, 2.35), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.36})
    fig.subplots_adjust(bottom=0.25, top=0.88)
    for label, (m, sd, colour, marker) in task_accuracy().items():
        line = {"ls": "-", "lw": 1.8, **MARK, "elinewidth": 1.0, "capsize": 2, "zorder": 3}
        a.errorbar(xs, m, yerr=sd, marker=marker, color=colour, label=label, **line)
    a.set_xticks(xs, xt)
    a.set_ylim(0.2, 0.65)
    a.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    a.set_yticklabels([".2", ".3", ".4", ".5", ".6"])
    a.set_ylabel("Accuracy", fontsize=9)
    a.set_title("Task accuracy", fontsize=9, weight="bold", family=bold, color=INK, pad=5)

    m, se = usage_effects()
    b.axhline(0, color="#9A9A9A", lw=0.9, zorder=2)
    bars = {"elinewidth": 1.4, "capsize": 2.5, "capthick": 1.4, **MARK, "zorder": 3}
    b.errorbar(xs, m, yerr=se, fmt="s", color=SET2[0], ecolor=SET2[0], label="MHS, 20 seeds", **bars)
    b.set_xticks(xs, xt)
    lower = np.floor(min(v - e for v, e in zip(m, se))) - 0.5
    b.set_ylim(lower, max(0.5, max(v + e for v, e in zip(m, se)) + 0.25))
    b.set_yticks(np.arange(np.ceil(lower), 1, 1))
    b.set_ylabel("Decode randomization Δ (pp)", fontsize=9)
    b.set_title("Grounding usage", fontsize=9, weight="bold", family=bold, color=INK, pad=5)
    for ax in (a, b):
        ax.tick_params(length=0, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(True)
    fig.supxlabel("Fraction of training-trace coordinates randomized", fontsize=9.5, color=INK, y=0.085)
    frame = {"frameon": True, "fancybox": True, "edgecolor": "#C8CCD3", "facecolor": "white"}
    place = {"ncol": 4, "fontsize": 8.5, "loc": "upper center", "bbox_to_anchor": (0.52, 0.035)}
    leg = fig.legend(
        *a.get_legend_handles_labels(), **place, **frame, handlelength=1.8, columnspacing=1.2, labelcolor=INK
    )
    leg.get_frame().set_linewidth(0.6)
    save(fig, "train_rand")


if __name__ == "__main__":
    main()
