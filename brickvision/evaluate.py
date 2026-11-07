"""BrickVision prompts, scoring and Table 4 (CPU only).

python -m brickvision.evaluate score --prompts prompts.jsonl       # {task, scene_id, image, prompt} per item
python -m brickvision.evaluate score --predictions preds.jsonl     # rows {task, scene_id, response[, sample]}
python -m brickvision.evaluate tables     # $GR_RESULTS/brickvision/* -> stdout, tables.json, section_table.json
"""

import argparse
import base64
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from brickvision import DATA, RESULTS, read_jsonl
from brickvision.generate import GEOM_ATTRS
from brickvision.render import render_id_mask

TASK_NAMES = {
    "crop_id": "Crop ID",
    "counting": "Fine-grained counting",
    "duplicate": "Duplicate",
    "unpaired": "Unpaired",
}

# ---------------------------------------------------------------------------------------------------------------------
# Prompts and scoring. Gemini gets the stored [y, x] questions; every other model gets `model_question`. Pointing is
# exact iff the reply has exactly N points and each target mask label is hit once; Counting is the parsed integer.

PAIR_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")  # [a, b] pairs, int or float
INT_PAIR_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")  # Gemini Unpaired: integer pairs only
TAIL_MARKERS = ("Return only a JSON list", "Instead of naming any color", "Instead of describing them")
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
_TEENS = ("thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")
NUMBER_WORDS = {w: i for i, w in enumerate(_ONES + _TEENS)}


def target_labels(item):
    """Mask labels a pointing item's points must hit; None for Counting items."""
    return item.get("target_labels") or ([item["target_label"]] if "target_label" in item else None)


def xy_question(item, n_points):
    """Stored question with its Gemini [y, x] tail replaced by the xy 0-1000 contract."""
    q = item["question"]
    body = q[: min([q.find(m) for m in TAIL_MARKERS if q.find(m) >= 0] or [len(q)])].rstrip()
    what = {1: "exactly one point", 2: "exactly two points, one on each target"}[n_points]
    return (
        f"{body}\n\nAnswer by pointing: return {what} and no other points.\n"
        'Return only a JSON list in this exact form: [{"point_2d": [x, y]}'
        + (', {"point_2d": [x, y]}' if n_points == 2 else "")
        + "]. Coordinates are integers normalized to 0-1000 of image WIDTH (x) and "
        "HEIGHT (y), origin at the top-left corner. No labels, prose, or Markdown."
    )


def model_question(item):
    """Prompt for every non-Gemini model: xy contract for pointing items, stored question for Counting."""
    return xy_question(item, len(target_labels(item))) if "mask" in item else item["question"]


def png_b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def to_pixel(y1000, x1000, width, height):
    return max(0, min(width - 1, int(round(x1000 / 1000 * width)))), max(
        0, min(height - 1, int(round(y1000 / 1000 * height)))
    )


def score_points(points_yx, item, mask):
    """points_yx [(y1000, x1000)], mask (H, W) label map -> verdicts + exact_ok (N points, each target hit once)."""
    width, height = item["image_wh"]
    targets = list(item["target_labels"])
    pixel_sets = {int(v): np.nonzero(mask == v) for v in np.unique(mask) if v != 0}
    radius = max(3, int(0.015 * (width**2 + height**2) ** 0.5))
    verdicts = []
    for y1000, x1000 in points_yx:
        px, py = to_pixel(y1000, x1000, width, height)
        near = sorted(
            lab for lab, (ys, xs) in pixel_sets.items() if ((xs - px) ** 2 + (ys - py) ** 2).min() <= radius**2
        )
        verdicts.append({"point_xy": [px, py], "exact_label": int(mask[py, px]), "dilated_labels": near})
    exact_hits = Counter(v["exact_label"] for v in verdicts)
    count_ok = len(points_yx) == len(targets)
    return {
        "n_points": len(points_yx),
        "verdicts": verdicts,
        "hit_labels": [v["exact_label"] for v in verdicts],
        "n_targets_hit": sum(exact_hits[t] > 0 for t in targets),
        "count_ok": count_ok,
        "exact_ok": count_ok and all(exact_hits[t] == 1 for t in targets),
        "dilated_ok": count_ok and all(sum(v["dilated_labels"] == [t] for v in verdicts) == 1 for t in targets),
    }


def score_singleton_point(points_yx, item, mask):
    """Gemini Unpaired scorer: exact_ok iff exactly one point, inside the target stack (+ diagnostics)."""
    width, height = item["image_wh"]
    target = item["target_label"]
    pixel_sets, centroids = {}, {}
    for label in range(1, item["meta"]["total_stacks"] + 1):
        ys, xs = np.nonzero(mask == label)
        pixel_sets[label] = (xs, ys)
        centroids[label] = (float(xs.mean()), float(ys.mean()))
    radius = max(3, int(0.015 * (width**2 + height**2) ** 0.5))
    verdicts = []
    for y1000, x1000 in points_yx:
        px, py = to_pixel(y1000, x1000, width, height)
        near = {lab for lab, (xs, ys) in pixel_sets.items() if ((xs - px) ** 2 + (ys - py) ** 2).min() <= radius**2}
        nearest = min(centroids, key=lambda lab: (centroids[lab][0] - px) ** 2 + (centroids[lab][1] - py) ** 2)
        verdicts.append(
            {
                "point_xy": [px, py],
                "exact_label": int(mask[py, px]),
                "dilated_labels": sorted(near),
                "nearest_label": nearest,
            }
        )
    single = len(points_yx) == 1
    first = verdicts[0] if verdicts else {}
    return {
        "n_points": len(points_yx),
        "verdicts": verdicts,
        "exact_ok": single and first.get("exact_label") == target,
        "dilated_ok": single and first.get("dilated_labels") == [target],
        "centroid_ok": single and first.get("nearest_label") == target,
    }


def parse_open_answer(text):
    """Free-form integer answer -> (pred | None, parse route). Tries the \\boxed{} content, the last line and the
    whole reply as a bare number or number word, then the only integer or number word in the reply."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None, "unparsed"
    no_markdown = re.sub(r"[*_`]", "", cleaned).strip()
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", no_markdown, flags=re.IGNORECASE)
    candidates = boxed[-1:] + [line.strip() for line in no_markdown.splitlines() if line.strip()][-1:] + [no_markdown]
    for candidate in candidates:
        value = re.sub(
            r"^(?:(?:the\s+)?(?:final\s+)?answer|the\s+color|the\s+count)\s*(?:is|:|=)?\s*",
            "",
            candidate.strip().lower(),
        )
        value = re.sub(r"[^a-z0-9-]+", " ", value).strip()
        if re.fullmatch(r"\d+", value):
            return str(int(value)), "exact_integer"
        if value in NUMBER_WORDS:
            return str(NUMBER_WORDS[value]), "exact_number_word"
    numbers = re.findall(r"(?<![\w.-])(\d+)(?![\w.-])", no_markdown)
    if len(set(numbers)) == 1:
        return str(int(numbers[0])), "unique_integer"
    words = [word for word in NUMBER_WORDS if re.search(rf"\b{word}\b", no_markdown.lower())]
    if len(words) == 1:
        return str(NUMBER_WORDS[words[0]]), "unique_number_word"
    return None, "unparsed"


def score_xy_reply(raw, item, data, pixel=False):
    """Score one non-Gemini reply for `item` (data = its bench dir). Pointing: every [a, b] pair is an (x, y)
    point, 0-1000 (pixel=True: image pixels, rescaled to 0-1000 first). Counting: integer == gold."""
    if "mask" not in item:
        pred, route = parse_open_answer(raw)
        ok = pred is not None and str(pred) == str(item["answer"])
        return {"kind": "open_int", "gold": item["answer"], "pred": pred, "parse_route": route, "exact_ok": ok}
    mask = np.asarray(Image.open(data / item["mask"]))
    pts_xy = [(float(a), float(b)) for a, b in PAIR_RE.findall(raw)]
    out = {}
    if pixel:
        w, h = item["image_wh"]
        out["points_xy_px"] = pts_xy
        pts_xy = [(x / w * 1000, y / h * 1000) for x, y in pts_xy]
    targets = target_labels(item)
    scored = score_points([(y, x) for x, y in pts_xy], {"image_wh": item["image_wh"], "target_labels": targets}, mask)
    return {**out, "kind": "pointing", "target_labels": targets, "points_xy_1000": pts_xy, **scored}


# ---------------------------------------------------------------------------------------------------------------------
# `score`: exact accuracy per task and overall (items pooled), mean +- SD over samples; a missing answer is wrong.


def write_prompts(bench, data, path):
    with open(path, "w") as handle:
        for task, items in bench.items():
            for item in items:
                image = str((data / task / item["image"]).resolve())
                row = {"task": task, "scene_id": item["scene_id"], "image": image, "prompt": model_question(item)}
                handle.write(json.dumps(row) + "\n")


def mean_sd(values):
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def score(args, error):
    bench = {task: read_jsonl(args.data / task / "items.jsonl") for task in TASK_NAMES}
    if args.prompts:
        write_prompts(bench, args.data, args.prompts)
        return
    if not args.predictions:
        error("pass --predictions (or --prompts)")

    responses = {}
    for row in read_jsonl(args.predictions):
        responses[row["task"], row["scene_id"], row.get("sample", 0)] = row["response"]
    known = {(task, item["scene_id"]) for task, items in bench.items() for item in items}
    unknown = sorted({key[:2] for key in responses} - known)
    if unknown:
        raise SystemExit(f"{len(unknown)} predictions match no BrickVision item, e.g. {unknown[0]}")
    tasks = [task for task in TASK_NAMES if any(key[0] == task for key in responses)]
    samples = sorted({key[2] for key in responses})

    exact = {}  # (task, sample) -> [bool per item]
    scored, missing = [], 0
    for sample in samples:
        for task in tasks:
            exact[task, sample] = []
            for item in bench[task]:
                response = responses.get((task, item["scene_id"], sample))
                if response is None:
                    missing += 1
                    result = {"exact_ok": False, "missing": True}
                else:
                    result = score_xy_reply(response, item, args.data / task, pixel=args.pixel)
                exact[task, sample].append(bool(result["exact_ok"]))
                scored.append({"task": task, "scene_id": item["scene_id"], "sample": sample, **result})

    sd_note = f" (mean +- SD over {len(samples)} samples)" if len(samples) > 1 else ""
    print(f"| task | items | exact %{sd_note} |\n|---|---|---|")
    for task in tasks:
        mean, sd = mean_sd([100 * sum(v) / len(v) for v in (exact[task, s] for s in samples)])
        print(f"| {TASK_NAMES[task]} | {len(bench[task])} | {mean:.1f}" + (f" +- {sd:.1f} |" if sd_note else " |"))
    pooled = [sum(sum(exact[t, s]) for t in tasks) / sum(len(exact[t, s]) for t in tasks) for s in samples]
    mean, sd = mean_sd([100 * p for p in pooled])
    n_items = sum(len(bench[t]) for t in tasks)
    print(f"| Overall | {n_items} | {mean:.1f}" + (f" +- {sd:.1f} |" if sd_note else " |"))
    if missing:
        print(f"{missing} item answers missing (scored wrong)")
    if len(tasks) < len(TASK_NAMES):
        print(f"Overall covers only {', '.join(tasks)}")
    if args.out:
        args.out.write_text("".join(json.dumps(row) + "\n" for row in scored))


# ---------------------------------------------------------------------------------------------------------------------
# `tables`: Table 4 from the stored per-item `exact_ok` (API rows: one draw; Qwen3.5-9B rows: mean +- SD over 3 decode
# seeds) and the failure categories quoted in the text.

API_ROWS = {"gpt-6-astra": "GPT-6 Astra", "claude-opus-5.5": "Claude Opus 5.5", "gpt-5.6-sol": "GPT-5.6 Sol"}
API_ROWS.update({"gemini-3.8-flash": "Gemini 3.8 Flash", "claude-opus-5": "Claude Opus 5"})
API_ROWS["qwen3.8-max"] = "Qwen3.8-Max"
SFT_ROWS = {"qwen3.5-9b": "Qwen3.5-9B (base)", "brickvision_direct_s17": "Finetuned Direct"}
SFT_ROWS.update(
    brickvision_text_s17="Finetuned Text Reasoning", brickvision_grounded_s17="Finetuned Grounded Reasoning"
)


def api_rows(name, task):
    return read_jsonl(RESULTS / name / f"{task}.jsonl")


def api_table(out):
    exact = {}
    for n in API_ROWS:
        for t in TASK_NAMES:
            rows = {r["idx"]: r for r in api_rows(n, t) if r.get("error") is None}
            n_items = len(read_jsonl(DATA / "restyled" / t / "items.jsonl"))
            assert sorted(rows) == list(range(n_items)), (n, t, "missing rows: rerun models.py api")
            exact[n, t] = {i: bool(r.get("exact_ok", r.get("correct"))) for i, r in rows.items()}
    print("| model | " + " | ".join(TASK_NAMES.values()) + " | Overall |\n|---|" + "---|" * (len(TASK_NAMES) + 1))
    for n, label in API_ROWS.items():
        cells = []
        for t in TASK_NAMES:
            d = exact[n, t]
            out[f"{n}/{t}"] = {"exact": sum(d.values()) / len(d), "n": len(d)}
            cells.append(f"{100 * sum(d.values()) / len(d):.1f}")
        hit, tot = sum(sum(exact[n, t].values()) for t in TASK_NAMES), sum(len(exact[n, t]) for t in TASK_NAMES)
        out[f"{n}/pooled"] = {"exact": hit / tot, "n": tot}
        print(f"| {label} | " + " | ".join(cells) + f" | {100 * hit / tot:.1f} |")


def hamming(a, b):
    return sum(json.dumps(a[g]) != json.dumps(b[g]) for g in GEOM_ATTRS)


def failure_categories():
    """Failure categories of the API rows (fractions of all items unless noted); Crop ID uses the first point only;
    `no_twin` = both Duplicate points on non-twins; Counting = signed count error."""
    out = {}
    sc = {s["scene_id"]: s for s in read_jsonl(DATA / "synthetic/duplicate/scenes.jsonl")}
    for m in API_ROWS:
        c, picks, exp1 = Counter(), Counter(), 0.0
        for r in api_rows(m, "duplicate"):
            s = sc[r["scene_id"]]
            designs = {x["index"]: x["design"] for x in s["stacks"]}
            twins = s["duplicate_group"]
            labs = [v["exact_label"] for v in r.get("verdicts") or []]
            if r["exact_ok"]:
                c["correct"] += 1
            elif len(labs) != 2 or labs[0] == labs[1]:
                c["malformed"] += 1
            elif 0 in labs:
                c["off"] += 1
            else:
                c["one_twin" if len(set(twins) & {labs[0] - 1, labs[1] - 1}) == 1 else "no_twin"] += 1
                distractors = [j for j in designs if j not in twins]
                for lab in labs:
                    if lab - 1 not in twins:
                        picks["h1" if hamming(designs[lab - 1], designs[twins[0]]) == 1 else "h2+"] += 1
                        exp1 += np.mean([hamming(designs[j], designs[twins[0]]) == 1 for j in distractors])
        n = sum(picks.values())
        out[f"duplicate/{m}"] = {**{k: v / 250 for k, v in c.items()}, "wrong_pick_h1": picks["h1"] / n}
        out[f"duplicate/{m}"]["wrong_pick_h1_chance"] = exp1 / n
    sc = {s["scene_id"]: s for s in read_jsonl(DATA / "synthetic/unpaired/scenes.jsonl")}
    for m in API_ROWS:
        c, exp1, n_wrong = Counter(), 0.0, 0
        for r in api_rows(m, "unpaired"):
            s = sc[r["scene_id"]]
            designs = {x["index"]: x["design"] for x in s["stacks"]}
            single = s["task"]["singleton_stack"]
            labs = [v["exact_label"] for v in r.get("verdicts") or []]
            if r["exact_ok"]:
                c["correct"] += 1
            elif len(labs) != 1:
                c["malformed"] += 1
            elif labs[0] == 0:
                c["off"] += 1
            else:
                c["near_h1" if hamming(designs[labs[0] - 1], designs[single]) == 1 else "unrelated_h2+"] += 1
                n_wrong += 1
                exp1 += np.mean([hamming(designs[j], designs[single]) == 1 for j in designs if j != single])
        out[f"unpaired/{m}"] = {**{k: v / 250 for k, v in c.items()}, "near_share_of_wrong": c["near_h1"] / n_wrong}
        out[f"unpaired/{m}"]["near_chance"] = exp1 / n_wrong
    items = {i["scene_id"]: i for i in read_jsonl(DATA / "restyled/crop_id/items.jsonl")}
    sc = {s["scene_id"]: s for s in read_jsonl(DATA / "synthetic/crop_id/scenes.jsonl")}
    frames = {}  # scene_id -> full-composite brick-id map (H, W)
    for sid, s in sc.items():
        idm = render_id_mask(s)
        w, h = items[sid]["image_wh"]
        frames[sid] = np.zeros((h, w), np.int32)
        ox, oy = s["crop"]["full_offset"]
        frames[sid][oy : oy + idm.shape[0], ox : ox + idm.shape[1]] = idm
    for m in API_ROWS:
        c = Counter()
        for r in api_rows(m, "crop_id"):
            s, frame = sc[r["scene_id"]], frames[r["scene_id"]]
            if not r["verdicts"]:  # no point in the reply
                c["no_point"] += 1
                continue
            x, y = r["verdicts"][0]["point_xy"]
            bid = frame[min(max(y, 0), frame.shape[0] - 1), min(max(x, 0), frame.shape[1] - 1)] - 1
            target = s["bricks"][s["crop"]["highlight_brick"]]
            if r["exact_ok"]:
                c["correct"] += 1
            elif bid < 0:
                c["off"] += 1
            elif s["bricks"][bid]["pod_index"] == target["pod_index"]:
                c["right_struct_wrong_piece"] += 1
            else:
                same = s["bricks"][bid]["construction_role"] == target["construction_role"]
                c["wrong_struct_same_piece" if same else "wrong_struct_other_piece"] += 1
        out[f"crop_id/{m}"] = {k: v / 250 for k, v in c.items()}
    for m in API_ROWS:
        errors = []
        for r in api_rows(m, "counting"):
            p = r.get("pred", r.get("pred_answer"))
            ok = p is not None and str(p).lstrip("-").isdigit()
            errors.append(int(p) - int(r.get("gold", r.get("answer"))) if ok else None)
        v, n = [x for x in errors if x is not None], len(errors)
        out[f"counting/{m}"] = {
            "under": sum(x < 0 for x in v) / n,
            "exact": sum(x == 0 for x in v) / n,
            "over": sum(x > 0 for x in v) / n,
            "off_by_one": sum(abs(x) == 1 for x in v) / n,
            "off_by_3plus": sum(abs(x) >= 3 for x in v) / n,
            "mean_err": float(np.mean(v)),
            "unparsed": (n - len(v)) / n,
        }
    with open(RESULTS / "section_table.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nfailure categories -> {RESULTS / 'section_table.json'}")


def sft_table(out):
    """Exact % mean +- SD over decode seeds (unclosed / unparsed = wrong); pooled = mean over seeds of all items."""
    print("\n| SFT row (mean +- SD over seeds) | " + " | ".join(TASK_NAMES.values()) + " | Overall |")
    print("|---|" + "---|" * (len(TASK_NAMES) + 1))
    for name, label in SFT_ROWS.items():
        row, pooled_by_seed = [], defaultdict(list)
        for task in TASK_NAMES:
            seeds = {}
            for f in sorted((RESULTS / name).glob(f"{task}_s*.jsonl")):
                seeds[int(re.search(r"_s(\d+)\.jsonl$", f.name).group(1))] = [
                    bool(r["exact_ok"]) for r in read_jsonl(f)
                ]
            accs = [100 * sum(v) / len(v) for v in seeds.values()]
            for s, v in seeds.items():
                pooled_by_seed[s] += v
            m = sum(accs) / len(accs)
            sd = math.sqrt(sum((x - m) ** 2 for x in accs) / (len(accs) - 1))
            out[f"{task}/{name}"] = {"mean": m, "sd": sd, "seeds": accs}
            row.append(f"{m:.1f} ± {sd:.1f}")
        pooled = [100 * sum(v) / len(v) for v in pooled_by_seed.values()]
        out[f"pooled/{name}"] = {"mean": sum(pooled) / len(pooled), "seeds": pooled}
        print(f"| {label} | " + " | ".join(row) + f" | {sum(pooled) / len(pooled):.1f} |")


def tables():
    api, sft = {}, {}
    api_table(api)
    failure_categories()
    sft_table(sft)
    (RESULTS / "tables.json").write_text(json.dumps({"api": api, "sft": sft}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score BrickVision predictions or rebuild Table 4.")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score", help="score predictions (or write the prompts)")
    p.add_argument("--predictions", type=Path, help="JSONL rows {task, scene_id, response = final answer[, sample]}")
    p.add_argument("--data", type=Path, default=DATA / "restyled", help="bench dir holding <task>/items.jsonl")
    p.add_argument("--pixel", action="store_true", help="points are image pixels instead of 0-1000")
    p.add_argument("--out", type=Path, help="write per-item scores (JSONL)")
    p.add_argument("--prompts", type=Path, help="write {task, scene_id, image, prompt} rows and exit")
    sub.add_parser("tables", help="Table 4 + failure categories from $GR_RESULTS/brickvision")
    args = ap.parse_args()
    if args.command == "score":
        score(args, p.error)
    else:
        tables()


if __name__ == "__main__":
    main()
