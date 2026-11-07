"""SFT data: TVG train questions -> two-stage Gemini traces -> the SFT sets -> LLaMA-Factory dataset dirs.
Sets: text, grounded (+ _unfiltered), grounded_box, appended, grounded_rand{05,20,50,100}; images relative to $GR_DATA.
    python -m grounded_reasoning.data cache [--parquet-glob G]  # HF JunkaiZ/TVG -> $GR_DATA/tvg/<subset>/
    python -m grounded_reasoning.data generate --subset S       # Gemini (GEMINI_API_KEY) -> $GR_DATA/sft/generations/
    python -m grounded_reasoning.data sets                      # -> $GR_DATA/sft/<set>.json
    python -m grounded_reasoning.data llamafactory <set.json>   # -> <dir>/lf/<set>/ (LLaMA-Factory dataset_dir)
"""

import argparse
import glob
import io
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from grounded_reasoning import DATA
from grounded_reasoning.benchmarks import SUFFIX

TVG_DIR = DATA / "tvg"
SFT_DIR = DATA / "sft"
GENERATIONS_DIR = SFT_DIR / "generations"
SUBSETS = {"smqa": "spatialmqa", "mhs": "multihopspatial"}  # tag in the `source` field ("smqa_2s2") -> subset name

ANSWER_LETTER_LINE = "Answer with the option letter."  # trailing line of the SpatialMQA questions, dropped
ANS_LINE_RE = re.compile(r"\n\s*Answer\s*:\s*\(?([a-f])\)?\s*\.?\s*$", re.I)  # "Answer: (x)" line of a trace
PT_RE = re.compile(r"\[\s*-?\d+\s*,\s*-?\d+\s*\]")


def load_json(path: Path):
    with open(path) as handle:
        return json.load(handle)


def load_cache(subset: str) -> list[dict]:
    """TVG cache rows {idx, image, question, answer, objects: [{phrase, points, boxes}]} (0-1000 frame)."""
    with open(TVG_DIR / SUBSETS[subset] / "meta.jsonl") as handle:
        return [json.loads(line) for line in handle]


def load_generations(subset: str) -> list[dict]:
    with open(GENERATIONS_DIR / f"{SUBSETS[subset]}.jsonl") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def human_turn(question: str) -> str:
    question = question.strip()
    if question.endswith(ANSWER_LETTER_LINE):
        question = question[: -len(ANSWER_LETTER_LINE)].strip()
    return f"<image>\n{question}{SUFFIX}"


def to_target(trace: str, letter: str) -> str:
    """Trace minus its 'Answer: (x)' line -> '<think>\\n..\\n</think>\\n\\n\\boxed{letter}'."""
    inner = ANS_LINE_RE.sub("", trace.rstrip()).rstrip()
    return f"<think>\n{inner}\n</think>\n\n\\boxed{{{letter}}}"


def llava_row(d: dict, source: str, trace: str) -> dict:
    """One generation row + chosen trace -> SFT row."""
    return {
        "image": d["image"],
        "source": source,
        "pair_idx": d["idx"],
        "conversations": [
            {"from": "human", "value": human_turn(d["question"])},
            {"from": "gpt", "value": to_target(trace, d["answer"].strip().lower())},
        ],
    }


def is_valid_trace(d: dict, field: str, grounded: bool) -> bool:
    """The trace ends on the GT letter and carries coordinates iff `grounded`."""
    trace = d[field]
    m = ANS_LINE_RE.search(trace.rstrip())
    ok = bool(m) and m.group(1).lower() == d["answer"].strip().lower()
    return ok and bool(PT_RE.search(trace)) == grounded


def row_key(row: dict) -> tuple[str, int]:
    """(subset, pair_idx), the question shared by every set."""
    return row["source"].split("_")[0], row["pair_idx"]


def write_json(rows: list[dict], path: Path, ensure_ascii: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(rows, handle, ensure_ascii=ensure_ascii)


# ---- cache: TVG parquet -> images + meta.jsonl; rows with point grounding are numbered 0..N-1 in parquet order

DATASET_NAMES = {"smqa": "spatial_mqa", "mhs": "multihop_spatial"}
HF_HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def grounded_objects(supervision: dict) -> list[dict]:
    """TVG point objects joined with their boxes (one box per point, same objects in the same order)."""
    points = [
        {"phrase": o["phrase"], "points": [[float(x), float(y)] for x, y in o["points"]]}
        for o in (supervision.get("point_grounded_objects") or [])
        if o.get("points")
    ]
    boxes = [
        {"phrase": o["phrase"], "boxes": [[float(a) for a in b] for b in o["boxes"]]}
        for o in (supervision.get("box_grounded_objects") or [])
        if o.get("boxes")
    ]
    if [o["phrase"] for o in points] != [o["phrase"] for o in boxes] or any(
        len(p["points"]) != len(b["boxes"]) for p, b in zip(points, boxes)
    ):
        raise ValueError(f"point and box objects do not pair up: {points} vs {boxes}")
    return [{**p, "boxes": b["boxes"]} for p, b in zip(points, boxes)]


def write_cache(files: list[str], subset: str) -> None:
    import pyarrow.parquet as pq

    cache = TVG_DIR / SUBSETS[subset]
    (cache / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(cache / "meta.jsonl", "w") as fout:
        for path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for r in batch.to_pylist():
                    if (r.get("extra_info") or {}).get("dataset_name") != DATASET_NAMES[subset]:
                        continue
                    supervision = r.get("grounding_supervision") or {}
                    if not supervision.get("available") or not any(
                        o.get("points") for o in supervision.get("point_grounded_objects") or []
                    ):
                        continue
                    image = f"tvg/{SUBSETS[subset]}/images/{n}.jpg"
                    Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(DATA / image, quality=92)
                    row = {
                        "idx": n,
                        "image": image,
                        "question": r["extra_info"]["question"],
                        "answer": r["reward_model"]["ground_truth"],
                        "objects": grounded_objects(supervision),
                    }
                    fout.write(json.dumps(row) + "\n")
                    n += 1


# ---- generate: judge (rejects bad GT) -> stage 1 (coordinate-free trace, clean image) -> stage 2 (rewrite that
# locates and looks at every GT [x, y], marked image); each stage retries until its validator passes

MAGENTA = (255, 0, 255)
GEMINI_MODEL = "gemini-3.5-flash"  # judge, stage 1 and stage 2
TEMPERATURE = 0.8  # stage 1 and stage 2 (judge: 0.2)
WORKERS = 24

JUDGE_SYSTEM = (
    "You are a strict data-quality judge for a grounded visual-reasoning training set. You "
    "are given an image on which candidate ground-truth points are drawn as numbered "
    "markers, a legend giving each numbered point's descriptive phrase and exact [x, y] "
    "coordinate (normalized 0-1000), a multiple-choice spatial question about the image, "
    "and the labeled correct answer.\n\n"
    "Decide whether this row's ground truth is GOOD enough to teach from. Reject it if ANY "
    "of these hold:\n"
    "- WRONG POINT: some numbered marker does not lie on the object its phrase names.\n"
    "- BAD COVERAGE: the set of given points does not cover the objects that must be "
    "located to answer the question — every object (or text element) whose position the "
    "question asks about must have its own point (e.g. a question comparing letter T to "
    "letter O needs a point for EACH letter, not just one; a question about A relative to B "
    "needs both A and B, unless one of them is the viewer/camera).\n"
    "- WRONG ANSWER: the labeled answer contradicts what the image clearly shows.\n"
    "- UNANSWERABLE: the question is ambiguous or cannot be answered from the image.\n\n"
    'Reply with ONLY a JSON object: {"verdict": "ok" | "bad", "reason": "<one short '
    'sentence>"}.'
)

STAGE1_SYSTEM = (
    "You are given an image, a multiple-choice question about it, and the correct answer. "
    "Write a detailed first-person reasoning trace that explains how to arrive at that "
    "answer by looking at the image — as if you are working it out and do not yet know the "
    "answer.\n\n"
    "Describe positions in words rather than coordinates. Never reveal that you were told "
    "the answer. Stay faithful to the image: do not invent details you cannot see. Output "
    "only the trace, ending with a line on its own: Answer: (x)"
)

REWRITE_SYSTEM = (
    "You are given an image with the relevant objects marked, a legend giving each marked "
    "object's descriptive phrase and its exact [x, y] point coordinate (normalized 0-1000, "
    "where [0, 0] is the top-left corner and [1000, 1000] the bottom-right corner of the "
    "image), a multiple-choice question about the image with its correct option, and a draft "
    "reasoning trace written by someone who looked at the same image but did not have the "
    "coordinates.\n\n"
    "Rewrite the draft so that it reasons from the coordinates, in two moves.\n\n"
    "LOCATE. When the draft first brings up an object that is in the legend, say where you "
    'find it, giving its exact [x, y] coordinate: "I find the bicycle at [212, 640]."\n\n'
    "LOOK. Then look at that coordinate and say what you see there, and let the draft's own "
    "description of the object be what you report seeing. Address the coordinate itself, not "
    "just the object's name — write \"Looking at [212, 640], I see a bicycle leaning against "
    'the weathered brick wall, its chrome handlebars turned sharply toward the curb." and NOT '
    '"Looking at the bicycle, it leans against the wall." Do this wherever the draft '
    "describes or re-examines that object, at that point in the draft's reasoning. Every time "
    "you write a coordinate after its locate, it should be you looking at that coordinate and "
    "reporting what is there.\n\n"
    "Do both moves for every object you are given a coordinate for, wherever the logic of the "
    "draft allows it — the wording is yours, only the shape (locate the point, then look at "
    "the point and describe what is there) is fixed.\n\n"
    "Keep everything else the draft does: its reasoning, its observations, its order, and all "
    "of its detail — where the draft describes something closely, describe it just as closely.\n\n"
    "Use each coordinate you are given verbatim, inside [x, y] brackets. Write as if you "
    "simply looked at the image — never mention the markers, their numbers, or anything drawn "
    "on the image. End with a line on its own: Answer: (x)"
)

# any mention of the drawn annotation
LEAK_RE = re.compile(
    r"\b(marked|markers?|highlighted|annotat\w+|magenta|crosshair|numbered|"
    r"green (?:dot|marker|point)|target|"
    r"(?:point|marker|label|number)\s+#?\d+\b|"
    r"(?:first|second|third|fourth|fifth) (?:point|marker))\b",
    re.I,
)
ANS_RE = re.compile(r"(?i)answer\s*:\s*\(?([a-f])\)?")
JSON_RE = re.compile(r"\{.*\}", re.S)
# a look verb within 60 chars before the coordinate, with no sentence end in between: "Looking at [491, 570], I see"
LOOK_ANCHOR_RE = r"(?:look(?:ing|s|ed)?|see|examin\w+|inspect\w+|turn\w* (?:my|to))\b[^.!?\[]{0,60}$"


def fmt_pt(x, y):
    return f"[{int(round(x))}, {int(round(y))}]"


def coord_pat(x, y):
    return re.compile(rf"\[\s*{int(round(x))}\s*,\s*{int(round(y))}\s*\]")


def numbered_points(rec):
    """Objects -> [(num, phrase, x, y)], 1-based, legend order."""
    out = []
    for o in rec["objects"]:
        for x, y in o["points"]:
            out.append((len(out) + 1, o["phrase"], x, y))
    return out


def build_legend(rec):
    return "; ".join(f"{num}. {phrase}: {fmt_pt(x, y)}" for num, phrase, x, y in numbered_points(rec))


def draw_marker(im, x_pct, y_pct, color=MAGENTA):
    """Crosshair with a centre gap + ring + dot on an RGB copy, sized with the image (capped)."""
    im = im.convert("RGB").copy()
    W, H = im.size
    x, y = x_pct / 100.0 * W, y_pct / 100.0 * H
    d = ImageDraw.Draw(im)
    r = max(4, min(18, int(min(W, H) * 0.008)))
    w = 2 if min(W, H) >= 400 else 1
    arm = int(r * 1.9)
    g = max(2, r // 2)  # centre gap
    d.line([(x - arm, y), (x - g, y)], fill=color, width=w)
    d.line([(x + g, y), (x + arm, y)], fill=color, width=w)
    d.line([(x, y - arm), (x, y - g)], fill=color, width=w)
    d.line([(x, y + g), (x, y + arm)], fill=color, width=w)
    d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=w)
    d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)  # exact-pixel dot
    return im


def marked_image(rec):
    """All GT points as magenta markers, each labeled with its legend number."""
    im = Image.open(DATA / rec["image"]).convert("RGB")
    pts = numbered_points(rec)
    for _, _, x, y in pts:  # 0-1000 -> percent for draw_marker
        im = draw_marker(im, x / 10.0, y / 10.0)
    W, H = im.size
    r = max(4, min(18, int(min(W, H) * 0.008)))
    size = max(11, min(30, int(min(W, H) * 0.022)))
    font = ImageFont.load_default(size=size)
    d = ImageDraw.Draw(im)
    for num, _, xp, yp in pts:
        x, y = xp / 1000.0 * W, yp / 1000.0 * H
        tx = min(max(x + r * 2.2, 2), W - size)
        ty = min(max(y - r * 2.2 - size, 2), H - size)
        d.text((tx, ty), str(num), fill=MAGENTA, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return im


def build_judge_task(rec):
    return (
        f'Question:\n"""\n{rec["question"]}\n"""\n\n'
        f"Numbered points (normalized 0-1000, [x, y]): {build_legend(rec)}.\n"
        f"Labeled correct answer: ({rec['answer'].strip().lower()}).\n"
        f"Is this ground truth good? Reply with the JSON verdict only."
    )


def build_task_stage1(rec):
    ans = rec["answer"].strip().lower()
    return (
        f'Question:\n"""\n{rec["question"]}\n"""\n\n'
        f"The correct answer is ({ans}).\n"
        f"Write the trace; end with the line: Answer: ({ans})"
    )


def build_task_rewrite(rec, draft):
    ans = rec["answer"].strip().lower()
    return (
        f'Question:\n"""\n{rec["question"]}\n"""\n\n'
        f"Located objects, numbered as on the image (coordinates normalized 0-1000, "
        f"[x, y]): {build_legend(rec)}.\n"
        f"The correct option is ({ans}).\n\n"
        f'DRAFT TRACE:\n"""\n{draft}\n"""\n\n'
        f"Rewrite the draft to reason from these coordinates, keeping its reasoning and "
        f"structure. End with the line: Answer: ({ans})"
    )


def look_anchored(trace, x, y):
    """True if some mention of [x, y] is addressed by a look verb."""
    pat, anchor = coord_pat(x, y), re.compile(LOOK_ANCHOR_RE, re.I)
    return any(anchor.search(trace[: m.start()]) for m in pat.finditer(trace))


def validate_stage1(trace, rec):
    """Long enough, no leak, no coordinate, final answer = GT. -> (ok, reason)."""
    if not trace or len(trace.strip()) < 80:
        return False, "empty/too-short trace"
    m = LEAK_RE.search(trace)
    if m:
        return False, f"leak: '{m.group(0)}'"
    m = PT_RE.search(trace)
    if m:
        return False, f"stage-1 trace contains a coordinate '{m.group(0)}'"
    am = ANS_RE.findall(trace)
    if not am:
        return False, "no 'Answer: (x)' line"
    if am[-1].lower() != rec["answer"].strip().lower():
        return False, f"final answer ({am[-1]}) != GT ({rec['answer']})"
    return True, ""


def validate_rewrite(trace, rec):
    """Long enough, no leak, final answer = GT, every GT coordinate verbatim and looked at. -> (ok, reason)."""
    if not trace or len(trace.strip()) < 80:
        return False, "empty/too-short trace"
    m = LEAK_RE.search(trace)
    if m:
        return False, f"leak: '{m.group(0)}'"
    am = ANS_RE.findall(trace)
    if not am:
        return False, "no 'Answer: (x)' line"
    if am[-1].lower() != rec["answer"].strip().lower():
        return False, f"final answer ({am[-1]}) != GT ({rec['answer']})"
    missing = [fmt_pt(x, y) for _, _, x, y in numbered_points(rec) if not coord_pat(x, y).search(trace)]
    if missing:
        return False, f"did not point to {missing}"
    unlooked = [fmt_pt(x, y) for _, _, x, y in numbered_points(rec) if not look_anchored(trace, x, y)]
    if unlooked:
        return False, (
            f'did not look at {unlooked} — each point needs a "Looking at '
            f'[x, y], I see ..." that describes what is there'
        )
    return True, ""


def gemini(client, types, system, contents, temperature, max_tokens, thinking_budget):
    """One Gemini call -> stripped text."""
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=cfg)
    return (resp.text or "").strip()


def judge_gt(client, types, rec, im, retries=4):
    """-> (verdict "ok" | "bad", reason)."""
    for attempt in range(retries):
        try:
            text = gemini(client, types, JUDGE_SYSTEM, [im, build_judge_task(rec)], 0.2, 400, 512)
            m = JSON_RE.search(text)
            if m:
                v = json.loads(m.group(0))
                if v.get("verdict") in ("ok", "bad"):
                    return v["verdict"], v.get("reason", "")
        except Exception:
            time.sleep(min(2**attempt, 20))
    return "bad", "judge failed to return a verdict"


def run_stage(client, types, rec, system, im, task, validate, correction, retries=5):
    """Call until `validate` passes, appending a correction naming the last rejection. -> trace | None."""
    fix = ""
    for attempt in range(retries):
        try:
            trace = gemini(client, types, system, [im, task + fix], TEMPERATURE, 3000, 1024)
            ok, why = validate(trace, rec)
            if ok:
                return trace
            fix = " CORRECTION (previous attempt rejected): " + why + correction
        except Exception as e:
            time.sleep(min(2**attempt, 20))
            if attempt == retries - 1:
                print(f"  idx={rec['idx']} {type(e).__name__}: {e}", flush=True)
    return None


def describe_one(client, types, rec, judge):
    """Judge (optional) + stage 1 + stage 2 for one cache row. -> row, skip row, or None (failed)."""
    im = marked_image(rec)
    base = {k: rec[k] for k in ("idx", "image", "question", "answer")}
    base["objects"] = [{"phrase": o["phrase"], "points": o["points"]} for o in rec["objects"]]
    if judge:
        verdict, reason = judge_gt(client, types, rec, im)
        if verdict != "ok":
            return {**base, "skip": f"GT judge: {reason}"}
    ans = rec["answer"].strip().lower()
    clean = Image.open(DATA / rec["image"]).convert("RGB")
    draft = run_stage(
        client,
        types,
        rec,
        STAGE1_SYSTEM,
        clean,
        build_task_stage1(rec),
        validate_stage1,
        ". Describe positions in words with no coordinates, never mention anything drawn on the image, and end "
        f"with the exact line 'Answer: ({ans})'.",
    )
    if draft is None:
        return None
    rewrite = run_stage(
        client,
        types,
        rec,
        REWRITE_SYSTEM,
        im,
        build_task_rewrite(rec, draft),
        validate_rewrite,
        ". Keep the draft's reasoning, order and detail; locate each object with its exact coordinate and then "
        'look at that coordinate and describe what is there ("Looking at [x, y], I see ..."); use every given '
        "coordinate verbatim inside [x, y] brackets; never mention markers, their numbers, anything drawn on "
        f"the image, or the word 'target'; end with the exact line 'Answer: ({ans})'.",
    )
    if rewrite is None:
        return None
    return {**base, "draft": draft, "thinking": rewrite}


def generate_traces(name: str) -> None:
    """Appends to generations/<name>.jsonl, skipping idxs already there; the judge is skipped for the idxs listed
    in generations/<name>_judge_ok.json when that file exists."""
    subset = next(k for k, v in SUBSETS.items() if v == name)
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    okset_path, out_path = GENERATIONS_DIR / f"{name}_judge_ok.json", GENERATIONS_DIR / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = set(load_json(okset_path)) if okset_path.exists() else None
    done = set()
    if out_path.exists():
        done = {json.loads(line)["idx"] for line in out_path.read_text().splitlines()}
    todo = [r for r in load_cache(subset) if r["idx"] not in done and (keep is None or r["idx"] in keep)]
    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(describe_one, client, types, rec, keep is None) for rec in todo]
        for fut in as_completed(futs):
            out = fut.result()
            if out is not None:
                fout.write(json.dumps(out) + "\n")
                fout.flush()


# ---- sets: text, grounded (+ _unfiltered), grounded_box, appended, grounded_rand{05,20,50,100}

MAX_POINTS = 7  # the filtered sets drop questions with >= this many GT points
FIELD = {"s1": "draft", "s2": "thinking"}  # s1 = stage-1 draft (text), s2 = stage-2 rewrite (grounded)


def build_half(subset: str, stage: str, max_points: int | None) -> list[dict]:
    npts = {r["idx"]: sum(len(o["points"]) for o in r["objects"]) for r in load_cache(subset)}
    rows = [
        llava_row(d, f"{subset}_2{stage}", d[FIELD[stage]])
        for d in load_generations(subset)
        if not d.get("skip")
        and d.get(FIELD[stage])
        and is_valid_trace(d, FIELD[stage], grounded=stage == "s2")
        and not (max_points and npts.get(d["idx"], 0) >= max_points)
    ]
    random.Random(17).shuffle(rows)
    return rows


def build_set(stage: str, max_points: int | None) -> list[dict]:
    """Seed-17 shuffle per subset, then SpatialMQA + MultihopSpatial concatenated and shuffled with seed 17."""
    rows = build_half("smqa", stage, max_points) + build_half("mhs", stage, max_points)
    random.Random(17).shuffle(rows)
    return rows


def build_text_grounded() -> None:
    """`text` (stage-1 draft) and `grounded` (stage-2 rewrite), filtered and unfiltered."""
    sets = {
        "text_unfiltered": build_set("s1", None),
        "grounded_unfiltered": build_set("s2", None),
        "text": build_set("s1", MAX_POINTS),
        "grounded": build_set("s2", MAX_POINTS),
    }
    for name, rows in sets.items():  # `text` was released UTF-8 encoded, the others ASCII-escaped
        write_json(rows, SFT_DIR / f"{name}.json", ensure_ascii=name != "text")


BOX_PT_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")
SNAP_CAP = 100  # L-inf ceiling for snapping a truncated coordinate to its GT point


def point_to_box(cache_row: dict) -> dict:
    """(x, y) -> box over all objects in legend order; the first object wins a shared point."""
    lut = {}
    for obj in cache_row["objects"]:
        for pt, bx in zip(obj["points"], obj["boxes"]):
            lut.setdefault((int(round(pt[0])), int(round(pt[1]))), [int(round(v)) for v in bx])
    return lut


def build_grounded_box() -> None:
    """`grounded` with each [x, y] replaced by its GT {"bbox_2d": [x1, y1, x2, y2]} (positional lookup in the cache;
    a coordinate missing from it snaps to the unique nearest GT point within SNAP_CAP)."""
    caches = {s: {r["idx"]: r for r in load_cache(s)} for s in SUBSETS}
    out = []
    for r in load_json(SFT_DIR / "grounded.json"):
        lut = point_to_box(caches["smqa" if r["source"].startswith("smqa") else "mhs"][r["pair_idx"]])

        def repl(m, lut=lut):
            k = (int(m.group(1)), int(m.group(2)))
            bx = lut.get(k)
            if bx is None:
                cands = sorted((max(abs(k[0] - a), abs(k[1] - b)), (a, b)) for a, b in lut)
                if not cands or cands[0][0] > SNAP_CAP or (len(cands) > 1 and cands[0][0] == cands[1][0]):
                    return m.group(0)
                bx = lut[cands[0][1]]
            return '{"bbox_2d": ' + json.dumps(bx) + "}"

        row = dict(r)
        row["source"] = r["source"].replace("_2s2", "_2s2box")
        row["conversations"] = [
            dict(r["conversations"][0]),
            {**r["conversations"][1], "value": BOX_PT_RE.sub(repl, r["conversations"][1]["value"])},
        ]
        out.append(row)
    write_json(out, SFT_DIR / "grounded_box.json", ensure_ascii=False)


STOP = {
    "the", "a", "an", "of", "on", "in", "at", "with", "and", "that", "this", "side",
    "image", "picture", "left", "right", "top", "bottom", "front", "behind",
}  # fmt: skip


def head_word(phrase: str) -> str | None:
    words = [w for w in re.findall(r"[a-z]+", phrase.lower()) if w not in STOP]
    return words[0] if words else None


def paste(draft: str, objects: list[dict]) -> str:
    """Each GT " [x, y]" after the first whole-word match of its phrase, else of a word starting with its head word."""
    text = draft
    for o in objects:
        phrase = o["phrase"].strip()
        for x, y in o["points"]:
            coord = f" [{int(round(x))}, {int(round(y))}]"
            m = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.I) if phrase else None
            if not m:
                h = head_word(phrase)
                m = re.search(rf"(?<!\w){re.escape(h)}\w*", text, re.I) if h else None
            if m:
                text = text[: m.end()] + coord + text[m.end() :]
    return text


def build_appended() -> None:
    """`appended`: the stage-1 draft with the GT points pasted in, rows in `grounded` order."""
    by_key = {}
    for subset in SUBSETS:
        cache = {r["idx"]: r for r in load_cache(subset)}
        for d in load_generations(subset):
            if not d.get("skip") and d.get("draft") and is_valid_trace(d, "draft", grounded=False):
                row = llava_row(d, f"{subset}_paste", paste(d["draft"], cache[d["idx"]]["objects"]))
                by_key[row_key(row)] = row
    rows = [by_key[row_key(k)] for k in load_json(SFT_DIR / "grounded.json")]
    write_json(rows, SFT_DIR / "appended.json", ensure_ascii=False)


DOSES = [5, 20, 50, 100]
RAND_PT_RE = re.compile(r"\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]")
BX_RE = re.compile(r'\{"bbox_2d":\s*\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]\}')


def random_box_1k(rng):
    x1 = rng.randint(0, 899)
    y1 = rng.randint(0, 899)
    return x1, y1, rng.randint(x1 + 20, 1000), rng.randint(y1 + 20, 1000)


def gpt_i(row):
    return next(i for i, c in enumerate(row["conversations"]) if c["from"] == "gpt")


def rewrite_points(text, slot_repl):
    """Replace the k-th coordinate mention by slot_repl[k]."""
    out, last = [], 0
    for k, m in enumerate(RAND_PT_RE.finditer(text)):
        new = slot_repl.get(k)
        out += [text[last : m.start()], f"[{new[0]}, {new[1]}]" if new else m.group(0)]
        last = m.end()
    return "".join(out) + text[last:]


def build_groups(grounded_rows, box_rows):
    """-> [(row index, box value, {point value: [slots]})], one entry per distinct GT box per row."""
    groups = []
    for ri, (rp, rb) in enumerate(zip(grounded_rows, box_rows)):
        pts = [tuple(map(int, m.groups())) for m in RAND_PT_RE.finditer(rp["conversations"][gpt_i(rp)]["value"])]
        bxs = [tuple(map(int, m.groups())) for m in BX_RE.finditer(rb["conversations"][gpt_i(rb)]["value"])]
        slots_by_box = {}
        for k, bv in enumerate(bxs):
            slots_by_box.setdefault(bv, []).append(k)
        for bv, slots in slots_by_box.items():
            pmap = {}
            for k in slots:
                pmap.setdefault(pts[k], []).append(k)
            groups.append((ri, bv, pmap))
    return groups


def build_grounded_rand() -> None:
    """`grounded` with the points of 5/20/50/100 % of GT objects randomized: one seed-0 shuffle of all (row, box)
    units, dose X% takes the first round(X% N); each point value of a unit gets one fixed random point != itself."""
    grounded_rows = load_json(SFT_DIR / "grounded.json")
    groups = build_groups(grounded_rows, load_json(SFT_DIR / "grounded_box.json"))
    rng = random.Random(0)
    order = list(range(len(groups)))
    rng.shuffle(order)
    pt_repl = {}
    for gi in order:
        _, bv, pmap = groups[gi]
        while random_box_1k(rng) == bv:  # drawn and discarded: fixes the RNG stream of the released sets
            pass
        pr = {}
        for pv in pmap:
            new = (rng.randint(0, 1000), rng.randint(0, 1000))
            while new == pv:
                new = (rng.randint(0, 1000), rng.randint(0, 1000))
            pr[pv] = new
        pt_repl[gi] = pr

    for pct in DOSES:
        slots_by_row = {}
        for gi in order[: round(pct / 100 * len(groups))]:
            ri, _, pmap = groups[gi]
            for pv, slots in pmap.items():
                slots_by_row.setdefault(ri, {}).update(dict.fromkeys(slots, pt_repl[gi][pv]))
        out = []
        for ri, row in enumerate(grounded_rows):
            r = {k: v for k, v in row.items() if k != "conversations"}  # conversations (from/value only) last
            r["conversations"] = [{"from": c["from"], "value": c["value"]} for c in row["conversations"]]
            if ri in slots_by_row:
                turn = r["conversations"][gpt_i(r)]
                turn["value"] = rewrite_points(turn["value"], slots_by_row[ri])
            out.append(r)
        with open(SFT_DIR / f"grounded_rand{pct:02d}.json", "w") as handle:
            json.dump(out, handle)


# ---- llamafactory: LLaVA set file -> ShareGPT dataset dir (dataset "vlm_sft"), the `dataset_dir` of configs/sft.yaml

DATASET_INFO = {
    "vlm_sft": {
        "file_name": "train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }
}


def to_sharegpt(record: dict) -> dict:
    """LLaVA row -> ShareGPT row: one '<image>' line per image (absolute path) leads the user turn."""
    raw_images = record["images"] if "images" in record else [record["image"]]
    images = [str((DATA / os.path.expanduser(str(path))).resolve()) for path in raw_images]
    user, assistant = record["conversations"][:2]
    prompt = "\n".join(["<image>"] * len(images) + [user["value"].replace("<image>", "").strip()])
    return {
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant["value"]}],
        "images": images,
        "source_id": str(record.get("id", "")),
    }


def llamafactory(set_files: list[str]) -> None:
    for arg in set_files:
        path = Path(arg).absolute()  # not resolve(): a symlinked set file keeps its lf/ dir beside the link
        out_dir = path.parent / "lf" / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        with path.open() as handle:
            rows = [to_sharegpt(record) for record in json.load(handle)]
        with (out_dir / "train.json").open("w") as handle:
            json.dump(rows, handle, ensure_ascii=False)
        with (out_dir / "dataset_info.json").open("w") as handle:
            json.dump(DATASET_INFO, handle, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cache = sub.add_parser("cache", help="TVG train parquet -> $GR_DATA/tvg/<subset>/")
    cache.add_argument("--parquet-glob", default=f"{HF_HUB}/datasets--JunkaiZ--TVG/snapshots/*/data/train-*.parquet")
    gen = sub.add_parser("generate", help="two-stage Gemini traces of one subset (paid API calls)")
    gen.add_argument("--subset", choices=list(SUBSETS.values()), required=True)
    sub.add_parser("sets", help="every SFT set from the generations (CPU)")
    lf = sub.add_parser("llamafactory", help="set file(s) -> LLaMA-Factory dataset dir(s)")
    lf.add_argument("set_files", nargs="+", metavar="set.json")
    args = ap.parse_args()
    if args.cmd == "cache":
        files = sorted(glob.glob(args.parquet_glob))
        for subset in SUBSETS:
            write_cache(files, subset)
    elif args.cmd == "generate":
        generate_traces(args.subset)
    elif args.cmd == "sets":
        build_text_grounded()  # text, grounded, *_unfiltered
        build_grounded_box()  # needs grounded
        build_appended()  # needs grounded
        build_grounded_rand()  # needs grounded, grounded_box
    else:
        llamafactory(args.set_files)


if __name__ == "__main__":
    main()
