"""GRPO after SFT: the verl reward, the prompt set (TreeVGR-RL-37K subset + a pass@8 band of TVG questions) and pass@8.
    python -m grounded_reasoning.rl build-data treevgr|band|verl  # -> $GR_DATA/rl/{treevgr_rl.parquet,band/,train/}
    python -m grounded_reasoning.rl passk <sft run dir> spatialmqa|multihopspatial <out.jsonl>  # vLLM, 1 GPU
verl loads `compute_score` from this file (reward.custom_reward_function.path), so its top-level imports stay light.
"""

import argparse
import base64
import io
import json
import os
import re
import time
import urllib.request
from collections import defaultdict

from PIL import Image

from grounded_reasoning import DATA, RESULTS
from grounded_reasoning.benchmarks import LETTERS, SUFFIX, extract_last_boxed, parse_mcq_index

# ---- reward R = w_acc * R_acc + w_fmt * R_format + w_gnd * R_gnd on the answer JSON of build-data
# {answer, kind (mcq|yesno|open), options, gold_index, boxes, prompt}. R_acc reads the LAST \boxed{}: mcq = the eval's
# option parser, yesno = leading yes/no, else (and open) a YES/NO answer-equivalence judge at $JUDGE_URL.

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_YESNO_RE = re.compile(r"^\W*(yes|no)\b", re.I)
# [x, y] (point) or [x1, y1, x2, y2] (box -> its centre); 3-value groups are ignored
_GROUP_RE = re.compile(r"\[\s*(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){1,3})\s*\]")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")  # one level of brace nesting
_VERDICT_RE = re.compile(r"\b(YES|NO)\b", re.I)
JUDGE_MODEL = "Qwen/Qwen3.5-9B"  # the served model name of scripts/serve_judge.sh


def grounding_reward(resp: str, boxes: list) -> float:
    """F1 between the response's points (clipped to 0..1000, de-duplicated) and the GT boxes, a box being matched
    when some point lies in it: P = matched / #points, R = matched / #boxes. No points or no boxes -> 0."""
    pts = []
    for g in _GROUP_RE.findall(resp):
        v = [float(x) for x in g.split(",")]
        if len(v) == 2:
            pts.append((v[0], v[1]))
        elif len(v) == 4:
            pts.append(((v[0] + v[2]) / 2.0, (v[1] + v[3]) / 2.0))
    pts = list(dict.fromkeys((min(1000.0, max(0.0, x)), min(1000.0, max(0.0, y))) for x, y in pts))
    if not pts or not boxes:
        return 0.0
    hit = lambda p, b: 1.0 if b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3] else 0.0
    matched = sum(max(hit(p, b) for p in pts) for b in boxes)
    recall, prec = matched / len(boxes), matched / len(pts)
    return 2.0 * prec * recall / (prec + recall + 1e-8)


def judge(question: str, gold: str, pred: str) -> float | None:
    """YES/NO verdict of the judge at $JUDGE_URL (OpenAI-compatible) -> 1.0 / 0.0; None after 3 tries."""
    prompt = (
        "You are grading a vision-language model's answer to a question about an "
        "image. You cannot see the image; judge ONLY whether the model's answer "
        "means the same thing as the reference answer.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {gold}\n"
        f"Model answer: {pred}\n\n"
        "Does the model's answer convey the same answer as the reference? "
        "Ignore differences in wording, phrasing, verbosity and punctuation. "
        "Reply with exactly one word: YES or NO."
    )
    body = json.dumps(
        {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"choice": ["YES", "NO"]},
        }
    ).encode()
    for attempt in range(3):
        try:
            url = os.environ["JUDGE_URL"].strip().rstrip("/") + "/chat/completions"
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30.0) as r:
                msg = json.loads(r.read())["choices"][0]["message"]
            txt = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
            # the committed answer (last <answer>, else last \boxed{}, else text after </think>), then any YES/NO
            answers, boxed = _ANSWER_RE.findall(txt), _BOXED_RE.findall(txt)
            if answers:
                span = answers[-1].strip()
            elif boxed:
                span = boxed[-1].strip()
            else:
                span = txt.split("</think>")[-1].strip() if "</think>" in txt else txt.strip()
            verdicts = _VERDICT_RE.findall(span)
            if not verdicts and span != txt:
                verdicts = _VERDICT_RE.findall(txt)
            if verdicts:
                return 1.0 if verdicts[-1].upper() == "YES" else 0.0
        except Exception:  # network flake or malformed reply: retry
            pass
        time.sleep(0.5 * (attempt + 1))
    return None


def accuracy_score(resp: str, gt: dict) -> float:
    boxed = extract_last_boxed(resp)
    if boxed is None:
        return 0.0
    if gt["kind"] == "mcq":
        opts = gt.get("options") or []
        pred = parse_mcq_index(boxed, opts, len(opts) or 26)
        return 1.0 if pred is not None and pred == gt["gold_index"] else 0.0
    if gt["kind"] == "yesno":
        pm, gm = _YESNO_RE.match(boxed.strip()), _YESNO_RE.match(gt["answer"].strip())
        if pm and gm:
            return 1.0 if pm.group(1).lower() == gm.group(1).lower() else 0.0
    s = judge(gt["prompt"], gt["answer"], boxed)
    return 0.0 if s is None else float(s)


def format_score(resp: str) -> float:
    """1 for exactly one non-empty <think>...</think> followed by a \\boxed{}."""
    thinks = _THINK_RE.findall(resp)
    if len(thinks) != 1 or not thinks[0].strip() or resp.count("<think>") != 1 or resp.count("</think>") != 1:
        return 0.0
    return 1.0 if extract_last_boxed(resp.split("</think>", 1)[1]) is not None else 0.0


def compute_score(solution_str, ground_truth, success_weight=1.0, format_weight=1.0, grounding_weight=1.0, **_):
    """One rollout -> {"score", "accuracy", "format", "grounding"} (verl passes every argument by keyword)."""
    resp, gt = str(solution_str), json.loads(ground_truth)
    acc, fmt = accuracy_score(resp, gt), format_score(resp)
    gnd = grounding_reward(resp, [tuple(b) for b in gt.get("boxes", [])])
    score = float(success_weight) * acc + float(format_weight) * fmt + float(grounding_weight) * gnd
    return {"score": float(score), "accuracy": float(acc), "format": float(fmt), "grounding": float(gnd)}


# ---- prompt set
# treevgr  $GR_DATA/rl/treevgr/ (HF HaochenWang/TreeVGR-RL-37K: parquet + extracted images/) -> rl/treevgr_rl.parquet
# band     $GR_RESULTS/passk/<init>/<subset>.jsonl + the shipped rl/band/heldout.jsonl -> rl/band/band.jsonl
# verl     rl/treevgr_rl.parquet + rl/band/{band,heldout}.jsonl -> rl/train/{train,val}.parquet (absolute image paths)

RL_DIR = DATA / "rl"
TREEVGR_SOURCE = RL_DIR / "treevgr" / "vstar30k_visdrone6k_x1y1x2y2.parquet"
TREEVGR = RL_DIR / "treevgr_rl.parquet"
BAND_DIR, TRAIN_DIR = RL_DIR / "band", RL_DIR / "train"
INITS, K, LO, HI = ("text_unfiltered", "grounded_unfiltered"), 8, 0.2, 0.8  # band: pass@K rate in [LO, HI]
MIN_PIXELS, MAX_PIXELS = 65536, 16777216  # Qwen3-VL processor pixel range used for training
# TVG subset -> (tag in the data, SFT `source` tag, row_id offset)
SUBSETS = {"spatialmqa": ("smqa", "smqa_2s2", 0), "multihopspatial": ("mhs", "mhs_2s2", 100000)}

VSTAR_TEST_TAIL = 36561  # source rows from here on duplicate V* Bench test images and are left out
OPTION_RE = re.compile(r"^\(?([A-H])[\).]\s*(.+?)\s*$", re.M)  # "(A) left" / "A. left" / "A) left"
MULTI_QA_RE = re.compile(r"\n\s*(==\s*\n)?\s*Question\s*:", re.I)  # several QA pairs concatenated in one answer


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def answer_kind(answer: str) -> str:
    a = answer.strip()
    if len(a) <= 2 and a.strip(".) ").upper() in "ABCDEFGH" and a.strip(".) "):
        return "mcq"
    if re.match(r"^(yes|no)\b", a, re.I):
        return "yesno"
    return "open"


def treevgr_record(row: dict) -> dict | None:
    """TreeVGR row -> {problem, answer JSON, images}, boxes clipped to the image and rescaled to 0..1000; None for a
    missing image, a multi-QA answer or no usable box."""
    image = f"rl/treevgr/{row['images'][0]}"
    if not (DATA / image).exists() or MULTI_QA_RE.search(row["answer"]) or not row["target_instances"]:
        return None
    try:
        with Image.open(DATA / image) as im:
            width, height = im.size
    except Exception:
        return None
    boxes = []
    for inst in row["target_instances"]:
        x1, y1, x2, y2 = inst["bbox"]
        if x2 <= x1 or y2 <= y1:
            continue
        x1, x2 = (max(0.0, min(float(width), float(v))) for v in (x1, x2))
        y1, y2 = (max(0.0, min(float(height), float(v))) for v in (y1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        scaled = (x1 / width * 1000, y1 / height * 1000, x2 / width * 1000, y2 / height * 1000)
        boxes.append([min(1000.0, max(0.0, v)) for v in scaled])
    if not boxes:
        return None
    question = re.sub(r"^<image>\s*", "", row["problem"]).strip()
    kind = answer_kind(row["answer"])
    options = []
    if kind == "mcq":  # option texts in letter order
        found = {m.group(1).upper(): m.group(2) for m in OPTION_RE.finditer(question)}
        options = [found[c] for c in "ABCDEFGH" if c in found]
        if not options:  # a letter answer without parseable options is judged as open
            kind = "open"
    answer = {"answer": row["answer"], "kind": kind, "options": options, "boxes": boxes, "prompt": question}
    answer["coordinate_space"] = "norm1000"
    return {"problem": question, "answer": json.dumps(answer), "images": [image]}


def build_treevgr():
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(TREEVGR_SOURCE).to_pylist()[:VSTAR_TEST_TAIL]
    records = [r for r in map(treevgr_record, rows) if r is not None]
    pq.write_table(pa.Table.from_pylist(records), TREEVGR)
    print(f"treevgr: {len(records)} of {len(rows)} rows -> {TREEVGR}")


def build_band():
    """Train questions whose pass@8 rate is in [LO, HI] under BOTH unfiltered SFT inits, minus the held-out rows and
    every question sharing their images."""
    rows, counts = {}, defaultdict(dict)
    for init in INITS:
        for subset in SUBSETS:
            for r in jsonl(RESULTS / "passk" / init / f"{subset}.jsonl"):
                rows.setdefault((r["source"], r["idx"]), r)
                counts[(r["source"], r["idx"])][init] = r["n_hit"]
    heldout = jsonl(BAND_DIR / "heldout.jsonl")
    held_keys = {(r["source"], r["idx"]) for r in heldout}
    held_images = {r["images"][0] for r in heldout}
    kept = [
        key
        for key in rows
        if set(counts[key]) == set(INITS)
        and key not in held_keys
        and rows[key]["image"] not in held_images
        and all(LO - 1e-9 <= counts[key][t] / K <= HI + 1e-9 for t in INITS)
    ]
    with open(BAND_DIR / "band.jsonl", "w") as f:
        for key in sorted(kept, key=lambda kk: rows[kk]["row_id"]):
            r = rows[key]
            answer = {"letter": r["gold_letter"], "index": r["gold_index"], "options": r["options"]}
            answer["n_options"] = r["n_options"]
            rec = {"problem": r["prompt"], "answer": json.dumps(answer), "images": [r["image"]]}
            for field in ("gold_letter", "gold_index", "n_options", "options", "source", "idx", "row_id"):
                rec[field] = r[field]
            rec.update(in_grounded_unfiltered=r["in_grounded_unfiltered"], in_grounded=r["in_grounded"], pass_n=K)
            rec.update({f"pass_count_{t}": counts[key][t] for t in INITS})
            f.write(json.dumps(rec) + "\n")
    print(f"band: {len(kept)} rows -> {BAND_DIR / 'band.jsonl'}")


def treevgr_rows():
    """TreeVGR rows: `problem` gains the SFT suffix, mcq answers gain gold_index."""
    import pyarrow.parquet as pq

    out = []
    for i, r in enumerate(pq.read_table(TREEVGR, columns=["problem", "answer", "images"]).to_pylist()):
        a = json.loads(r["answer"])
        if a["kind"] == "mcq":
            a["gold_index"] = "ABCDEFGH".find(a["answer"].strip().upper())
        src = "visdrone" if a["kind"] == "mcq" else "vstar"
        out.append((r["problem"] + SUFFIX, json.dumps(a), r["images"], src, a["kind"], 1_000_000 + i))
    return out


def band_rows(records):
    """Band records -> prompt rows whose answer JSON carries the GT boxes of the question's objects."""
    boxes = {}
    for name, (subset, _, _) in SUBSETS.items():
        for m in jsonl(DATA / "tvg" / name / "meta.jsonl"):
            boxes[(subset, m["idx"])] = [b for o in m["objects"] for b in o["boxes"]]
    out = []
    for r in records:
        a = {
            "answer": r["gold_letter"].upper(),
            "kind": "mcq",
            "options": r["options"],
            "gold_index": r["gold_index"],
            "boxes": boxes[(r["source"], r["idx"])],
            "prompt": r["problem"][: -len(SUFFIX)],
            "coordinate_space": "norm1000",
        }
        out.append((r["problem"], json.dumps(a), r["images"], r["source"], "mcq", r["row_id"]))
    return out


def write_verl(path, rows, split):
    """verl RLHF rows: one `<image>` placeholder + problem as the user turn (= the SFT human turn)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = [
        {
            "data_source": "grounded_reasoning",
            "prompt": [{"role": "user", "content": "<image>\n" + problem}],
            "images": [
                {
                    "image": "file://" + os.path.abspath(DATA / images[0]),
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                }
            ],
            "ability": kind,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, "source": source, "kind": kind, "row_id": int(row_id)},
        }
        for problem, answer, images, source, kind, row_id in rows
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(table), path)


def build_verl():
    """verl val = the first 64 held-out rows (verl requires one; validation is never run)."""
    train = treevgr_rows() + band_rows(jsonl(BAND_DIR / "band.jsonl"))
    write_verl(str(TRAIN_DIR / "train.parquet"), train, "train")
    write_verl(str(TRAIN_DIR / "val.parquet"), band_rows(jsonl(BAND_DIR / "heldout.jsonl"))[:64], "val")
    print(f"verl: {len(train)} train rows -> {TRAIN_DIR}")


# ---- pass@8 of an SFT init on the TVG train questions: prompt = the SFT human turn, image as a base64 PNG; temperature
# 1, top-p 1, 8192 tokens, seed = 1000 + row_id; strict last-\boxed{} scoring. One JSON row per question.

SMQA_TAIL = "\nAnswer with the option letter."
_OPT = re.compile(r"^\(([a-z])\) (.*)$", re.M)
SEED_BASE, CHUNK = 1000, 256  # CHUNK = prompts per llm.chat call


def load_rows(subset: str) -> list[dict]:
    """TVG train questions of one subset -> prompt rows; rows whose '(a) ..' option lines do not run from 'a' are
    skipped."""
    source, tag, offset = SUBSETS[subset]
    sft = {
        name: {(r["source"], r["pair_idx"]) for r in json.loads((DATA / "sft" / f"{name}.json").read_text())}
        for name in ("grounded_unfiltered", "grounded")
    }
    rows = []
    with open(DATA / "tvg" / subset / "meta.jsonl") as f:
        metas = [json.loads(line) for line in f]
    for m in metas:
        found = _OPT.findall(m["question"])
        letters = [letter for letter, _ in found]
        gold_index = LETTERS.find(m["answer"])
        if not letters or letters != list(LETTERS[: len(letters)]) or not 0 <= gold_index < len(found):
            continue
        q = m["question"]
        rows.append(
            {
                "source": source,
                "idx": m["idx"],
                "row_id": offset + m["idx"],
                "image": m["image"],
                "prompt": (q[: -len(SMQA_TAIL)] if q.endswith(SMQA_TAIL) else q) + SUFFIX,
                "question": q,
                "options": [t.strip() for _, t in found],
                "n_options": len(found),
                "gold_letter": m["answer"],
                "gold_index": gold_index,
                "in_grounded_unfiltered": (tag, m["idx"]) in sft["grounded_unfiltered"],
                "in_grounded": (tag, m["idx"]) in sft["grounded"],
            }
        )
    return rows


def image_data_url(path: str) -> str:
    buf = io.BytesIO()
    Image.open(DATA / path).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def passk(model: str, subset: str, out: str) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        seed=1,
        max_model_len=32768,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        disable_log_stats=True,
    )
    rows = load_rows(subset)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        for c0 in range(0, len(rows), CHUNK):
            chunk = rows[c0 : c0 + CHUNK]
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url(r["image"])}},
                            {"type": "text", "text": r["prompt"]},
                        ],
                    }
                ]
                for r in chunk
            ]
            params = [
                SamplingParams(n=K, temperature=1.0, top_p=1.0, max_tokens=8192, seed=SEED_BASE + r["row_id"])
                for r in chunk
            ]
            for r, o in zip(chunk, llm.chat(messages=messages, sampling_params=params, use_tqdm=False)):
                samples = []
                for s in o.outputs:
                    pred = parse_mcq_index(extract_last_boxed(s.text), r["options"], r["n_options"])
                    samples.append({"text": s.text, "finish_reason": s.finish_reason, "hit": pred == r["gold_index"]})
                n_hit = sum(s["hit"] for s in samples)
                f.write(
                    json.dumps({**r, "seed": SEED_BASE + r["row_id"], "n": K, "n_hit": n_hit, "samples": samples})
                    + "\n"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    bd = sub.add_parser("build-data", help="one step of the RL prompt set")
    bd.add_argument("step", choices=["treevgr", "band", "verl"])
    pk = sub.add_parser("passk", help="pass@8 of an SFT init -> one JSON row per question")
    pk.add_argument("model", help="SFT run dir, e.g. $GR_RUNS/sft/text_unfiltered_s17")
    pk.add_argument("subset", choices=list(SUBSETS))
    pk.add_argument("out", help="output .jsonl, e.g. $GR_RESULTS/passk/<init>/<subset>.jsonl")
    args = ap.parse_args()
    if args.cmd == "build-data":
        {"treevgr": build_treevgr, "band": build_band, "verl": build_verl}[args.step]()
    else:
        passk(args.model, args.subset, args.out)


if __name__ == "__main__":  # vLLM spawns workers that re-import this module
    main()
