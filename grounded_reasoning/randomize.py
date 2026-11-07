"""Decode-time coordinate randomization (Table 1; Figure 3, right): each arm decodes four benchmarks (grounded_rand*:
MultihopSpatial only) under clean, replay (each completed coordinate kept) and randomize (redrawn), 20 seeds each;
grounding usage = mean randomize accuracy - mean replay accuracy. Output: $GR_RESULTS/decode_randomization/.
    python -m grounded_reasoning.randomize prepare        # download + cache the four benchmarks (once, before run)
    python -m grounded_reasoning.randomize run --arm grounded --bench multihopspatial --seed 1000    # one GPU
    python -m grounded_reasoning.randomize report         # rollouts -> report/{report.json,table_decode_randomization.tex}
"""

import argparse
import glob
import hashlib
import io
import json
import math
import os
import random
import re
import statistics as st
import time
from pathlib import Path

from PIL import Image, ImageFile

from grounded_reasoning import DATA, RESULTS, RUNS
from grounded_reasoning import randomize_families as families
from grounded_reasoning.benchmarks import ITEMS
from grounded_reasoning.randomize_families import DE_TOOL_RE

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a truncated image file loads instead of raising

ROLLOUTS = RESULTS / "decode_randomization" / "rollouts"
# arm -> (prompt family, checkpoint, Hub revision)
ARMS = {
    "treevgr_ci": ("treevgr", "HaochenWang/TreeVGR-7B-CI", "d87519234153c5357d9ba39e2d0d1400b2bd3a02"),
    "treevgr_rl": ("treevgr", "HaochenWang/TreeVGR-7B", "5c8b749d6ff99e916c16ad6543f36c5342ad9214"),
    "ivgr": ("ivgr", "allencbzhang/iVGR-Qwen3-VL-8B", "32191f45eca2e8bcbd7c4cabd7e911cc120d7545"),
    "deepeyes": ("deepeyes", "ChenShawn/DeepEyes-7B", "3b31e4b342592d14cb12bb8434a6ea55e9cd78c8"),
    **{
        arm: ("ours", str(RUNS / "sft" / f"{arm}_s17"), None)
        for arm in ["appended", "grounded", "grounded_rand05", "grounded_rand20", "grounded_rand50", "grounded_rand100"]
    },
}

# Canonical items: idx, image_path, question, and options + gold_index[, hop] or count. MultihopSpatial: a seed-17
# random 1,500 of the 4,498 parseable test questions; SpatialMQA: test; CountQA: every test question (images cached
# in $GR_DATA/benchmarks/countqa_decode); V*: the evaluation's items.

BENCH_SIZES = {"multihopspatial": 1500, "spatialmqa": 1076, "vstar": 191, "countqa": 1528}
REVISIONS = {
    "etri-vilab/MultihopSpatial": "6c962f5227acc133d2196cbf89f09441c7ccb703",
    "liuziyan/SpatialMQA": "2c297135743209b91fe0729c033b49bb1d72f788",
    "Jayant-Sravan/CountQA": "f92cc6fe46542c61e2916e3d2ae9a911e2216b1a",
}
VSTAR_ITEMS = ITEMS / "vstar.jsonl"  # image paths relative to $GR_DATA


def hub_dataset(repo):
    """Hub dataset id -> local snapshot directory (downloads unless HF_HUB_OFFLINE=1)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo, repo_type="dataset", revision=REVISIONS[repo])


# Two rows have a malformed block: "<choice>... /choice>" (3056) and an untagged "(a) ... ." list (4444).
_CHOICE_RE = re.compile(r"<choice>(.*?)</choice>", re.S)
_OPTION_RE = re.compile(r"\(([a-h])\)\s*(.*?)(?=\s*,?\s*\([a-h]\)|$)", re.S)


def load_multihopspatial(n=1500, seed=17):
    """Questions embed "<choice>(a) X, (b) Y</choice>"; gold "(b)" or "b"."""
    snap = hub_dataset("etri-vilab/MultihopSpatial")
    with open(f"{snap}/data/multihop_test_4500.json") as stream:
        rows = json.load(stream)
    items = []
    for i, r in enumerate(rows):
        m = _CHOICE_RE.search(r["question"])
        pairs = _OPTION_RE.findall(m.group(1)) if m else []
        letters = [letter for letter, _ in pairs]
        options = [text.strip().rstrip(",").strip() for _, text in pairs]
        gm = re.match(r"\s*\(?([a-h])\)?", r["answer"].strip())
        gold = gm.group(1) if gm else None
        path = f"{snap}/data/images/{r['image_path']}"
        if len(options) < 2 or gold not in letters or not os.path.exists(path):
            continue
        # gold_index is the letter's position in the alphabet, not in `letters`.
        items.append(
            {
                "idx": i,
                "image_path": path,
                "question": r["question"],
                "options": options,
                "gold_index": "abcdefgh".find(gold),
                "hop": r["hop"],
            }
        )
    random.Random(seed).shuffle(items)
    return sorted(items[:n], key=lambda x: x["idx"])


def load_spatialmqa():
    snap = hub_dataset("liuziyan/SpatialMQA")
    items = []
    with open(f"{snap}/test.jsonl") as stream:
        rows = [json.loads(line) for line in stream]
    for i, r in enumerate(rows):
        path = f"{snap}/images/{r['image']}"
        if os.path.exists(path):
            items.append(
                {
                    "idx": i,
                    "image_path": path,
                    "question": r["question"],
                    "options": r["options"],
                    "gold_index": r["options"].index(r["answer"]),
                }
            )
    return items


def load_vstar():
    with open(VSTAR_ITEMS) as stream:
        rows = [json.loads(line) for line in stream]
    items = []
    for r in rows:
        path = DATA / r["images"][0]
        if path.exists() and path.stat().st_size > 0:
            items.append(
                {
                    "idx": r["idx"],
                    "image_path": str(path),
                    "question": r["question"],
                    "options": r["options"],
                    "gold_index": r["gold_index"],
                }
            )
    return items


def load_countqa():
    """Every question (all answers are integers); idx = question position. Parquet shards are read in sorted order
    (what `datasets.load_dataset` does); the image cache is keyed by the dataset row index. O(rows)."""
    import pyarrow.parquet as pq

    snap = hub_dataset("Jayant-Sravan/CountQA")
    cache = DATA / "benchmarks" / "countqa_decode"
    cache.mkdir(parents=True, exist_ok=True)
    items, row_index = [], 0
    for shard in sorted(glob.glob(f"{snap}/data/test-*.parquet")):
        for r in pq.read_table(shard).to_pylist():
            path = cache / f"{row_index:05d}.png"
            if not path.exists() or path.stat().st_size < 256:
                Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(path)
            for question, answer in zip(r["questions"], r["answers"]):
                count = int(str(answer).strip())
                items.append({"idx": len(items), "image_path": str(path), "question": question, "count": count})
            row_index += 1
    return items


def load_bench(bench):
    """Bench name -> canonical items (the paper's item count)."""
    loaders = {
        "multihopspatial": load_multihopspatial,
        "spatialmqa": load_spatialmqa,
        "vstar": load_vstar,
        "countqa": load_countqa,
    }
    items = loaders[bench]()
    assert len(items) == BENCH_SIZES[bench], (bench, len(items))
    return items


# Decode loop (vLLM): generation stops after every completed coordinate and restarts from the (replayed or randomized)
# text as a new request. Request i of a response uses seed request_seed(seed, idx, i); the randomizer uses
# random.Random(1000 + idx) for every seed.

EVAL_TAG = "corrv5-cap100-vllm"  # written to every output; "corrv5" also salts the sampling seeds below
CAP = 100  # interventions per response: a replay/randomize response stops there and is wrong
MAX_NEW_TOKENS = 8192
TEMPERATURE = 1.0


def request_seed(seed_base, item_idx, request_index):
    """-> 63-bit sampling seed; request 0 uses seed_base + item_idx, like a plain eval."""
    if request_index == 0:
        return seed_base + item_idx
    key = f"corrv5:{seed_base}:{item_idx}:{request_index}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") & ((1 << 63) - 1)


def whole_stop_suffix(tokenizer, output):
    """vLLM cuts output.text at the stop string, but the last sampled token may continue past it (e.g. "]}"
    completing a JSON box). -> the cut-off remainder of that last token."""
    if not isinstance(output.stop_reason, str):
        return ""
    full = tokenizer.decode(output.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return full[len(output.text) :]


def completion_hit(answer_hit, finish):
    """A correct answer counts only if the response ended with EOS."""
    return bool(answer_hit and finish == "eos")


_COORD_LIST = r"\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){1,3}\s*\]"
_BBOX_LIST = r'"bbox_2d"\s*:\s*(' + _COORD_LIST + ")"
_NUMBER = r"-?\d+(?:\.\d+)?"


def replace_coord_values(original, drawn):
    """-> `original` with only the numbers of its first bracket list (the one after "bbox_2d" if present, since
    tool labels may contain bracketed numbers) replaced by `drawn`'s."""
    old_box, new_box = re.search(_BBOX_LIST, original), re.search(_BBOX_LIST, drawn)
    offset = old_box.start(1) if old_box else 0
    old = re.search(_COORD_LIST, old_box.group(1) if old_box else original)
    new = re.search(_COORD_LIST, new_box.group(1) if new_box else drawn)
    if old is None or new is None:
        raise ValueError("coordinate replacement lacks numeric bracket group")
    old_numbers = list(re.finditer(_NUMBER, old.group(0)))
    new_numbers = re.findall(_NUMBER, new.group(0))
    if len(old_numbers) != len(new_numbers):
        raise ValueError("coordinate arity changed")
    result = original
    for match, value in reversed(list(zip(old_numbers, new_numbers))):
        start, end = offset + old.start() + match.start(), offset + old.start() + match.end()
        result = result[:start] + value + result[end:]
    return result


def cell_paths(outdir, bench, arm, mode, seed):
    stem = os.path.join(outdir, f"{bench}_{arm}_{mode}_s{seed}")
    return stem + ".json", stem + "_rollouts.jsonl"


def decode(arm, family, model_id, bench, items, cells, outdir, revision=None):
    """Decode every item under every (mode, seed) cell -> per-cell summary JSON + rollouts JSONL.

    items: family items (idx, image_path, user_text, options/gold_index or count); revision: Hub commit of model_id
    (None for a local directory). All states advance together in batches; each round rescans the response for
    coordinates, O(response length^2).
    """
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    maxnew = MAX_NEW_TOKENS
    deepeyes = family == "deepeyes"
    max_model_len = 32768 if deepeyes else 16384
    coord_re = families.COORD_RES[family]
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    tok = processor.tokenizer
    llm = LLM(
        model=model_id,
        revision=revision,
        dtype="bfloat16",
        gpu_memory_utilization=0.80 if deepeyes else 0.90,
        max_model_len=max_model_len,
        seed=cells[0][1],
        # Only token blocks before an edit are reused, so a restart equals a fresh prefill.
        enable_prefix_caching=True,
        # DeepEyes needs one image slot per possible tool call (each costs >= 1 output token).
        limit_mm_per_prompt={"image": maxnew + 1 if deepeyes else 1},
        **({"skip_mm_profiling": True, "max_num_seqs": 32} if deepeyes else {}),
    )

    def visual_tokens(img):
        """PIL image -> (visual token count, resized width, resized height)."""
        grid = processor.image_processor(images=[img])["image_grid_thw"][0]
        merge, patch = processor.image_processor.merge_size, processor.image_processor.patch_size
        return int(grid[0] * grid[1] * grid[2]) // (merge * merge), int(grid[2]) * patch, int(grid[1]) * patch

    assets = {}
    for it in items:
        img = Image.open(it["image_path"]).convert("RGB")
        messages, after_prompt = families.chat_messages(family, it["user_text"])
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + after_prompt
        assets[it["idx"]] = (img, prefix, *visual_tokens(img))

    states = []
    for mode, seed in cells:
        for it in items:
            img, prefix, count, width, height = assets[it["idx"]]
            states.append(
                {
                    "it": it,
                    "img": img,
                    "images": [img],
                    "prefix": prefix,
                    "visual_tokens": [count],
                    "gen_text": "",
                    "interventions": [],
                    "finish": None,
                    "mode": mode,
                    "seed": seed,
                    "n_generated_tokens": 0,
                    "sampling_seeds": [],
                    "rng": random.Random(1000 + it["idx"]),
                    "resW": width,
                    "resH": height,
                }
            )

    started, rounds = time.time(), 0
    while any(s["finish"] is None for s in states):
        rounds += 1
        batch, requests, params = [], [], []
        for s in states:
            if s["finish"] is not None:
                continue
            remaining = maxnew - s["n_generated_tokens"]
            if remaining <= 0:
                s["finish"] = "budget"
                continue
            prompt = s["prefix"] + s["gen_text"]
            # Qwen expands each image-pad placeholder into visual_tokens entries.
            prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"]) + sum(
                n - 1 for n in s["visual_tokens"]
            )
            available = max_model_len - prompt_len
            if available <= 0:
                s["finish"] = "context_limit"
                continue
            s["context_limited_request"] = available < remaining
            seed = request_seed(s["seed"], s["it"]["idx"], len(s["sampling_seeds"]))
            s["sampling_seeds"].append(seed)
            batch.append(s)
            requests.append({"prompt": prompt, "multi_modal_data": {"image": s["images"] if deepeyes else s["img"]}})
            params.append(
                SamplingParams(
                    temperature=TEMPERATURE,
                    top_p=1.0,
                    max_tokens=min(remaining, available),
                    seed=seed,
                    stop=families.STOPS[family] if s["mode"] != "clean" or deepeyes else None,
                    include_stop_str_in_output=True,
                )
            )
        if not batch:
            break
        outputs = llm.generate(requests, params, use_tqdm=False)
        for s, output in zip(batch, outputs):
            out = output.outputs[0]
            sampled = len(out.token_ids)
            s["n_generated_tokens"] += sampled
            prior_groups = len(coord_re.findall(s["gen_text"]))
            chunk_start = len(s["gen_text"])
            # Restore the stop token's tail before matching: it may complete a JSON box.
            s["gen_text"] += out.text + whole_stop_suffix(tok, out)
            if out.finish_reason == "length":
                s["finish"] = "context_limit" if s["context_limited_request"] else "length"
                continue
            if out.finish_reason == "stop" and not isinstance(out.stop_reason, str):
                s["finish"] = "eos"
                continue
            if sampled == 0:
                raise RuntimeError(f"no-progress nonterminal request idx={s['it']['idx']}")
            if deepeyes and "</answer>" in s["gen_text"].rsplit("<|im_start|>assistant\n", 1)[-1]:
                s["finish"], s["answer_stop"] = "eos", True
                continue
            groups = list(coord_re.finditer(s["gen_text"]))
            if (s["mode"] != "clean" or deepeyes) and len(groups) > prior_groups:
                match = groups[prior_groups]
                replacement = match.group(0)
                if s["mode"] == "randomize":
                    drawn = families.random_coordinate(family, match, s["rng"], s["resW"], s["resH"])
                    replacement = replace_coord_values(match.group(0), drawn)
                    # Keep only the rest of the SAME sampled token after the coordinate; later tokens were
                    # conditioned on the old value and are discarded.
                    before_last = tok.decode(
                        out.token_ids[:-1], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    tail = s["gen_text"][match.end() :] if match.end() > chunk_start + len(before_last) else ""
                    s["gen_text"] = s["gen_text"][: match.start()] + replacement + tail
                    s["interventions"].append([match.group(0), replacement])
                else:
                    # Replay keeps the text (overshoot included) and records every coordinate of this request.
                    completed_groups = groups[prior_groups:]
                    if s["mode"] != "clean":
                        completed_groups = completed_groups[: CAP - len(s["interventions"])]
                    for completed in completed_groups:
                        s["interventions"].append([completed.group(0), completed.group(0)])
                if s["mode"] != "clean" and len(s["interventions"]) >= CAP:
                    s["finish"] = "intervention_limit"
                    continue
                if deepeyes and s["n_generated_tokens"] < maxnew:
                    tool_match = families.DE_TOOL_RE.search(replacement)
                    box = [int(float(tool_match.group(k))) for k in range(1, 5)]
                    count_before = len(s["images"])
                    s["gen_text"] += families.de_tool_turn(s, box)
                    if len(s["images"]) > count_before:
                        s["visual_tokens"].append(visual_tokens(s["images"][-1])[0])
            elif deepeyes and isinstance(out.stop_reason, str):
                s["gen_text"] += families.DE_ERR.format("Invalid tool call format: no bbox_2d found")
        if rounds % 10 == 0:
            done = sum(s["finish"] is not None for s in states)
            print(f"[round {rounds}] done={done}/{len(states)} elapsed={time.time() - started:.0f}s", flush=True)

    os.makedirs(outdir, exist_ok=True)
    for mode, seed in cells:
        rows = []
        for s in (s for s in states if s["mode"] == mode and s["seed"] == seed):
            it = s["it"]
            pred, answer_hit = families.score_item(family, bench, it, s["gen_text"])
            row = {
                "idx": it["idx"],
                "bench": bench,
                "mode": mode,
                "eval_tag": EVAL_TAG,
                "arm": arm,
                "prompt_mode": family,
                "sample_seed": seed,
                "answer": it["count"] if bench == "countqa" else it["options"][it["gold_index"]],
                "gold_index": it.get("gold_index"),
                "options": it.get("options"),
                "pred": pred,
                "answer_hit": bool(answer_hit),
                "hit": completion_hit(answer_hit, s["finish"]),
                "n_interventions": len(s["interventions"]),
                "interventions": s["interventions"],
                "finish": s["finish"],
                "resp": s["gen_text"],
                "n_generated_tokens": s["n_generated_tokens"],
                "sampling_seeds": s["sampling_seeds"],
            }
            if family in ("treevgr", "deepeyes"):
                row["resized_wh"] = [s["resW"], s["resH"]]
            if deepeyes:
                row.update(n_images=len(s["images"]), answer_stop=s.get("answer_stop", False))
            rows.append(row)
        summary_path, rollout_path = cell_paths(outdir, bench, arm, mode, seed)
        with open(rollout_path, "w") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)
        summary = {
            "eval_tag": EVAL_TAG,
            "arm": arm,
            "model": model_id,
            "revision": revision,
            "bench": bench,
            "mode": mode,
            "temperature": TEMPERATURE,
            "seed_base": seed,
            "max_interventions": CAP,
            "max_new_tokens": maxnew,
            "max_model_len": max_model_len,
            "n": len(rows),
            "acc": sum(r["hit"] for r in rows) / len(rows),
        }
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[{mode}:s{seed}] acc={summary['acc']:.4f}", flush=True)


# Scoring: a response is correct only if it ended with EOS and its answer is right; a replay/randomize response with
# >= CAP interventions is wrong. Rows are checked strictly: finish and correctness must agree with the rule, replay
# events must not change bytes, randomize events only numbers.

# eval_tag of the rollouts decoded without the stop at CAP (seeds 1000-5000): the cap is applied when scoring them.
UNCAPPED_TAG = "corrv5-vllm"
COORD = re.compile(r"\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){1,3})\s*\]")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def coordinate_match(text):
    """Intervention event text -> match of its numeric coordinate list, or None.

    DeepEyes tool calls use the box the decode loop's DE_TOOL_RE captured (a call may start with a malformed
    "bbox_2d" before the valid one). Other dialects: the list after "bbox_2d", else the first bracket list.
    """
    if text.startswith("<tool_call>"):
        tool = DE_TOOL_RE.fullmatch(text)
        return tool and COORD.match(text, text.rfind("[", tool.start(), tool.start(1) + 1))
    bbox = re.search(r'"bbox_2d"\s*:\s*', text)
    return COORD.match(text, bbox.end()) if bbox else COORD.search(text)


def values(text):
    match = coordinate_match(text)
    return tuple(float(x) for x in match.group(1).split(",")) if match else ()


def skeleton(text):
    """Event text with the coordinate numbers masked (every other byte kept)."""
    match = coordinate_match(text)
    if match is None:
        return text
    return text[: match.start()] + re.sub(r"-?\d+(?:\.\d+)?", "#", match.group(0)) + text[match.end() :]


def check_row(row, mode, capped, context):
    """Raise ValueError unless the row's finish, correctness and intervention events are consistent."""
    count, events = row["n_interventions"], row["interventions"]
    require(len(events) == count, f"{context}: intervention count")
    require(mode != "clean" or row["prompt_mode"] == "deepeyes" or count == 0, f"{context}: clean interventions")
    require(not capped or mode == "clean" or count <= CAP, f"{context}: more than {CAP} interventions")
    limit = capped and mode != "clean" and count >= CAP
    require((row["finish"] == "intervention_limit") == limit, f"{context}: cap finish mismatch")
    require(row["hit"] == (row["answer_hit"] and row["finish"] == "eos"), f"{context}: completion scoring")
    for before, after in events:
        require(len(values(before)) in (2, 4) and len(values(before)) == len(values(after)), f"{context}: coordinates")
        require(mode != "replay" or before == after, f"{context}: replay byte edit")
        require(mode != "randomize" or skeleton(before) == skeleton(after), f"{context}: non-numeric edit")


def load_cell(directory, bench, arm, mode, seed):
    """{bench}_{arm}_{mode}_s{seed}{.json,_rollouts.jsonl} -> {idx: correct (bool)} with the cap applied. Each row is
    validated under its own eval_tag (a cell can join rows decoded with and without the intervention cap)."""
    stem = Path(directory) / f"{bench}_{arm}_{mode}_s{seed}"
    header_tag = json.loads(stem.with_suffix(".json").read_text())["eval_tag"]
    hits = {}
    with open(f"{stem}_rollouts.jsonl") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            check_row(row, mode, row.get("eval_tag", header_tag) != UNCAPPED_TAG, f"{stem}:{line_number}")
            require(row["idx"] not in hits, f"{stem}: duplicate item {row['idx']}")
            hits[row["idx"]] = row["hit"] and not (mode != "clean" and row["n_interventions"] >= CAP)
    require(len(hits) == BENCH_SIZES[bench], f"{stem}: wrong item count")
    return hits


# Report: per (arm, bench), mean(randomize acc) - mean(replay acc) over the 20 decode seeds, SE = sqrt(s_R^2/20 +
# s_Z^2/20). report.json[results][arm][bench] = {n, accuracy: {mode: [20 per-seed %]}, effect: {mean, se}}.

SEEDS = list(range(1000, 20001, 1000))
MODES = ["clean", "replay", "randomize"]
BENCHES = ["multihopspatial", "spatialmqa", "vstar", "countqa"]
TABLE_ROWS = [
    ("treevgr_ci", "TreeVGR-7B-CI"),
    ("treevgr_rl", "TreeVGR-7B-RL"),
    ("ivgr", "iVGR-8B"),
    ("deepeyes", "DeepEyes-7B"),
    ("appended", "Appended"),
    ("grounded", r"\textbf{Reflection}"),
]
# Training-time coordinate randomization (5/20/50/100 % of objects; 0 % is `grounded`), MultihopSpatial only.
LADDER_ARMS = ["grounded_rand05", "grounded_rand20", "grounded_rand50", "grounded_rand100"]
CELLS = [(arm, bench) for arm, _ in TABLE_ROWS for bench in BENCHES] + [(arm, "multihopspatial") for arm in LADDER_ARMS]
CAPTION = (
    r"\textbf{Grounding usage experiments.} We report change in mean performance (\% acc) for our coordinate "
    r"corruption experiments. Many open models don't rely on their grounding for spatial reasoning and perception. Our "
    r"retrained models (below) are more reliant on their grounding outputs."
)


def score_cell(directory, arm, bench):
    """-> {n, accuracy: {mode: [per-seed accuracy %]}, effect: {mean, se}} for one (arm, bench)."""
    accuracy = {mode: [] for mode in MODES}
    for seed in SEEDS:
        cells = {mode: load_cell(directory, bench, arm, mode, seed) for mode in MODES}
        assert cells["clean"].keys() == cells["replay"].keys() == cells["randomize"].keys(), (arm, bench, seed)
        for mode, hits in cells.items():
            accuracy[mode].append(100 * sum(hits.values()) / len(hits))
    replay, randomize = accuracy["replay"], accuracy["randomize"]
    effect = {
        "mean": st.fmean(randomize) - st.fmean(replay),
        "se": math.sqrt(st.stdev(replay) ** 2 / 20 + st.stdev(randomize) ** 2 / 20),
    }
    return {"n": len(cells["replay"]), "accuracy": accuracy, "effect": effect}


def latex(results):
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & MultihopSpatial & SpatialMQA & V$^*$ & CountQA \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Open models}} \\",
    ]
    for index, (arm, label) in enumerate(TABLE_ROWS):
        if index == 4:
            lines += [r"\midrule", r"\multicolumn{5}{l}{\textit{Ours} (format type)} \\"]
        effects = [results[arm][bench]["effect"] for bench in BENCHES]
        lines.append(label + " & " + " & ".join(f"${e['mean']:+.2f} \\pm {e['se']:.2f}$" for e in effects) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + CAPTION + "}",
        r"\label{tab:decode-corr}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def report():
    """ROLLOUTS -> report/{report.json,table_decode_randomization.tex} next to it."""
    results = {}
    for arm, bench in CELLS:
        results.setdefault(arm, {})[bench] = score_cell(ROLLOUTS, arm, bench)
    outdir = RESULTS / "decode_randomization" / "report"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(json.dumps({"seeds": SEEDS, "results": results}, indent=2) + "\n")
    (outdir / "table_decode_randomization.tex").write_text(latex(results))
    print(f"{len(CELLS)} cells -> {outdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="download the four benchmarks and cache the CountQA images")
    run = sub.add_parser("run", help="decode one arm on one bench for one seed under clean, replay and randomize")
    run.add_argument("--arm", required=True, choices=list(ARMS))
    run.add_argument("--bench", required=True, choices=list(BENCH_SIZES))
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--checkpoint", help="model directory or Hub id instead of the arm's checkpoint")
    run.add_argument("--revision", help="Hub revision to load instead of the pinned one")
    sub.add_parser("report", help="20-seed aggregation of the rollouts -> report.json and the LaTeX table")
    args = parser.parse_args()
    if args.command == "prepare":
        for name in BENCH_SIZES:
            print(name, len(load_bench(name)), flush=True)
    elif args.command == "run":
        family, checkpoint, revision = ARMS[args.arm]
        if args.checkpoint:
            checkpoint, revision = args.checkpoint, None
        revision = args.revision or revision
        items = families.build_items(family, args.bench, load_bench(args.bench))
        cells = [(mode, args.seed) for mode in ("clean", "replay", "randomize")]
        print(f"arm={args.arm} model={checkpoint}@{revision} bench={args.bench} n={len(items)}", flush=True)
        decode(args.arm, family, checkpoint, args.bench, items, cells, str(ROLLOUTS), revision)
    else:
        report()


if __name__ == "__main__":
    main()
