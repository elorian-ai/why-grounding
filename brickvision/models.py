"""Run models on the restyled BrickVision -> $GR_RESULTS/brickvision/<row>/<task>.jsonl (API, one draw per item) or
<run>/<task>_s<seed>.jsonl (Qwen3.5-9B under vLLM, 3 decode seeds), scored per item.

  python -m brickvision.models api {gpt-6-astra,claude-opus-5.5,gpt-5.6-sol,gemini-3.8-flash,claude-opus-5,qwen3.8-max}
  python -m brickvision.models qwen {qwen3.5-9b,brickvision_direct_s17,brickvision_text_s17,brickvision_grounded_s17}
"""

import argparse
import json
import os
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import repeat
from pathlib import Path

import numpy as np
from PIL import Image

from brickvision import DATA, RESULTS, RUNS, TASKS, read_jsonl
from brickvision.evaluate import (
    INT_PAIR_RE,
    PAIR_RE,
    model_question,
    parse_open_answer,
    png_b64,
    score_points,
    score_singleton_point,
    score_xy_reply,
)

# ---------------------------------------------------------------------------------------------------------------------
# API rows. GPT and Qwen: OpenAI-compatible chat completions with `model_question` (xy, 0-1000); Claude: the same
# prompt in pixels; Gemini: the stored [y, x] questions. Error rows are kept (`evaluate tables` asserts there are none).

API_MODELS = {  # row -> (provider, model, max output tokens incl. reasoning); reasoning effort = API default
    "gpt-6-astra": ("openai", "gpt-6-astra", 16384),
    "gpt-5.6-sol": ("openai", "gpt-5.6-sol", 16384),
    "qwen3.8-max": ("dashscope", "qwen3.8-max", 65536),  # paper: snapshot qwen3.8-max-0902
    "gemini-3.8-flash": ("gemini", "gemini-3.8-flash", 65536),
    "claude-opus-5.5": ("anthropic", "claude-opus-5-5", 64000),  # API-default effort: medium
    "claude-opus-5": ("anthropic", "claude-opus-5", 64000),
}
# Alibaba Cloud Model Studio keys are region-bound: set DASHSCOPE_BASE_URL outside Singapore
DASHSCOPE_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "dashscope": (DASHSCOPE_BASE_URL.rstrip("/") + "/chat/completions", "DASHSCOPE_API_KEY"),
}
GEMINI_COUNT_SUFFIX = "\n\nStudy the image carefully. Answer with only the answer, with no explanation."
NORM_TAIL = (
    "Coordinates are integers normalized to 0-1000 of image WIDTH (x) and HEIGHT (y), origin at the top-left corner."
)
WORKERS = 16


def retry(call, tries=6):
    for k in range(tries - 1):
        try:
            return call()
        except Exception:
            time.sleep(min(60, 3 * 2**k))
    return call()


def ask_openai(item, data, bucket, model, max_tokens, url, key):
    question = model_question(item)
    image_url = {"url": "data:image/png;base64," + png_b64(data / item["image"]), "detail": "high"}
    body = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": image_url}, {"type": "text", "text": question}],
            }
        ],
    }
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    resp = retry(lambda: json.load(urllib.request.urlopen(request, timeout=900)))
    choice, usage = resp["choices"][0], resp.get("usage", {})
    raw = choice["message"].get("content") or ""
    row = {
        "model": resp.get("model"),
        "bucket": bucket,
        "question": question,
        "raw": raw[:600],
        "finish_reason": choice.get("finish_reason"),
    }
    row.update(empty=raw.strip() == "", prompt_tokens=usage.get("prompt_tokens", 0))
    row["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    row["completion_tokens"] = usage.get("completion_tokens", 0)
    row["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    return {**row, **score_xy_reply(raw, item, data)}


def ask_anthropic(item, data, bucket, model, max_tokens, client):
    question = model_question(item)
    if "mask" in item:  # Claude points in image pixels, so its pointing contract asks for pixels
        w, h = item["image_wh"]
        pixel = f"Coordinates are integer PIXELS in this {w}x{h} image: x from the left edge (0-{w - 1}), y from the top edge (0-{h - 1})."
        question = question.replace(NORM_TAIL, pixel)
    image = {"type": "base64", "media_type": "image/png", "data": png_b64(data / item["image"])}
    content = [{"type": "image", "source": image}, {"type": "text", "text": question}]
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}

    def call():
        with client.messages.stream(**kwargs) as stream:  # thinking tokens count against max_tokens
            return stream.get_final_message()

    msg = retry(call)
    raw = "".join(b.text for b in msg.content if b.type == "text")
    u = msg.usage
    row = {
        "model": msg.model,
        "coords": "pixel",
        "bucket": bucket,
        "question": question,
        "raw": raw[:600],
        "stop_reason": msg.stop_reason,
    }
    row.update(empty=raw.strip() == "", cached_tokens=u.cache_read_input_tokens or 0, completion_tokens=u.output_tokens)
    row["prompt_tokens"] = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
    if msg.stop_reason == "refusal":
        row["refusal_category"] = getattr(getattr(msg, "stop_details", None), "category", None)
    return {**row, **score_xy_reply(raw, item, data, pixel=True)}


def ask_gemini(item, data, bucket, model, max_tokens, client):
    """Pointing: stored question, JSON mode, (y, x) pairs; Counting: question + suffix, no JSON mode."""
    from google.genai import types

    counting = "mask" not in item
    extra = {} if counting else {"response_mime_type": "application/json"}
    config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_tokens, **extra)

    def call():
        with Image.open(data / item["image"]) as source:
            image = source.convert("RGB")
        prompt = item["question"] + GEMINI_COUNT_SUFFIX if counting else item["question"]
        return client.models.generate_content(model=model, contents=[image, prompt], config=config)

    response = retry(call)
    raw = response.text or ""
    usage = response.usage_metadata
    cands = response.candidates or []
    finish = getattr(cands[0], "finish_reason", None) if cands else None
    if counting:
        fields = (
            "prompt_token_count",
            "candidates_token_count",
            "thoughts_token_count",
            "cached_content_token_count",
            "total_token_count",
        )
        pred, route = parse_open_answer(raw)
        row = {
            "image": item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "meta": item.get("meta") or {},
        }
        row.update(model=model, response=raw, pred_answer=pred, parse_route=route, correct=pred == str(item["answer"]))
        row.update(finish_reason=str(finish) if finish is not None else None)
        return {**row, "usage": {f: int(getattr(usage, f, 0) or 0) for f in fields}}
    mask = np.asarray(Image.open(data / item["mask"]))
    row = {"model": model, "finish_reason": str(finish) if cands else None}
    row["prompt_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
    row["thought_tokens"] = getattr(usage, "thoughts_token_count", 0) or 0
    row["output_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
    if item["family"] == "singleton_design_pointing_geo":  # Unpaired: integer pairs only, own scorer
        points_yx = [(int(a), int(b)) for a, b in INT_PAIR_RE.findall(raw)]
        row.update(target_label=item["target_label"], total_stacks=item["meta"]["total_stacks"])
        row.update(group_count=item["meta"]["group_count"], points_yx_1000=points_yx, raw=raw[:300])
        return {**row, **score_singleton_point(points_yx, item, mask)}
    points_yx = [(float(a), float(b)) for a, b in PAIR_RE.findall(raw)]
    row.update(target_labels=item["target_labels"], bucket=bucket, points_yx_1000=points_yx, raw=raw[:400])
    return {**row, **score_points(points_yx, item, mask)}


def run_api(name):
    provider, model, max_tokens = API_MODELS[name]
    if provider in ENDPOINTS:
        url, key_env = ENDPOINTS[provider]
        ask = partial(ask_openai, model=model, max_tokens=max_tokens, url=url, key=os.environ[key_env])
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(max_retries=0, timeout=1800.0)
        ask = partial(ask_anthropic, model=model, max_tokens=max_tokens, client=client)
    else:
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        ask = partial(ask_gemini, model=model, max_tokens=max_tokens, client=client)

    def run_one(idx, item, data, bucket):
        head = {"idx": idx, "scene_id": item["scene_id"], "family": item["family"]}
        if provider != "gemini":
            head["model_requested"] = model
        try:
            return {**head, "error": None, **ask(item, data, bucket)}
        except Exception as exc:
            return {**head, "error": repr(exc)[:300]}

    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, field in TASKS.items():
        data = DATA / "restyled" / task
        items = read_jsonl(data / "items.jsonl")
        buckets = [item["meta"].get(field) for item in items]
        with ThreadPoolExecutor(WORKERS) as pool, open(out_dir / f"{task}.jsonl", "w") as fh:
            for row in pool.map(run_one, range(len(items)), items, repeat(data), buckets):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                if row["error"]:
                    print(f"{task}[{row['idx']}] ERROR {row['error']}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Qwen3.5-9B (base, or an SFT run in $GR_RUNS/sft/<run>): `model_question` under the base chat template, thinking off
# for base / direct, default for text / grounded; answer = text after the last </think> (thinking on and no </think>
# -> wrong). Temperature 1, top-p 1, 8192 new tokens, request k gets seed + k.

BASE = "Qwen/Qwen3.5-9B"  # chat template / tokenizer source for every run
THINKING = {
    "qwen3.5-9b": "off",
    "brickvision_direct": "off",
    "brickvision_text": "default",
    "brickvision_grounded": "default",
}
SEEDS = (1000, 2000, 3000)


def split_answer(text, thinking):
    """-> (answer text after the last </think>, think_unclosed)."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], False
    return (text, False) if thinking == "off" else ("", True)


def run_qwen(name):
    thinking = THINKING[re.sub(r"_s\d+$", "", name)]
    model = BASE
    if name != "qwen3.5-9b":
        model = str(RUNS / "sft" / name)
        from huggingface_hub import snapshot_download

        base_dir = Path(snapshot_download(BASE, allow_patterns=["*.json", "*.jinja"]))
        for f in (
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "chat_template.jinja",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            if not (Path(model) / f).exists() and (base_dir / f).exists():  # processor files the SFT export lacks
                shutil.copy(base_dir / f, Path(model) / f)
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(BASE)
    template_kwargs = {"enable_thinking": False} if thinking == "off" else {}
    units, prompts = [], []  # units: (task, idx, item, data dir, bucket field)
    for task, bucket in TASKS.items():
        data = DATA / "restyled" / task
        for i, item in enumerate(read_jsonl(data / "items.jsonl")):
            units.append((task, i, item, data, bucket))
            content = [{"type": "image"}, {"type": "text", "text": model_question(item)}]
            chat = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            with Image.open(data / item["image"]) as im:
                prompts.append({"prompt": chat, "multi_modal_data": {"image": im.convert("RGB")}})
    llm = LLM(
        model=model,
        tokenizer=BASE,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        limit_mm_per_prompt={"image": 1},
    )
    (RESULTS / name).mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        params = [
            SamplingParams(temperature=1.0, top_p=1.0, max_tokens=8192, seed=seed + k) for k in range(len(prompts))
        ]
        per_task = {}
        for (task, i, item, data, bucket), out in zip(units, llm.generate(prompts, params)):
            text = out.outputs[0].text
            answer, unclosed = split_answer(text, thinking)
            row = {"idx": i, "scene_id": item["scene_id"], "family": item["family"], "model": name, "seed": seed}
            row.update(
                thinking=thinking, error=None, bucket=item["meta"].get(bucket), raw=text, answer_text=answer[:400]
            )
            row.update(finish_reason=out.outputs[0].finish_reason, think_unclosed=unclosed)
            row.update(completion_tokens=len(out.outputs[0].token_ids), empty=answer.strip() == "")
            row.update(score_xy_reply(answer, item, data))
            if row["kind"] == "open_int" and not answer.strip():
                row["parse_route"] = "empty"
            per_task.setdefault(task, []).append(row)
        for task, rows in per_task.items():
            (RESULTS / name / f"{task}_s{seed}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a model on the restyled BrickVision.")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("api", help="zero-shot API eval of one Table 4 row").add_argument("row", choices=API_MODELS)
    qwen = sub.add_parser("qwen", help="vLLM eval of Qwen3.5-9B, 3 decode seeds")
    qwen.add_argument("run", help="qwen3.5-9b (the base model) or brickvision_{direct,text,grounded}_s<seed>")
    args = ap.parse_args()
    if args.command == "api":
        run_api(args.row)
    else:
        run_qwen(args.run)


if __name__ == "__main__":
    main()
