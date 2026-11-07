"""Evaluate one checkpoint on the gr_* lmms-eval tasks (tasks/*.yaml) with vLLM on one GPU, and load the results.

gr_sampled: temperature 1, request seed = --seed + doc_id; gr_greedy: greedy. Output: <out>/<task>.jsonl, one row per
item (doc_id, images, raw response, output_tokens + the record of benchmarks.process_results); lmms-eval's own logs in
<out>/lmms_eval/. paper/ reads <run>/sampled/seed<S>/<task>.jsonl and <run>/greedy/<task>.jsonl with load_run.
    python -m grounded_reasoning.evaluate --model Qwen/Qwen3-VL-8B-Instruct --tasks gr_sampled --seed 1000 --out results/eval/base/sampled/seed1000
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import statistics as st
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from grounded_reasoning import REPO
from grounded_reasoning.benchmarks import load_items

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
# limit_mm_per_prompt 16 = the most images of any item (ERQA); vLLM rejects a whole batch with a larger item.
MODEL_ARGS = (
    "tensor_parallel_size=1,gpu_memory_utilization=0.90,max_pixels=1605632,max_new_tokens=8192,"
    'max_model_len=49152,dtype=bfloat16,limit_mm_per_prompt={"image":16}'
)
BATCH_SIZE = 2048  # prompts per vLLM call


def __getattr__(name):
    """`SeededVLLM`, built on first access: lmms-eval loads it by name, and the result loaders need no lmms-eval."""
    if name != "SeededVLLM":
        raise AttributeError(name)
    from lmms_eval.api.instance import GenerationResult, TokenCounts
    from lmms_eval.models.chat.vllm import VLLM, WORKERS, SamplingParams

    class SeededVLLM(VLLM):
        """lmms-eval's chat vLLM backend with one SamplingParams per request; sampled requests get seed seed_base +
        doc_id."""

        def make_one_request(self, request):
            _, _, gen_kwargs, doc_id, _, _ = request.arguments
            messages, params = super().make_one_request(request)
            seed_base = (gen_kwargs or {}).get("seed_base")
            if float(params.get("temperature", 0)) > 0 and seed_base is not None:
                params["seed"] = int(seed_base) + int(doc_id)
            return messages, params

        def generate_until(self, requests) -> list[GenerationResult]:
            results = []
            pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
            for start in range(0, len(requests), self.batch_size_per_gpu):
                batch = requests[start : start + self.batch_size_per_gpu]
                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    built = list(pool.map(self.make_one_request, batch))
                messages = [m for m, _ in built]
                params = [SamplingParams(**p) for _, p in built]

                def run(inputs, params=params):
                    outputs = self.client.chat(
                        messages=inputs, sampling_params=params, chat_template=self.chat_template
                    )
                    return [
                        (o.outputs[0].text, len(o.prompt_token_ids or []), len(o.outputs[0].token_ids or []))
                        for o in outputs
                    ]

                for text, n_in, n_out in self._run_tp_synced(messages, run):
                    tokens = TokenCounts(input_tokens=n_in, output_tokens=n_out)
                    results.append(GenerationResult(text=text, token_counts=tokens))
                pbar.update(len(batch))
            pbar.close()
            return results

    globals()["SeededVLLM"] = SeededVLLM
    return SeededVLLM


def result_row(sample: dict, item: dict) -> dict:
    """One lmms-eval sample log entry + its item -> our result row."""
    tokens = (sample.get("token_counts") or [None])[0] or {}
    return {
        "doc_id": sample["doc_id"],
        "images": item["images"],
        # lmms-eval omits `resps` when the raw response equals the scored (reasoning-stripped) text
        "response": sample.get("resps", sample["filtered_resps"]),
        "output_tokens": tokens.get("output_tokens"),
        **sample["acc_parsed"],
    }


def write_results(lmms_dir: Path, out_dir: Path, since: float = 0.0) -> dict[str, list[dict]]:
    """lmms-eval sample logs (<lmms_dir>/<model>/<date>_samples_<task>.jsonl, the newest per task written after
    `since`) -> <out_dir>/<task>.jsonl. Returns {task: rows}."""
    logs = {}
    for path in sorted(lmms_dir.rglob("*_samples_*.jsonl"), key=lambda p: p.name):  # names start with the date
        if path.stat().st_mtime >= since:
            logs[path.stem.split("_samples_", 1)[1]] = path
    written = {}
    for task, path in logs.items():
        items = load_items(task.removeprefix("gr_"))
        with open(path) as f:
            rows = [result_row(sample, items[sample["doc_id"]]) for sample in map(json.loads, f)]
        with open(out_dir / f"{task}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        written[task] = rows
    return written


def main():
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint on gr_* lmms-eval tasks.")
    parser.add_argument("--model", required=True, help="HF checkpoint directory or hub id")
    parser.add_argument("--tasks", required=True, help="gr_sampled, gr_greedy or comma-separated gr_* tasks")
    parser.add_argument("--seed", type=int, default=1000, help="decode seed base of the sampled tasks")
    parser.add_argument("--out", type=Path, required=True, help="directory for <task>.jsonl")
    parser.add_argument("--limit", type=int, help="evaluate only the first N items of each task (smoke test)")
    args = parser.parse_args()

    from lmms_eval.__main__ import cli_evaluate
    from lmms_eval.models import MODEL_REGISTRY_V2
    from lmms_eval.models.registry_v2 import ModelManifest
    from lmms_eval.tasks import TaskManager

    # Index only the gr_* tasks (a git install of lmms-eval lacks the built-in tasks' `_default_template_yaml` files).
    TaskManager.__init__ = functools.partialmethod(TaskManager.__init__, include_defaults=False)
    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(model_id="vllm_seeded", chat_class_path="grounded_reasoning.evaluate.SeededVLLM"),
        overwrite=True,
    )
    model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # One vLLM compile cache per GPU: parallel evaluations of one checkpoint would race on the same kernel files.
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all").replace(",", "-")
    os.environ.setdefault("VLLM_CACHE_ROOT", os.path.join(tempfile.gettempdir(), f"vllm_gpu{gpus}"))
    sys.argv = ["lmms-eval", "--model", "vllm_seeded", "--model_args", f"model={model},{MODEL_ARGS}"]
    sys.argv += ["--tasks", args.tasks, "--include_path", str(TASKS_DIR), "--gen_kwargs", f"seed_base={args.seed}"]
    sys.argv += ["--batch_size", str(BATCH_SIZE), "--output_path", str(out / "lmms_eval"), "--log_samples"]
    sys.argv += ["--limit", str(args.limit)] if args.limit else []
    started = time.time()
    os.chdir(REPO)  # the task yamls' data_files are relative to the repository root
    cli_evaluate()  # logs (does not raise) evaluation errors, hence the check below

    written = write_results(out / "lmms_eval", out, since=started)
    if not written:
        sys.exit(f"no lmms-eval sample logs under {out / 'lmms_eval'}; see the error above")
    for task, rows in written.items():
        print(f"{task}: {len(rows)} items, acc_raw {sum(r['hit'] for r in rows) / len(rows):.4f} -> {out}/{task}.jsonl")


# Result loaders, used by the paper tables and figures. Rows need at least idx, hit and images. Example:
# load_run(RESULTS / "eval" / "sft/grounded_s17")["gr_vsr"][idx] -> (hit of item idx averaged over the decode seeds,
# its image list).


def _rows(run_dir: Path, protocol: str):
    """Yield (seed, task, row) over every result file of one protocol ("sampled" or "greedy"); seed 0 = no seed dir."""
    base = Path(run_dir) / protocol
    for path in sorted([*base.glob("*.jsonl"), *base.glob("seed*/*.jsonl")]):
        seed = int(path.parent.name[4:]) if path.parent.name.startswith("seed") else 0
        with open(path) as f:
            for line in f:
                yield seed, path.stem, json.loads(line)


def load_seeds(run_dir: Path, protocol: str = "sampled") -> dict:
    """-> {task: {seed: {idx: hit}}}."""
    out = defaultdict(dict)
    for seed, task, row in _rows(run_dir, protocol):
        out[task].setdefault(seed, {})[int(row["idx"])] = bool(row["hit"])
    return out


def load_run(run_dir: Path, protocol: str = "sampled") -> dict:
    """-> {task: {idx: (mean hit over the decode seeds, cluster key)}}; cluster key = the item's image list."""
    hits = defaultdict(lambda: defaultdict(list))
    cluster = {}
    for _, task, row in _rows(run_dir, protocol):
        idx = int(row["idx"])
        hits[task][idx].append(1.0 if row["hit"] else 0.0)
        cluster[(task, idx)] = tuple(row.get("images") or [idx])
    return {t: {i: (st.mean(h), cluster[(t, i)]) for i, h in items.items()} for t, items in hits.items()}


def clustered_se(values: dict, clusters: dict) -> float:
    """SE of a mean over clustered items (Miller 2024): SE^2 = sum_c (sum_{i in c} (x_i - xbar))^2 / n^2."""
    n = len(values)
    xbar = st.mean(values.values())
    by_c = defaultdict(float)
    for i, x in values.items():
        by_c[clusters[i]] += x - xbar
    return math.sqrt(sum(s * s for s in by_c.values())) / n


def cell(recs: dict) -> dict:
    """{idx: hit} of one (run, task, seed) -> {n, acc_raw} (unparsed = wrong)."""
    return {"n": len(recs), "acc_raw": sum(recs.values()) / len(recs)}


if __name__ == "__main__":
    main()
