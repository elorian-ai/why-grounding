"""BrickVision: procedurally generated brick scenes restyled into photos, with exact answers from the scene geometry.

Data in $GR_DATA/brickvision, results in $GR_RESULTS/brickvision, checkpoints in $GR_RUNS/sft (defaults: <repo>/data,
<repo>/results, <repo>/runs).
"""

import json
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("GR_DATA", _REPO / "data")) / "brickvision"
RESULTS = Path(os.environ.get("GR_RESULTS", _REPO / "results")) / "brickvision"
RUNS = Path(os.environ.get("GR_RUNS", _REPO / "runs"))

# Task order (fixes the SFT row order and the per-request vLLM seeds) -> the item `meta` field logged as `bucket`.
TASKS = {"duplicate": "K", "crop_id": "highlight_role", "unpaired": "total_stacks", "counting": "relation"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
