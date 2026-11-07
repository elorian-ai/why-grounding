# BrickVision

BrickVision tests fine-grained visual search and comparison. Each item is a photo of a procedurally generated brick
scene: an isometric render restyled into a photograph with the layout unchanged, so the answer is exact (it comes
from the scene geometry) and the render's segmentation masks still apply pixel for pixel. 4 tasks x 250 items:

| task | the model must | answer |
|---|---|---|
| Crop ID (`crop_id`) | find which of ten look-alike structures an enlarged inset shows and point to its outlined piece | 1 point |
| Fine-grained counting (`counting`) | count the 2x4 bricks with a 1x1 on top / a 1x1 at a short end / lying crosswise on their support | integer |
| Duplicate (`duplicate`) | point to the only two stacks with the same design | 2 points |
| Unpaired (`unpaired`) | point to the one stack that has no twin | 1 point |

## Download

```bash
hf download elorian-ai/BrickVision --repo-type dataset --local-dir data/brickvision   # $GR_DATA/brickvision
```
`restyled/<task>/` is the benchmark (`items.jsonl`, `samples/*.png`); `synthetic/<task>/` holds the source renders,
masks and scene descriptions. The items' `question` is the prompt used for the Gemini models (points as [y, x]);
`score --prompts` writes the prompts for every other model.

## Evaluate a model

```bash
python -m brickvision.evaluate score --prompts prompts.jsonl       # one {task, scene_id, image, prompt} row per item
python -m brickvision.evaluate score --predictions preds.jsonl     # rows {task, scene_id, response}; [--pixel] [--out FILE]
```
The prompts ask for `[{"point_2d": [x, y]}]` in 0-1000 coordinates (`--pixel`: image pixels). `response` is the final
answer without the reasoning, since every coordinate pair in it counts as a point. A pointing reply is correct iff it
has exactly one point per target, each inside that target's mask; a Counting reply iff its integer equals the gold
count. Add `"sample": k` to score several samples per item (mean +- SD).

## Results (paper Table 4, exact accuracy %)

| model | Crop ID | Fine-grained counting | Duplicate | Unpaired | Overall |
|---|---|---|---|---|---|
| GPT-6 Astra | 90.0 | 25.2 | 74.8 | 84.8 | 68.7 |
| Claude Opus 5.5 | 40.4 | 16.8 | 19.6 | 27.6 | 26.1 |
| GPT-5.6 Sol | 35.2 | 8.4 | 11.6 | 20.4 | 18.9 |
| Gemini 3.8 Flash | 20.4 | 15.2 | 8.8 | 17.6 | 15.5 |
| Claude Opus 5 | 26.8 | 10.4 | 8.4 | 11.6 | 14.3 |
| Qwen3.8-Max | 6.4 | 2.0 | 13.2 | 12.4 | 8.5 |
| Qwen3.5-9B (base) | 1.5 | 8.5 | 2.4 | 8.4 | 5.2 |
| Finetuned Direct | 8.1 | 16.0 | 8.3 | 7.7 | 10.0 |
| Finetuned Text Reasoning | 85.2 | 66.4 | 39.1 | 54.1 | 61.2 |
| Finetuned Grounded Reasoning | 91.6 | 95.3 | 64.7 | 79.9 | 82.9 |

Models trained on data from `brickvision.generate` (like the fine-tuned rows above) should be reported separately
from models that were not.

## Generate the data

```bash
python -m brickvision.generate crop_id   --seed 66002 --out data/brickvision/synthetic/crop_id
python -m brickvision.generate counting  --seed 61002 --out data/brickvision/synthetic/counting
python -m brickvision.generate duplicate --seed 64004 --out data/brickvision/synthetic/duplicate
python -m brickvision.generate unpaired  --seed 56002 --out data/brickvision/synthetic/unpaired
python -m brickvision.restyle    # Nano Banana restyle + structure judge (GEMINI_API_KEY) -> restyled/<task>/
```
Model runs: `python -m brickvision.models api <row>` or `python -m brickvision.models qwen <run>`; the table:
`python -m brickvision.evaluate tables`; the SFT sets: `bash scripts/brickvision.sh sft-data`.

Citation: see [`CITATION.cff`](../CITATION.cff).
