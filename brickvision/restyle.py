"""Restyle the synthetic BrickVision into photos with Nano Banana (GEMINI_API_KEY); the masks carry over unchanged.

$GR_DATA/brickvision/synthetic/<task>/ -> restyled/<task>/{samples/, items.jsonl, dropped.json}. Each request sends
the scene (letterboxed to 16:9, cropped back after; Crop ID: the scene panel only) and a map of green structure boxes.
A judge call must return structure_preserved, else the next style template is tried (3 attempts, then dropped).
  python -m brickvision.restyle
"""

import argparse
import io
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from brickvision import DATA, TASKS, read_jsonl
from brickvision import render as G

OUT = DATA / "restyled"

REFINE_MODEL = "gemini-3.1-flash-image-preview"  # Nano Banana
VERIFY_MODEL = "gemini-3.7-flash"  # paired judge
WORKERS, RETRIES = 8, 3
TEMPLATE_ORDER = ["studio_product", "window_desk", "evening_lamp", "playroom_carpet"]
# task -> (structure group key, composite?)
STRUCTURES = {
    "duplicate": ("stack_index", False),
    "crop_id": ("pod_index", True),
    "unpaired": ("pod_index", False),
    "counting": ("pod_index", False),
}
GREEN = (0, 255, 0)

# Appended to every style template; {total}/{plate}/{inventory} filled per scene.
STRUCTURE_LOCK = (
    "\n\nSTRUCTURE LOCK — this is a synthetic benchmark image; the brick layout is "
    "ground truth and must survive your edit EXACTLY:\n"
    "- The scene contains exactly {total} Lego pieces on a {plate}x{plate}-stud "
    "baseplate: {inventory}. Your output must contain exactly these pieces and "
    "nothing else.\n"
    "- Do NOT add, remove, duplicate, merge, split, move, rotate, resize, restack, "
    "or recolor ANY brick. Every brick keeps its exact stud position, orientation, "
    "color, and size.\n"
    "- Do NOT add minifigures, text, logos, stickers, decals, printed tiles, extra "
    "loose studs, or any new object on or touching the baseplate.\n"
    "- Keep the camera viewpoint, framing, crop, and every brick's position in the "
    "image identical to the source render.\n"
    "- ONLY change: surface material realism (ABS plastic sheen, micro-texture), "
    "lighting, shadows, color grading, photographic qualities (noise, focus), and "
    "the surroundings OUTSIDE the baseplate.\n"
    "- If any styling instruction conflicts with preserving the brick layout, "
    "preserving the layout wins."
)
BOUNDARY_NOTE = (
    "\n\nBOUNDARY MAP — the second image is the same scene with green rectangles "
    "marking every brick structure. Every structure must remain entirely inside "
    "its marked rectangle at the same pixel location. Keep the EXACT camera "
    "framing and canvas aspect of the first image; do not crop, zoom, recenter, "
    "or recompose."
)
VERIFY_PROMPT = (
    "IMAGE 1 is a ground-truth synthetic Lego render. IMAGE 2 is a photorealistic "
    "restyle of it that is REQUIRED to contain the identical brick layout.\n"
    "Ground-truth inventory for both images: exactly {total} Lego pieces on a "
    "{plate}x{plate}-stud baseplate: {inventory}.\n\n"
    "Compare the two images piece by piece. Ignore lighting, materials, shadows, "
    "background, noise, and color-grading differences; report ONLY structural "
    "differences in the bricks themselves.\n"
    "Answer with JSON only:\n"
    '{{"bricks_added": <int>, "bricks_removed": <int>, "bricks_recolored": <int>, '
    '"bricks_moved_or_resized": <int>, "baseplate_changed": <bool>, '
    '"new_objects": [<str>...], "structure_preserved": <bool>, '
    '"confidence": <float 0-1>, "notes": <str>}}\n'
    "new_objects counts ONLY new objects ON or touching the baseplate; added "
    "background/scenery OUTSIDE the baseplate (desk, carpet, lamp, sky) is allowed "
    "and must be ignored.\n"
    "structure_preserved must be true ONLY if every count is 0, baseplate_changed "
    "is false, and new_objects is empty. Be strict: any hallucinated, missing, "
    "moved, or recolored brick fails."
)


def inventory_text(scene):
    """Ground-truth bricks -> '4 green 2x4 bricks, 8 gray 1x1 bricks, ...' (sorted for determinism)."""
    counts = Counter()
    for b in scene["bricks"]:
        dims = tuple(sorted((b["lx"], b["ly"])))
        counts[(b["color"], dims)] += 1
    return ", ".join(
        f"{n} {color} {dims[0]}x{dims[1]} {'brick' if n == 1 else 'bricks'}"
        for (color, dims), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )


def render_base(scene):
    """Re-render a stored scene's base panel (plate recolored) + id mask, byte-identical to the generator."""
    image, id_mask = G.rerender(scene)
    if scene.get("plate_theme"):
        image = G.recolor_plate(image, scene["plate_theme"])
    return image.convert("RGB"), id_mask


def structure_boxes(id_mask, bricks, group_key):
    """{structure id: [x0, y0, x1, y1]} pixel boxes over the visible bricks of each structure."""
    idm = np.asarray(id_mask)
    parts = defaultdict(list)
    for brick in bricks:
        ys, xs = np.nonzero(idm == brick["id"] + 1)
        if len(xs):
            parts[brick[group_key]].append((xs.min(), ys.min(), xs.max(), ys.max()))
    return {k: [int(v) for v in (*np.min(b, axis=0)[:2], *np.max(b, axis=0)[2:])] for k, b in parts.items()}


def letterbox_16_9(image):
    """Pad (never crop) to 16:9 with the corner color -> (padded, (ox, oy))."""
    w, h = image.size
    bg = image.getpixel((0, 0))
    if w / h < 16 / 9:
        W, H = int(round(h * 16 / 9)), h
    else:
        W, H = w, int(round(w * 9 / 16))
    canvas = Image.new("RGB", (W, H), bg)
    ox, oy = (W - w) // 2, (H - h) // 2
    canvas.paste(image, (ox, oy))
    return canvas, (ox, oy)


def draw_boxes(image, boxes, offset=(0, 0)):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for b in boxes.values():
        draw.rectangle([b[0] + offset[0], b[1] + offset[1], b[2] + offset[0], b[3] + offset[1]], outline=GREEN, width=4)
    return canvas


def request_inputs(task, item, scene, data_dir):
    """-> (source = the panel to restyle, padded 16:9 source, (ox, oy), padded boundary map)."""
    group_key, composite = STRUCTURES[task]
    if composite:
        source, id_mask = render_base(scene)
    else:
        source = Image.open(data_dir / item["image"]).convert("RGB")
        _, id_mask = render_base(scene)
    boxes = structure_boxes(id_mask, scene["bricks"], group_key)
    padded, (ox, oy) = letterbox_16_9(source)
    return source, padded, (ox, oy), draw_boxes(padded, boxes, offset=(ox, oy))


def lock_fields(scene):
    return {"total": len(scene["bricks"]), "plate": scene.get("plate", 30), "inventory": inventory_text(scene)}


def image_from_response(response):
    for part in response.candidates[0].content.parts:
        data = getattr(part, "inline_data", None)
        if data and data.data:
            return Image.open(io.BytesIO(data.data)).convert("RGB")
    return None


def main() -> None:
    argparse.ArgumentParser(description="Restyle synthetic/<task>/ into restyled/<task>/.").parse_args()
    import yaml
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    style_file = Path(__file__).with_name("style_prompts.yaml")
    templates = {t["id"]: t for t in yaml.safe_load(style_file.read_text())["templates"]}

    def process(unit):
        """Restyle one item -> nano provenance {image, template, attempts, raw_size}, or None if dropped."""
        task, idx, item, scene, data_dir = unit
        source, padded, (ox, oy), bmap = request_inputs(task, item, scene, data_dir)
        for attempt in range(RETRIES):
            template = templates[TEMPLATE_ORDER[(idx + attempt) % len(TEMPLATE_ORDER)]]
            try:
                resp = client.models.generate_content(
                    model=REFINE_MODEL,
                    contents=[
                        "IMAGE 1 (the scene to restyle):",
                        padded,
                        "IMAGE 2 (mandatory structure boundaries in green):",
                        bmap,
                        template["prompt"].strip() + STRUCTURE_LOCK.format(**lock_fields(scene)) + BOUNDARY_NOTE,
                    ],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"], image_config=types.ImageConfig(aspect_ratio="16:9")
                    ),
                )
                raw = image_from_response(resp)
                refined = raw.resize(padded.size, Image.Resampling.LANCZOS).crop(
                    (ox, oy, ox + source.width, oy + source.height)
                )
                verify = client.models.generate_content(
                    model=VERIFY_MODEL,
                    contents=[
                        "IMAGE 1 (ground truth):",
                        source,
                        "IMAGE 2 (restyled):",
                        refined,
                        VERIFY_PROMPT.format(**lock_fields(scene)),
                    ],
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
                )
                if json.loads(verify.text).get("structure_preserved"):
                    out_rel = f"samples/{item['scene_id']}.png"
                    if STRUCTURES[task][1]:  # paste the restyled base panel back into the stored composite
                        comp = Image.open(data_dir / item["image"]).convert("RGB")
                        comp.paste(refined, tuple(scene["crop"]["full_offset"]))
                        refined = comp
                    refined.save(OUT / task / out_rel)
                    return {
                        "image": out_rel,
                        "template": template["id"],
                        "attempts": attempt + 1,
                        "raw_size": list(raw.size),
                    }
            except Exception as exc:  # no image / bad JSON / API error: retried with the next template
                print(f"{task} {item['scene_id']} attempt {attempt}: {exc!r}"[:300], flush=True)
                time.sleep(min(20, 2**attempt))
        return None

    for task in TASKS:
        data_dir = DATA / "synthetic" / task
        items = read_jsonl(data_dir / "items.jsonl")
        scenes = {s["scene_id"]: s for s in read_jsonl(data_dir / "scenes.jsonl")}
        (OUT / task / "samples").mkdir(parents=True, exist_ok=True)
        units = [(task, idx, item, scenes[item["scene_id"]], data_dir) for idx, item in enumerate(items)]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            recs = list(pool.map(process, units))
        out_items = []
        for item, rec in zip(items, recs):
            if rec:
                # paths relative to restyled/<task>/, like every path in an items.jsonl
                new = {**item, "image": rec["image"], "synthetic_image": f"../../synthetic/{task}/{item['image']}"}
                if "mask" in item:
                    new["mask"] = f"../../synthetic/{task}/{item['mask']}"
                nano = {
                    **{k: rec[k] for k in ("template", "attempts", "raw_size")},
                    "aspect_locked": True,
                    "model": REFINE_MODEL,
                }
                new["meta"] = {**item.get("meta", {}), "nano": nano}
                out_items.append(new)
        (OUT / task / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in out_items))
        dropped = [item["scene_id"] for item, rec in zip(items, recs) if not rec]
        (OUT / task / "dropped.json").write_text(json.dumps(dropped))


if __name__ == "__main__":
    main()
