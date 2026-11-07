"""Per model family of the decode-randomization arms: user prompt, chat messages, stop strings, coordinate regex,
random replacement coordinate, DeepEyes crop tool and answer parser (randomize.py runs the decode loop).

  ours      our Qwen3-VL SFT checkpoints: the evaluation prompt; {"bbox_2d": [...]} boxes / [x, y] points, 0-1000
  treevgr   TreeVGR-7B-CI/RL, Qwen2.5-VL: TreeVGR system prompt, starts after "<think>"; <box>[...]</box> in resized px
  ivgr      iVGR-8B, Qwen3-VL: treevgr's system prompt and <box> dialect in 0-1000, iVGR's instruction, 1-shot example
  deepeyes  DeepEyes-7B, Qwen2.5-VL: each <tool_call> box is cropped from the image and fed back as a new image
"""

import math
import re

from PIL import Image

from grounded_reasoning.benchmarks import LETTERS, SUFFIX, extract_last_boxed, parse_mcq_index

TREEVGR_SYS = (
    "A conversation between user and assistant. The user asks a question, and the "
    "Assistant solves it. The assistant MUST first think about the reasoning process "
    "in the mind and then provide the user with the answer. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively. When referring to particular objects in the reasoning process, the "
    "assistant MUST localize the object with bounding box coordinates between <box> and "
    "</box>. You MUST strictly follow the format."
)
TREEVGR_MCQ_SUFFIX = (
    "\nSelect the best answer to the above multiple-choice question based on the image. "
    "After the reasoning process, respond with only the letter of the correct option "
    "between <answer> and </answer>."
)
# Counting prompt for the <answer> families (treevgr, ivgr).
COUNT_SUFFIX = "\nAfter the reasoning process, respond with only the final number between <answer> and </answer>."

IVGR_MCQ_SUFFIX = (
    "\nSelect the best answer to the above multiple-choice question based on the image. "
    "After the reasoning process, respond with ONLY the letter of the correct option "
    "between <answer> and </answer>."
)
IVGR_EXAMPLE = (
    "\n**Example**:\nUser: What is the man doing in the image?\n"
    "Assistant: <think>The image shows a man in a kitchen, wearing an apron and holding a "
    "wooden spoon while stirring a pot on the stove. He appears to be cooking a meal.</think> "
    "<answer>The man is cooking a meal in the kitchen.</answer>"
)

# DeepEyes: the released eval's system prompt (tool schema), user layout and tool responses.
DE_SYS = (
    "You are a helpful assistant.\n"
    "\n"
    "# Tools\n"
    "You may call one or more functions to assist with the user query.\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}\n'
    "</tools>\n"
    "\n"
    "# How to call a tool\n"
    "Return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n"
    "\n"
    "**Example**:  \n"
    "<tool_call>  \n"
    '{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  \n'
    "</tool_call>"
)
DE_USER = (
    "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:"
    "  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
)
# Appended to the response after a tool call: the crop comes back as a user turn.
DE_OBS = (
    "<|im_end|>\n<|im_start|>user\n<tool_response><|vision_start|><|image_pad|><|vision_end|>"
    + DE_USER
    + "</tool_response><|im_end|>\n<|im_start|>assistant\n"
)
DE_ERR = "<|im_end|>\n<|im_start|>user\nError: {}<|im_end|>\n<|im_start|>assistant\n"
DE_FACTOR, DE_MIN_PIXELS, DE_MAX_PIXELS = 28, 4 * 28 * 28, 16384 * 28 * 28


def mcq_options_block(question, options):
    """Question + options as "(a) ..." lines (the ours / ivgr layout)."""
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    return f"{question}\n{opts}"


def de_user_text(question, options):
    """DeepEyes eval layout; options=None -> question only (counting)."""
    if options is None:
        return f"Question: {question}\n" + DE_USER
    opts = "\n" + "".join(f"{LETTERS[i].upper()}. {o}\n" for i, o in enumerate(options))
    return f"Question: {question}\nOptions: {opts}\n" + DE_USER


def user_text(family, bench, item):
    """(family, bench, canonical item of randomize.load_bench) -> user prompt text."""
    question, options = item["question"], item.get("options")
    if bench == "countqa":
        return {
            "ours": question.rstrip() + SUFFIX,
            "treevgr": question.rstrip() + COUNT_SUFFIX,
            "ivgr": question.rstrip() + COUNT_SUFFIX + IVGR_EXAMPLE,
            "deepeyes": de_user_text(question, None),
        }[family]
    if family == "ours":
        # MultihopSpatial questions already embed their options as "<choice>(a) ..., (b) ...</choice>".
        return (question if bench == "multihopspatial" else mcq_options_block(question, options)) + SUFFIX
    if family == "treevgr":
        if bench == "multihopspatial":
            return question + TREEVGR_MCQ_SUFFIX
        # SpatialMQA uses lowercase letters, V* uppercase (as the two TreeVGR harnesses do).
        letters = LETTERS if bench == "spatialmqa" else LETTERS.upper()
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        return question + " Options:\n" + opts + TREEVGR_MCQ_SUFFIX
    if family == "ivgr":
        return mcq_options_block(question, options) + IVGR_MCQ_SUFFIX + IVGR_EXAMPLE
    if family == "deepeyes":
        return de_user_text(question, options)
    raise ValueError(family)


def build_items(family, bench, items):
    """Canonical bench items -> the same items plus `user_text` for this family."""
    return [{**it, "user_text": user_text(family, bench, it)} for it in items]


def chat_messages(family, text):
    """-> (chat messages, string appended after the generation prompt)."""
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
    if family == "ours":
        return [user], ""
    if family == "deepeyes":
        return [{"role": "system", "content": DE_SYS}, user], ""
    return [{"role": "system", "content": [{"type": "text", "text": TREEVGR_SYS}]}, user], "<think>"


OURS_COORD_RE = re.compile(
    r'\{"bbox_2d":\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\}'  # box g1-4
    r"|\[\s*(\d+)\s*,\s*(\d+)\s*\]"  # point g5-6
)
BOX_RE = re.compile(r"<box>\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*</box>")
# One complete tool call. The tempered dot never crosses a </tool_call>, so a malformed call
# followed by a good one yields ONE match whose span starts at the malformed call's tag.
_DE_NUM = r"(-?\d+(?:\.\d+)?)"
DE_TOOL_RE = re.compile(
    r"<tool_call>(?:(?!</tool_call>).)*?\"bbox_2d\"\s*:\s*\[\s*"
    + r"\s*,\s*".join([_DE_NUM] * 4)
    + r"\s*\](?:(?!</tool_call>).)*?</tool_call>",
    re.DOTALL,
)
COORD_RES = {"ours": OURS_COORD_RE, "treevgr": BOX_RE, "ivgr": BOX_RE, "deepeyes": DE_TOOL_RE}
# Stop strings that end a coordinate group, so the loop can intervene right after it.
STOPS = {"ours": ["]", "}"], "treevgr": ["</box>"], "ivgr": ["</box>"], "deepeyes": ["</tool_call>"]}


def random_box(rng, width, height):
    x1 = rng.randint(0, max(1, width - 40))
    y1 = rng.randint(0, max(1, height - 40))
    x2 = rng.randint(x1 + 20, width)
    y2 = rng.randint(y1 + 20, height)
    return x1, y1, x2, y2


def random_coordinate(family, match, rng, res_w, res_h):
    """Coordinate match -> text with freshly drawn values (consumes `rng`); the engine copies only its numbers."""
    if family == "deepeyes":
        values = de_random_bbox(rng, res_w, res_h)
        if values is None:  # no valid draw in 100 tries: leave the call unchanged
            return match.group(0)
        out, base = match.group(0), match.start()
        for k in (4, 3, 2, 1):
            a, b = match.span(k)
            out = out[: a - base] + str(values[k - 1]) + out[b - base :]
        return out
    if family == "ours":
        if match.group(1) is None:  # point
            return f"[{rng.randint(0, 1000)}, {rng.randint(0, 1000)}]"
        x1 = rng.randint(0, 899)
        y1 = rng.randint(0, 899)
        x2 = rng.randint(x1 + 20, 1000)
        y2 = rng.randint(y1 + 20, 1000)
        return f'{{"bbox_2d": [{x1}, {y1}, {x2}, {y2}]}}'
    width, height = (1000, 1000) if family == "ivgr" else (res_w, res_h)
    x1, y1, x2, y2 = random_box(rng, width, height)
    return f"<box>[{x1},{y1},{x2},{y2}]</box>"


def de_smart_resize(height, width):
    """qwen-vl-utils smart_resize as used by the DeepEyes eval -> (h, w), multiples of 28."""
    h_bar = max(DE_FACTOR, round(height / DE_FACTOR) * DE_FACTOR)
    w_bar = max(DE_FACTOR, round(width / DE_FACTOR) * DE_FACTOR)
    if h_bar * w_bar > DE_MAX_PIXELS:
        beta = math.sqrt((height * width) / DE_MAX_PIXELS)
        h_bar = math.floor(height / beta / DE_FACTOR) * DE_FACTOR
        w_bar = math.floor(width / beta / DE_FACTOR) * DE_FACTOR
    elif h_bar * w_bar < DE_MIN_PIXELS:
        beta = math.sqrt(DE_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / DE_FACTOR) * DE_FACTOR
        w_bar = math.ceil(width * beta / DE_FACTOR) * DE_FACTOR
    return h_bar, w_bar


def de_fix_bbox(left, top, right, bottom, width, height):
    """DeepEyes toolbox: clip to the image, then require a proper box with min side > 30 and
    aspect <= 100 -> [l, t, r, b], or None (the toolbox's invalid-arguments error)."""
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if not (left < right and bottom > top):
        return None
    h, w = bottom - top, right - left
    if not (max(h, w) / min(h, w) <= 100 and min(h, w) > 30):
        return None
    return [left, top, right, bottom]


def de_random_bbox(rng, width, height):
    """Random box that passes the toolbox gate, so randomize does not hit the error path more
    often than the model's own boxes. Redraws random_box until valid (at most 100 draws)."""
    for _ in range(100):
        box = de_fix_bbox(*random_box(rng, width, height), width, height)
        if box is not None:
            return box
    return None


def de_tool_turn(state, bbox):
    """Run image_zoom_in_tool for one decode state at `bbox` (in the resW x resH frame the model
    sees): crop the original image at the scaled box, smart-resize it, append it to
    state["images"] -> observation text to append to the response."""
    fixed = de_fix_bbox(*bbox, state["resW"], state["resH"])
    if fixed is None:
        return DE_ERR.format("ZOOM IN ARGUMENTS ARE INVALID")
    img = state["img"]
    sx, sy = img.width / state["resW"], img.height / state["resH"]
    left, top, right, bottom = fixed
    x0, y0 = int(round(left * sx)), int(round(top * sy))
    x1, y1 = max(x0 + 1, int(round(right * sx))), max(y0 + 1, int(round(bottom * sy)))
    crop = img.crop((x0, y0, x1, y1))
    new_h, new_w = de_smart_resize(crop.height, crop.width)
    state["images"].append(crop.resize((new_w, new_h), resample=Image.BICUBIC))
    return DE_OBS


# Final answer: last balanced \boxed{} (ours) or last non-empty <answer> (others); unparsed = wrong.
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TEXT_WRAP = re.compile(r"\\(?:text|mathrm|mathbf|textbf)\s*\{([^{}]*)\}")
_OURS_INT = re.compile(r"-?\d[\d,]*")
OURS_COUNT_MAX = 100000


def answer_span(family, resp):
    """Response -> final answer text, or None if there is none."""
    if family == "ours":
        return extract_last_boxed(resp)
    spans = [m for m in ANSWER_RE.findall(resp) if m.strip()]
    return spans[-1] if spans else None


# Count parsing for the <answer> families: digits in the answer span (after removing every
# coordinate syntax), else a spelled-out number ("<answer>Six</answer>").
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL = re.compile(
    r"(?:final answer|final count|in total|total(?:\s+\w+){0,3}\s+is|"
    r"there are|i count(?:ed)?|answer(?:\s+is)?)\D{0,15}?(\d{1,4})\b",
    re.I,
)
_INT = re.compile(r"\b(\d{1,6})\b")
_ENUM = re.compile(r"^\s*(\d{1,4})[.)]\s", re.M)
_COORDS = [
    re.compile(r'\{\s*"bbox_2d"\s*:\s*\[[^\]]*\]\s*\}'),
    re.compile(r"\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]"),
    re.compile(r"\[\s*-?\d+\s*,\s*-?\d+\s*\]"),
    re.compile(r"\[\s*x\s*=\s*-?\d+\s*,\s*y\s*=\s*-?\d+\s*\]", re.I),
    re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)"),
]
_DANGLING = re.compile(r"\[[^\[\]]*$")
_BOX_TAG = re.compile(r"<box>.*?</box>", re.S)
_BOX_DANGLING = re.compile(r"<box>[^<]*$", re.S)
_TUPLE34 = re.compile(r"\[\s*-?\d+(?:\s*,\s*-?\d+){2,3}\s*\]")
_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
_W_UNIT = {w: i for i, w in enumerate(_UNITS.split())}
_W_TEN = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_W_ALL = sorted([*_W_UNIT, *_W_TEN, "hundred", "and"], key=len, reverse=True)
_PHRASE = re.compile(r"\b((?:(?:" + "|".join(_W_ALL) + r")[\s-]*)+)", re.I)


def _strip_boxes(text):
    text = _BOX_TAG.sub(" ", text)
    text = _BOX_DANGLING.sub(" ", text)
    return _TUPLE34.sub(" ", text)


def _parse_count(text, max_plausible=1000):
    """\\boxed{} -> final-answer phrase -> last enumeration marker -> last plausible integer,
    after stripping coordinate syntax and any unterminated bracket."""
    m = _BOXED.findall(text)
    if m:
        d = _INT.search(m[-1])
        return int(d.group(1)) if d else None
    for rx in _COORDS:
        text = rx.sub(" ", text)
    text = _DANGLING.sub(" ", text)
    d = _FINAL.findall(text)
    if d:
        return int(d[-1])
    e = _ENUM.findall(text)
    if e:
        return int(e[-1])
    d = [int(x) for x in _INT.findall(text) if int(x) <= max_plausible]
    return d[-1] if d else None


def _words_to_int(phrase, max_plausible=1000):
    """'twenty-four' -> 24, 'one hundred and five' -> 105, 'and' -> None."""
    toks = [t for t in re.split(r"[\s-]+", phrase.strip().lower()) if t and t != "and"]
    if not toks:
        return None
    cur, seen = 0, False
    for t in toks:
        if t in _W_UNIT:
            cur += _W_UNIT[t]
            seen = True
        elif t in _W_TEN:
            cur += _W_TEN[t]
            seen = True
        elif t == "hundred":
            if not seen:
                return None
            cur = max(cur, 1) * 100
        else:
            return None
    return cur if seen and cur <= max_plausible else None


def _spelled_count(text):
    """Last spelled-out number phrase in `text` that parses, or None."""
    for m in reversed(list(_PHRASE.finditer(text))):
        value = _words_to_int(m.group(1))
        if value is not None:
            return value
    return None


def score_item(family, bench, item, resp):
    """-> (pred, hit). MCQ pred is an uppercase letter; CountQA pred is an int."""
    span = answer_span(family, resp)
    if bench == "countqa":
        if span is None:
            return None, False
        if family == "ours":
            m = _OURS_INT.search(_TEXT_WRAP.sub(r"\1", span))
            pred = None
            if m:
                value = int(m.group(0).replace(",", ""))
                pred = value if 0 <= value <= OURS_COUNT_MAX else None
        else:
            body = _strip_boxes(span)
            pred = _parse_count(body)
            if pred is None:
                pred = _spelled_count(body)
        return pred, (pred is not None and pred == item["count"])
    # A leading letter (< number of options) wins, else the first option whose normalized text equals the span.
    k = parse_mcq_index(span, item["options"], len(item["options"]))
    pred = None if k is None else LETTERS[k].upper()
    return pred, (k is not None and k == item["gold_index"])
