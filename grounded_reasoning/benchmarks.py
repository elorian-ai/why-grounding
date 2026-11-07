"""The 11 evaluation benchmarks: item lists (items/<bench>.jsonl), prompt, answer parsing and scoring (the lmms-eval
hooks of tasks/*.yaml), and the preparation of the item lists and images from the downloads of
scripts/download_benchmarks.sh. Prompt = question (+ lettered options when they are text) + SUFFIX; only the LAST
\\boxed{...} counts, as an option letter, the exact option text or an integer; unparsed responses are wrong.
    python -m grounded_reasoning.benchmarks [bench ...]        # prepare; default: all 11
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import io
import json
import os
import random
import re
import shutil
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from grounded_reasoning import DATA

ITEMS = Path(__file__).resolve().parent / "items"
IMAGE_ROOT = DATA.resolve()

LETTERS = "abcdefghijklmnopqrstuvwxyz"  # TreeBench has an 11-option item; parsing checks k < n_options
SUFFIX = " Think step by step inside <think> </think> tags, then give your final answer in \\boxed{} format."
# ERQA questions end with this instruction; it is removed because it contradicts SUFFIX.
ERQA_DIRECT_INSTR = "Please answer directly with only the letter of the correct option and nothing else."

# The question text of these is not redistributable: items/<bench>.jsonl keeps only STUB_KEYS, and main() restores
# the full rows from the download into $GR_DATA/benchmarks/<bench>/items.jsonl.
RESTORED = ("mmerw", "omnispatial")
STUB_KEYS = ("bench", "idx", "images", "gold_index")


def items_path(bench: str) -> Path:
    """Full item list of one bench: committed in items/, or restored under $GR_DATA for the RESTORED benches."""
    if bench not in RESTORED:
        return ITEMS / f"{bench}.jsonl"
    path = IMAGE_ROOT / "benchmarks" / bench / "items.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `python -m grounded_reasoning.benchmarks {bench}` first")
    return path


def load_items(bench: str) -> list[dict]:
    """Full item list -> list of item dicts, in file order (the lmms-eval doc_id is the position)."""
    with open(items_path(bench)) as f:
        return [json.loads(line) for line in f]


def load_restored_items(stub):
    """lmms-eval `process_docs` hook of the RESTORED benches: the committed stub rows -> the restored full rows
    (same rows in the same order, checked)."""
    import datasets

    full = datasets.load_dataset("json", data_files={"test": str(items_path(stub[0]["bench"]))}, split="test")
    if full.select_columns(list(STUB_KEYS)).to_list() != stub.to_list():
        raise ValueError(
            f"{items_path(stub[0]['bench'])} does not match the committed item list; "
            "rerun `python -m grounded_reasoning.benchmarks`"
        )
    return full


def _options_block(doc) -> str:
    opts = doc["options"]
    if doc.get("hide_options"):  # TreeBench OCR rows: the options are rendered in the image (official protocol)
        return ""
    if not opts or not any(opts):  # count bench, or options embedded in the image / question
        return ""
    return "\n" + "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(opts))


def _question_text(doc) -> str:
    bench = doc["bench"]
    if doc["count"] >= 0:
        return doc["question"].rstrip() + SUFFIX
    if bench == "multihopspatial":  # options already embedded as <choice>(a) ...</choice>
        return doc["question"] + SUFFIX
    if bench == "erqa":  # options embedded in the question
        return doc["question"].replace(ERQA_DIRECT_INSTR, "").rstrip() + SUFFIX
    return doc["question"] + _options_block(doc) + SUFFIX


def _open(path: str) -> Image.Image:
    return Image.open(IMAGE_ROOT / path).convert("RGB")


def doc_to_visual(doc) -> list[Image.Image]:
    return [_open(p) for p in doc["images"]]


def doc_to_text(doc) -> str:
    return _question_text(doc)


def doc_to_messages(doc) -> list:
    """Chat-format request. Images precede the text, except ERQA which keeps its
    official interleave (visual_indices are character offsets into the question)."""
    content: list[dict[str, Any]] = []
    if doc["bench"] == "erqa" and doc["visual_indices"] and any(v != 0 for v in doc["visual_indices"]):
        q = doc["question"].replace(ERQA_DIRECT_INSTR, "").rstrip()
        pairs = sorted(zip(range(len(doc["images"])), doc["visual_indices"]), key=lambda x: x[1])
        last = 0
        for j, off in pairs:
            if off == 0:
                content.append({"type": "image", "url": _open(doc["images"][j])})
            elif off <= len(q):
                if q[last:off]:
                    content.append({"type": "text", "text": q[last:off]})
                content.append({"type": "image", "url": _open(doc["images"][j])})
                last = off
            else:
                content.append({"type": "image", "url": _open(doc["images"][j])})
        tail = q[last:] if last < len(q) else ""
        content.append({"type": "text", "text": tail + SUFFIX})
    else:
        for p in doc["images"]:
            content.append({"type": "image", "url": _open(p)})
        content.append({"type": "text", "text": _question_text(doc)})
    return [{"role": "user", "content": content}]


def doc_to_target(doc) -> str:
    return str(doc["count"]) if doc["count"] >= 0 else (doc["gold_text"] or LETTERS[doc["gold_index"]])


_TEXT_WRAP = re.compile(r"\\(?:text|mathrm|mathbf|textbf)\s*\{([^{}]*)\}")
_LETTER = re.compile(r"^\(?\s*([a-zA-Z])\s*[\)\.:]?(?:\s+.*)?$", re.S)
_INT = re.compile(r"-?\d[\d,]*")


def extract_last_boxed(text: str) -> str | None:
    """Content of the LAST \\boxed{...} with balanced braces; None if absent or unterminated."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    buf = []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf)
        buf.append(c)
        i += 1
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 /]", "", s.lower()).strip()


def parse_mcq_index(boxed: str | None, options: list[str], n_options: int) -> int | None:
    """Boxed content -> 0-based option index (letter first, then exact normalised option text), or None."""
    if boxed is None:
        return None
    c = _TEXT_WRAP.sub(r"\1", boxed).strip().strip("*").strip()
    m = _LETTER.match(c)
    if m:
        k = LETTERS.find(m.group(1).lower())
        if 0 <= k < n_options:
            return k
    if options and any(options):
        t = _norm(c)
        for k, o in enumerate(options):
            if o and _norm(o) == t:
                return k
    return None


def parse_count(boxed: str | None, max_plausible: int = 100000) -> int | None:
    """Boxed content -> first integer in it (commas allowed), or None if absent / out of [0, max_plausible]."""
    if boxed is None:
        return None
    c = _TEXT_WRAP.sub(r"\1", boxed)
    m = _INT.search(c)
    if not m:
        return None
    v = int(m.group(0).replace(",", ""))
    return v if 0 <= v <= max_plausible else None


def _close(pred: int, gt: int) -> bool:
    return abs(pred - gt) <= 1 + int(0.05 * gt)


def process_results(doc, results) -> dict[str, Any]:
    """lmms-eval hook: one response -> per-item metrics; `acc_parsed` carries the full scorer record."""
    resp = results[0] if results else ""
    if not isinstance(resp, str):
        resp = getattr(resp, "text", str(resp))
    boxed = extract_last_boxed(resp)
    rec: dict[str, Any] = {
        "idx": doc["idx"],
        "bench": doc["bench"],
        "category": doc["category"],
        "boxed": boxed,
        "resp_chars": len(resp),
    }
    if doc["count"] >= 0:
        pred = parse_count(boxed)
        rec.update(
            pred=pred,
            gold=doc["count"],
            parsed=pred is not None,
            hit=pred == doc["count"],
            close=pred is not None and _close(pred, doc["count"]),
        )
    else:
        pred = parse_mcq_index(boxed, doc["options"], doc["n_options"])
        rec.update(
            pred=pred, gold=doc["gold_index"], parsed=pred is not None, hit=pred == doc["gold_index"], close=None
        )
    out = {"acc_raw": float(rec["hit"]), "parse_rate": float(rec["parsed"]), "acc_parsed": rec}
    if rec["close"] is not None:
        out["close_raw"] = float(rec["close"])
    return out


def agg_acc_parsed(records: list[dict]) -> float:
    """`acc_parsed` aggregate: accuracy over parsed rows only."""
    parsed = [r for r in records if r["parsed"]]
    return sum(r["hit"] for r in parsed) / len(parsed) if parsed else 0.0


# Preparation: one function per bench, its selection rule over the download in $GR_DATA/benchmarks/<bench>/ -> rows
# with keys bench idx images question options gold_index gold_text count n_options visual_indices category
# [hide_options]; it also writes the images the rows point at.

SRC = DATA / "benchmarks"
MAXSIDE = 2048  # ERQA / MME-RW images are downscaled to this longest side


def row(
    bench, idx, images, question, options=(), gold_index=-1, gold_text="", count=-1, visual_indices=(), category=""
):
    return {
        "bench": bench,
        "idx": int(idx),
        "images": [str(p) for p in images],
        "question": question,
        "options": list(options),
        "gold_index": int(gold_index),
        "gold_text": gold_text,
        "count": int(count),
        "n_options": len(options) if options else 0,
        "visual_indices": [int(v) for v in visual_indices],
        "category": category or "",
    }


def text_mcq(bench, idx, image, question, options, answer, category):
    """MCQ whose gold is an option TEXT (the first option equal to it)."""
    return row(bench, idx, [image], question, options, options.index(answer), answer, category=category)


def downscale(im):
    w, h = im.size
    s = MAXSIDE / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))))


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def spatialmqa():
    """SpatialMQA test: every row; idx = line number."""
    rows = []
    for i, r in enumerate(jsonl(SRC / "spatialmqa" / "test.jsonl")):
        img = f"benchmarks/spatialmqa/images/{r['image']}"
        rows.append(text_mcq("spatialmqa", i, img, r["question"], r["options"], r["answer"], ""))
    return rows


def multihopspatial():
    """MultihopSpatial test: the 4,498 rows with a parseable <choice> block and gold letter, then a seed-17 random
    subset of 1,500, sorted back into idx order. The question keeps its <choice> block."""
    choice_re = re.compile(r"<choice>(.*?)</choice>", re.S)
    opt_re = re.compile(r"\(([a-h])\)\s*(.*?)(?=\s*,?\s*\([a-h]\)|$)", re.S)
    items = []
    for i, r in enumerate(json.loads((SRC / "multihopspatial/data/multihop_test_4500.json").read_text())):
        m = choice_re.search(r["question"])
        if not m:
            continue
        found = opt_re.findall(m.group(1))
        letters = [letter for letter, _ in found]
        opts = [txt.strip().rstrip(",").strip() for _, txt in found]
        gm = re.match(r"\s*\(?([a-h])\)?", r["answer"].strip())
        gl = gm.group(1) if gm else None
        img = f"benchmarks/multihopspatial/data/images/{r['image_path']}"
        if len(opts) < 2 or gl not in letters or not (DATA / img).exists():
            continue
        gi = letters.index(gl)
        cat = f"{r.get('hop', '')}/{r.get('view', '')}"
        items.append(row("multihopspatial", i, [img], r["question"], opts, gi, opts[gi], category=cat))
    random.Random(17).shuffle(items)
    return sorted(items[:1500], key=lambda x: x["idx"])


def embspatial():
    """EmbSpatial-Bench: every row; images decoded from the json (base64) to jpg."""
    d = SRC / "embspatial"
    (d / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(json.loads((d / "embspatial_bench.json").read_text())):
        fn = f"{r.get('question_id') or f'row_{i}'}.jpg"
        img = Image.open(io.BytesIO(base64.b64decode(r["image"]))).convert("RGB")
        img.save(d / "images" / fn, quality=95)
        opts = r["answer_options"]
        q = r["question"].strip()
        rows.append(
            text_mcq(
                "embspatial",
                i,
                f"benchmarks/embspatial/images/{fn}",
                q,
                opts,
                opts[int(r["answer"])],
                r.get("relation"),
            )
        )
    return rows


def vsr():
    """VSR zero-shot test: every caption as a True/False MCQ; the COCO 2017 images are fetched from their links."""
    d = SRC / "vsr"
    (d / "images").mkdir(parents=True, exist_ok=True)
    src = jsonl(d / "test.jsonl")

    def fetch(r):
        if not (d / "images" / r["image"]).exists():
            urllib.request.urlretrieve(r["image_link"], d / "images" / r["image"])

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch, src))
    rows = []
    for i, r in enumerate(src):
        q = f'Is the following statement about the image true or false? "{r["caption"]}"'
        answer = "True" if int(r["label"]) == 1 else "False"
        rows.append(
            text_mcq("vsr", i, f"benchmarks/vsr/images/{r['image']}", q, ["True", "False"], answer, r.get("relation"))
        )
    return rows


def vstar():
    """V*Bench: the "(A) ..." lines become the options, the answer-format instruction is dropped; images are copied
    to one flat directory."""
    d = SRC / "vstar"
    (d / "images").mkdir(parents=True, exist_ok=True)
    opt_re = re.compile(r"^\(([A-F])\)\s*(.*)$")
    instr = "Answer with the option's letter from the given choices directly."
    rows = []
    for i, r in enumerate(jsonl(d / "test_questions.jsonl")):
        q_lines, opts = [], []
        for line in r["text"].splitlines():
            m = opt_re.match(line.strip())
            if m:
                opts.append(m.group(2).strip())
            elif line.strip() != instr:
                q_lines.append(line)
        flat = r["image"].replace("/", "_")
        shutil.copyfile(d / r["image"], d / "images" / flat)
        q, answer = "\n".join(q_lines).strip(), opts["ABCDEF".index(r["label"])]
        rows.append(text_mcq("vstar", i, f"benchmarks/vstar/images/{flat}", q, opts, answer, r["category"]))
    return rows


def mmerw():
    """MME-RealWorld-Lite: every row; "(A) " prefixes stripped from the options; images capped at 2048 px."""
    import pyarrow.parquet as pq

    d = SRC / "mmerw"
    (d / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(4):
        for r in pq.ParquetFile(d / f"data/train-0000{k}-of-00004.parquet").read().to_pylist():
            idx = int(r["index"])
            im = Image.open(io.BytesIO(base64.b64decode(r["bytes"]))).convert("RGB")
            (downscale(im) if max(im.size) > MAXSIDE else im).save(d / f"images/{idx}.jpg", quality=95)
            opts = [re.sub(r"^\(([A-Za-z])\)\s*", "", o).strip() for o in r["multi-choice options"]]
            answer = opts[ord(r["answer"].strip().upper()) - ord("A")]
            rows.append(
                text_mcq("mmerw", idx, f"benchmarks/mmerw/images/{idx}.jpg", r["question"], opts, answer, r["category"])
            )
    return rows


def omnispatial():
    """OmniSpatial test (official qwenvl_eval.py layout): the gold is an option index; the 229 Complex_Logic rows
    have their options drawn in the image and get 4 "" placeholders so letter answers still parse."""
    d = SRC / "omnispatial"
    if not (d / "images").exists():
        os.symlink(os.path.join("unzipped", "OmniSpatial-test"), d / "images")  # relative: GR_DATA can move
    rows = []
    for i, r in enumerate(json.loads((d / "unzipped/OmniSpatial-test/data.json").read_text())):
        img = f"benchmarks/omnispatial/images/{r['task_type']}/{r['id'].split('_')[0]}.png"
        opts = r["options"] or [""] * 4
        gi = int(r["answer"])
        cat = f"{r['task_type']}/{r['sub_task_type']}"
        rows.append(row("omnispatial", i, [img], r["question"].strip(), opts, gi, opts[gi], category=cat))
    return rows


def _varint(buf, i):
    r = s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, i
        s += 7


def _fields(buf):
    """Protobuf wire format -> (field number, wire type, int or bytes)."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i : i + ln], i + ln
        else:  # 5: fixed32, 1: fixed64
            v, i = buf[i : i + (4 if wt == 5 else 8)], i + (4 if wt == 5 else 8)
        yield fn, wt, v


def _feature(buf):
    """tensorflow Feature -> list of values (BytesList or Int64List, packed or not)."""
    for fn, _, val in _fields(buf):
        if fn == 1:
            return [v for f2, _, v in _fields(val) if f2 == 1]
        if fn == 3:
            out = []
            for f2, w2, v in _fields(val):
                if f2 == 1 and w2 == 2:
                    j = 0
                    while j < len(v):
                        x, j = _varint(v, j)
                        out.append(x)
                elif f2 == 1:
                    out.append(v)
            return out
    return []


def _tfrecord_examples(path):
    """ERQA's TFRecord (uint64 len | crc | data | crc) -> {feature key: [values]} per tensorflow.Example."""
    with open(path, "rb") as f:
        while hdr := f.read(8):
            (ln,) = struct.unpack("<Q", hdr)
            f.read(4)
            rec = f.read(ln)
            f.read(4)
            feats = {}
            for fn, _, features in _fields(rec):
                if fn != 1:  # Example.features
                    continue
                for f2, _, entry in _fields(features):
                    if f2 != 1:  # Features.feature map entry
                        continue
                    key = feat = None
                    for f3, _, v in _fields(entry):
                        if f3 == 1:
                            key = v.decode("utf-8")
                        elif f3 == 2:
                            feat = _feature(v)
                    feats[key] = feat or []
            yield feats


def erqa():
    """ERQA: all 400 examples. Options live inside the question ("Choices: A. ..."), so the row gets n_options ""
    placeholders; visual_indices = character offsets where each image is interleaved into the question."""
    d = SRC / "erqa"
    (d / "images").mkdir(parents=True, exist_ok=True)
    choice_letter = re.compile(r"(?:^|\s)([A-H])[.):]\s")
    rows = []
    for idx, f in enumerate(_tfrecord_examples(d / "ERQA/data/erqa.tfrecord")):
        question = f["question"][0].decode("utf-8")
        qtype = [x.decode("utf-8") for x in f.get("question_type", [])]
        files = []
        for j, enc in enumerate(f.get("image/encoded", [])):
            fname, im = f"ex{idx:03d}_img{j}.jpg", Image.open(io.BytesIO(enc))
            if max(im.size) > MAXSIDE:
                downscale(im.convert("RGB")).save(d / "images" / fname, quality=95)
            else:
                (d / "images" / fname).write_bytes(enc)  # verbatim
            files.append(f"benchmarks/erqa/images/{fname}")
        letters = set(choice_letter.findall(question.split("Choices:")[-1]))
        gi = LETTERS.index(f["answer"][0].decode("utf-8").strip().lower())
        vis, cat = list(f.get("visual_indices", [])), qtype[0] if qtype else "Unknown"
        rows.append(row("erqa", idx, files, question, [""] * (len(letters) or 4), gi, "", -1, vis, cat))
    return rows


def _treebench_options(r):
    """Option texts of a TreeBench TSV row in A..K order: a sequential scan of the free-text block (A's marker,
    then B's after it, ...; any later stray marker cuts the text), falling back to the A..K columns (letter prefix
    stripped) when the block yields fewer than 2 options."""
    text, marks, pos = str(r["multi-choice options"] or ""), [], 0
    for letter in "ABCDEFGHIJK":
        m1 = re.search(r"(?:^|[\n\s])" + letter + r"[.)]\s*", text[pos:])
        m2 = re.search(r"(?:^|\n)\s*" + letter + r"\s+", text[pos:])
        cands = [m for m in (m1, m2) if m]
        if not cands:
            break
        m = min(cands, key=lambda x: x.start())
        marks.append((pos + m.start(), pos + m.end()))
        pos += m.end()
    out = []
    for k, (_, start) in enumerate(marks):
        opt = text[start : marks[k + 1][0] if k + 1 < len(marks) else len(text)]
        cut = re.search(r"\n\s*[A-K][.)]\s", opt)
        opt = (opt[: cut.start()] if cut else opt).strip()
        if opt:
            out.append(opt)
    if len(out) >= 2:
        return out
    cols = []
    for letter in "ABCDEFGHIJK":
        v = r.get(letter)
        v = "" if v is None else str(v).strip()
        v = "" if v.lower() == "nan" else re.sub(r"^" + letter + r"[.)\s]\s*", "", v).strip()
        if not v:
            break
        cols.append(v)
    return cols if len(cols) >= 2 else out


def treebench():
    """TreeBench, official protocol: gold = the TSV answer LETTER (row 103 lists "red" twice); the 68 Perception/OCR
    rows are shown without the options block (hide_options: the options are rendered in the image)."""
    import pandas as pd

    d = SRC / "treebench"
    (d / "images").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(d / "TreeBench.tsv", sep="\t")
    rows = []
    for i in range(len(df)):
        r = df.iloc[i].to_dict()
        img = Image.open(io.BytesIO(base64.b64decode(r["image"]))).convert("RGB")
        img.save(d / f"images/{i:05d}.jpg", quality=95)
        opts = _treebench_options(r)
        gi = LETTERS.find(str(r["answer"]).strip().lower())
        cat = str(r["category"])
        img_path = f"benchmarks/treebench/images/{i:05d}.jpg"
        item = row("treebench", i, [img_path], str(r["question"]), opts, gi, opts[gi], category=cat)
        item["hide_options"] = cat.endswith("OCR")
        rows.append(item)
    return rows


def countqa():
    """CountQA test: all 1,528 questions on the 1,001 images (every answer is an integer); idx = question position
    over the parquet shards in sorted order, images/<NNNNN>.png = the image's row position."""
    import pyarrow.parquet as pq

    (SRC / "countqa/images").mkdir(parents=True, exist_ok=True)
    rows, image_row = [], 0
    for shard in sorted(glob.glob(str(SRC / "countqa/data/test-*.parquet"))):
        for r in pq.read_table(shard).to_pylist():
            img = f"benchmarks/countqa/images/{image_row:05d}.png"
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(DATA / img)
            for q, a in zip(r["questions"], r["answers"]):
                rows.append(row("countqa", len(rows), [img], q, count=int(str(a).strip())))
            image_row += 1
    return rows


def fsc147():
    """FSC-147 test split: "Count the number of {category}." with gold = number of annotated dots."""
    snap = SRC / "fsc147"
    test = json.loads((snap / "Train_Test_Val_FSC_147.json").read_text())["test"]
    anno = json.loads((snap / "annotation_FSC147_384.json").read_text())
    lines = (snap / "ImageClasses_FSC147.txt").read_text().split("\n")
    classes = dict(line.split("\t") for line in lines if line.strip())
    return [
        row(
            "fsc147",
            i,
            [f"benchmarks/fsc147/images_384_VarV2/{f}"],
            f"Count the number of {classes[f]}.",
            count=len(anno[f]["points"]),
            category=classes[f],
        )
        for i, f in enumerate(test)
    ]


BENCHES = {
    f.__name__: f
    for f in (spatialmqa, multihopspatial, embspatial, vsr, vstar, mmerw, omnispatial, erqa, treebench, countqa, fsc147)
}


def dump(rows) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()


def main():
    """Prepare benches -> images under $GR_DATA/benchmarks, items/<bench>.jsonl, checked against items/MANIFEST.json."""
    parser = argparse.ArgumentParser(description="Write the benchmark images and item lists from the downloads.")
    parser.add_argument(
        "benches", nargs="*", metavar="bench", choices=list(BENCHES), help=f"default: {' '.join(BENCHES)}"
    )
    args = parser.parse_args()
    manifest = json.loads((ITEMS / "MANIFEST.json").read_text())
    for name in args.benches or BENCHES:
        rows = BENCHES[name]()
        data = dump(rows)
        same = hashlib.sha256(data).hexdigest() == manifest[name]["sha256"]
        if name in RESTORED:  # full list under GR_DATA (not redistributable), id/image/gold stub in the repository
            if not same:
                sys.exit(f"{name}: the restored item list differs from the paper's; check the source revision")
            (SRC / name / "items.jsonl").write_bytes(data)
            data = dump({k: r[k] for k in STUB_KEYS} for r in rows)
        (ITEMS / f"{name}.jsonl").write_bytes(data)
        print(f"{name}: {len(rows)} items,", "identical to" if same else "DIFFERS from", "the paper's list")


if __name__ == "__main__":
    main()
