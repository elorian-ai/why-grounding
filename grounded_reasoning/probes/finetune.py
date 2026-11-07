"""Full fine-tune of Qwen3-VL-8B-Instruct on one probe SFT json (LLaVA-style rows, loss on the assistant turn only).
1 epoch, 8 GPUs x batch 1 x grad-accum 8 (global batch 64), lr 1e-5 cosine (3% warmup), wd 0, ZeRO-3, bf16, seed 17.

  python -m torch.distributed.run --nproc_per_node 8 -m grounded_reasoning.probes.finetune \
      --data_path $GR_DATA/probes/<task>/sft/<run>.json --output_dir $GR_RUNS/probes/<task>_<run>
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, set_seed

from grounded_reasoning import DATA, REPO

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SEED = 17
DEEPSPEED = str(REPO / "configs" / "ds_zero3.json")


class SFTDataset(Dataset):
    """{"image": path under $GR_DATA, "conversations": [human "<image>\\n<prompt>", gpt "<target>"]} rows ->
    {"messages": chat turns, "image": RGB PIL}."""

    def __init__(self, data_path):
        with open(data_path) as f:
            self.records = json.load(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        convs = rec["conversations"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": convs[0]["value"].replace("<image>", "").strip()},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": convs[1]["value"]}]},
        ]
        return {"messages": messages, "image": Image.open(DATA / rec["image"]).convert("RGB")}


class SFTCollator:
    """Chat template + tokenize; labels = input_ids with padding and everything up to and including the last
    '<|im_start|>assistant\\n' set to -100. -> input_ids/attention_mask/labels [B, T], pixel_values, image_grid_thw."""

    def __init__(self, processor):
        self.processor = processor
        self.marker = torch.tensor(processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

    def __call__(self, batch):
        texts = [
            self.processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            for ex in batch
        ]
        inputs = self.processor(text=texts, images=[ex["image"] for ex in batch], padding=True, return_tensors="pt")
        ids = inputs["input_ids"]
        labels = ids.clone()
        labels[inputs["attention_mask"] == 0] = -100
        k = len(self.marker)
        for r in range(ids.size(0)):  # backward scan for the last marker, O(T * k)
            start = next(
                (s + k for s in range(ids.size(1) - k, -1, -1) if torch.equal(ids[r, s : s + k], self.marker)), None
            )
            if start is None:
                labels[r] = -100
            else:
                labels[r, :start] = -100
        inputs["labels"] = labels
        return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    set_seed(SEED)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    try:
        import flash_attn  # noqa: F401

        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn)
    model.config.use_cache = False
    model.enable_input_require_grads()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_ratio=0.03,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=DEEPSPEED,
        ddp_timeout=10800,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["tensorboard"] + (["wandb"] if os.environ.get("WANDB_API_KEY") else []),
        run_name=os.environ.get("WANDB_NAME") or Path(args.output_dir).name,
        seed=SEED,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=SFTDataset(args.data_path), data_collator=SFTCollator(processor)
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
