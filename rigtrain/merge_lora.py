"""Merge a LoRA adapter into Qwen3.5-4B and save the full model, for serving without LoRA kernels.

  .venv/bin/python merge_lora.py ADAPTER OUT_DIR

The persona pushes keep using the base model's W_down columns (vsteer.VSteered reads them from
the base checkpoint when it is given a merged model), so a merged model is pushed exactly as
the adapter on the base model is. OUT_DIR/unmerge is the same adapter with lora_B negated: the
merged model under it is the base model again, which the deciding voices run on.
"""
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

MODEL = "Qwen/Qwen3.5-4B"


def main(adapter, out):
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(out, max_shard_size="5GB")
    src = snapshot_download(MODEL, local_files_only=True)
    for f in os.listdir(src):
        if not f.startswith("model.") and not os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(src, f), out)
    un = os.path.join(out, "unmerge"); os.makedirs(un, exist_ok=True)
    w = load_file(os.path.join(adapter, "adapter_model.safetensors"))
    save_file({k: -v if ".lora_B." in k else v for k, v in w.items()}, os.path.join(un, "adapter_model.safetensors"))
    shutil.copy(os.path.join(adapter, "adapter_config.json"), un)
    print("merged", adapter, "->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
