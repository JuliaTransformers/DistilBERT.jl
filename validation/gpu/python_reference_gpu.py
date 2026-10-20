#!/usr/bin/env python3
"""
python_reference_gpu.py — GPU reference generator for DistilBERT parity validation.

Usage:
    python3 validation/gpu/python_reference_gpu.py <model_dir_or_hf_id> <output_dir> [--dtype fp16|bf16|fp32]

Generates the same files as the CPU reference, but running on CUDA device 0.
Saves an extra `dtype.txt` so the Julia validator knows the Python precision.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output_dir")
    parser.add_argument("--texts", nargs="+", default=[
        "Hello, world!",
        "DistilBERT is a great model.",
        "The quick brown fox jumps over the lazy dog.",
    ])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available. Run on a machine with a GPU.")

    device = torch.device("cuda:0")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForMaskedLM
    except ImportError:
        sys.exit("Install transformers: pip install transformers torch")

    print(f"[Python GPU] Loading model: {args.model}  dtype={args.dtype}")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model)
    model     = DistilBertModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    mlm_model = DistilBertForMaskedLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    enc = tokenizer(
        args.texts, padding=True, truncation=True,
        max_length=args.max_length, return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        if dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=dtype):
                hs  = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                lgs = mlm_model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            hs  = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            lgs = mlm_model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Always save as float32 so Julia can load without dtype-specific NPZ support
    np.save(os.path.join(args.output_dir, "input_ids.npy"),
            input_ids.cpu().numpy().astype(np.int32))
    np.save(os.path.join(args.output_dir, "attention_mask.npy"),
            attention_mask.cpu().numpy().astype(np.float32))
    np.save(os.path.join(args.output_dir, "hidden_states.npy"),
            hs.cpu().float().numpy())
    np.save(os.path.join(args.output_dir, "token_logits.npy"),
            lgs.cpu().float().numpy())

    meta = {
        "model": args.model, "texts": args.texts,
        "max_length": args.max_length, "device": "cuda",
        "dtype": args.dtype, "input_shape": list(input_ids.shape),
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.output_dir, "dtype.txt"), "w") as f:
        f.write(args.dtype)

    print(f"[Python GPU] Saved to {args.output_dir}/")
    print(f"             GPU: {torch.cuda.get_device_name(0)}")
    print(f"             dtype: {args.dtype}")


if __name__ == "__main__":
    main()
