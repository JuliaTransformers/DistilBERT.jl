#!/usr/bin/env python3
"""
python_reference.py — CPU reference generator for DistilBERT parity validation.

Usage:
    python3 validation/cpu/python_reference.py <model_dir_or_hf_id> <output_dir>

Generates:
    output_dir/
        input_ids.npy          — int32 (batch, seq_len)
        attention_mask.npy     — float32 (batch, seq_len)
        hidden_states.npy      — float32 (batch, seq_len, dim)
        token_logits.npy       — float32 (batch, seq_len, vocab_size)  [DistilBertForMaskedLM]
        config.json            — echo of model config used

The Julia validation script reads these files and compares its own outputs.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Generate CPU reference outputs for DistilBERT.")
    parser.add_argument("model", help="HuggingFace model ID or local directory")
    parser.add_argument("output_dir", help="Directory to write .npy reference files")
    parser.add_argument(
        "--texts",
        nargs="+",
        default=[
            "Hello, world!",
            "DistilBERT is a great model.",
            "The quick brown fox jumps over the lazy dog.",
        ],
        help="Test sentences",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    try:
        from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForMaskedLM
    except ImportError:
        sys.exit("Install transformers: pip install transformers torch")

    print(f"[Python] Loading model: {args.model}")
    tokenizer = DistilBertTokenizer.from_pretrained(args.model)
    model     = DistilBertModel.from_pretrained(args.model).to(device).eval()
    mlm_model = DistilBertForMaskedLM.from_pretrained(args.model).to(device).eval()

    # -------------------------------------------------------------------------
    # Tokenise
    # -------------------------------------------------------------------------
    enc = tokenizer(
        args.texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)       # (batch, seq)
    attention_mask = enc["attention_mask"].to(device)  # (batch, seq)

    # -------------------------------------------------------------------------
    # Base model hidden states
    # -------------------------------------------------------------------------
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state  # (batch, seq, dim)

    # -------------------------------------------------------------------------
    # MLM logits
    # -------------------------------------------------------------------------
    with torch.no_grad():
        mlm_outputs = mlm_model(input_ids=input_ids, attention_mask=attention_mask)
    token_logits = mlm_outputs.logits  # (batch, seq, vocab_size)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    np.save(os.path.join(args.output_dir, "input_ids.npy"),
            input_ids.cpu().numpy().astype(np.int32))
    np.save(os.path.join(args.output_dir, "attention_mask.npy"),
            attention_mask.cpu().numpy().astype(np.float32))
    np.save(os.path.join(args.output_dir, "hidden_states.npy"),
            hidden_states.cpu().numpy().astype(np.float32))
    np.save(os.path.join(args.output_dir, "token_logits.npy"),
            token_logits.cpu().numpy().astype(np.float32))

    config_snapshot = {
        "model":       args.model,
        "texts":       args.texts,
        "max_length":  args.max_length,
        "device":      args.device,
        "input_shape": list(input_ids.shape),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print(f"[Python] Saved reference outputs to {args.output_dir}/")
    print(f"         input_ids:      {input_ids.shape}")
    print(f"         hidden_states:  {hidden_states.shape}")
    print(f"         token_logits:   {token_logits.shape}")


if __name__ == "__main__":
    main()
