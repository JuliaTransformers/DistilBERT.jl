#!/usr/bin/env python3
"""Generate the tiny DistilBERT parity fixture used by test/test_parity.jl.

Usage (pinned environment; see validation/fixtures/README.md and requirements.lock):

    /tmp/distilbert-parity/bin/python validation/fixtures/generate.py

The fixture is committed to `test/fixtures/tiny-distilbert/` so the Julia test
suite can run a numeric parity check against HuggingFace Transformers without
Python or network access. Regenerate intentionally (and review the diff) when
upgrading torch/transformers.

The model is tiny (2 layers, dim 32, vocab ~30) and deterministic (fixed seed).
"""

import json
import os
import sys

import numpy as np
import torch
from transformers import BertTokenizer, DistilBertConfig, DistilBertForMaskedLM

SEED = 1234
MAX_LENGTH = 32

# Fixed vocabulary (order defines the ID; Julia is 1-based, Python 0-based).
VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "hello", "world", "distilbert", "is", "amazing", "test", "##ing",
    "cafe", "naive", "the", "quick", "brown", "fox", "jumps", "over",
    "lazy", "dog", ".", ",", "!", "?", "你", "好",
    "un", "##friend", "##ly",
]

# Texts whose hidden states / MLM logits are compared numerically.
NUMERIC_TEXTS = [
    "Hello, world!",
    "DistilBERT is amazing.",
    "Testing 你好 cafe naïve unfriendly",
]

# Tokenizer-only goldens (edge cases: whitespace, accents, CJK, subwords).
TOKENIZER_TEXTS = [
    "Hello, world!",
    "  DistilBERT   is amazing.  ",
    "café naïve",
    "你好世界",
    "unfriendly testing!",
]

PAIR_A = "Hello, world!"
PAIR_B = "DistilBERT is amazing."


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        repo_root, "test", "fixtures", "tiny-distilbert")
    golden_dir = os.path.join(out_dir, "golden")
    os.makedirs(golden_dir, exist_ok=True)

    torch.manual_seed(SEED)

    vocab_path = os.path.join(out_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(VOCAB) + "\n")

    config = DistilBertConfig(
        vocab_size=len(VOCAB),
        dim=32,
        n_layers=2,
        n_heads=4,
        hidden_dim=64,
        max_position_embeddings=MAX_LENGTH,
        dropout=0.1,
        attention_dropout=0.1,
        qa_dropout=0.1,
        seq_classif_dropout=0.2,
        initializer_range=0.02,
    )
    model = DistilBertForMaskedLM(config).eval()
    model.save_pretrained(out_dir, safe_serialization=True)

    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)

    # ---- numeric goldens -------------------------------------------------
    enc = tokenizer(
        NUMERIC_TEXTS,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = model.distilbert(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        ).last_hidden_state
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        ).logits

    np.save(os.path.join(golden_dir, "input_ids.npy"),
            enc["input_ids"].numpy().astype(np.int64))
    np.save(os.path.join(golden_dir, "attention_mask.npy"),
            enc["attention_mask"].numpy().astype(np.float32))
    np.save(os.path.join(golden_dir, "hidden_states.npy"),
            hidden.numpy().astype(np.float32))
    np.save(os.path.join(golden_dir, "mlm_logits.npy"),
            logits.numpy().astype(np.float32))

    # ---- tokenizer goldens ----------------------------------------------
    tok_cases = []
    for text in TOKENIZER_TEXTS:
        tok_cases.append({
            "text": text,
            "tokens": tokenizer.tokenize(text),
            "ids": tokenizer.encode(text, add_special_tokens=True),
        })

    pair = tokenizer(PAIR_A, PAIR_B, add_special_tokens=True)
    pair_case = {
        "text_a": PAIR_A,
        "text_b": PAIR_B,
        "input_ids": pair["input_ids"],
        "token_type_ids": pair["token_type_ids"],
        "attention_mask": pair["attention_mask"],
    }

    with open(os.path.join(golden_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"cases": tok_cases, "pair": pair_case}, f, indent=2, ensure_ascii=False)

    meta = {
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "vocab_size": len(VOCAB),
        "numeric_texts": NUMERIC_TEXTS,
        "tokenizer_texts": TOKENIZER_TEXTS,
        "pair": [PAIR_A, PAIR_B],
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "note": "Regenerate with validation/fixtures/generate.py; commit and review the diff.",
    }
    with open(os.path.join(golden_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[fixtures] wrote {out_dir}")
    print(f"[fixtures] hidden {tuple(hidden.shape)} logits {tuple(logits.shape)}")
    print(f"[fixtures] torch={meta['torch']} transformers={meta['transformers']}")


if __name__ == "__main__":
    main()
