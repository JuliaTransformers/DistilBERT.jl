#!/usr/bin/env python3
"""
Benchmark script for Python transformers DistilBERT.

Benchmarks:
1. Tokenizer encode (single text)
2. Tokenizer encode_batch (multiple texts)
3. Model forward pass
4. End-to-end inference
5. Batch inference

Run with: python benchmark_python.py
"""

import os
import time
import numpy as np
from statistics import median, mean
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Configuration
MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."
LONG_TEXT = "This is a sample sentence for benchmarking. " * 20
BATCH_TEXTS = [
    "Hello world!",
    "This is a test sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "DistilBERT is a smaller version of BERT.",
    "Machine learning models can be fast.",
    "Julia is a high-performance programming language.",
    "Natural language processing is fascinating.",
    "Transformers have revolutionized NLP.",
]

def benchmark(func, samples=100, warmup=5):
    """Run benchmark with warmup and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        'median': median(times),
        'mean': mean(times),
        'min': min(times),
    }

def run_benchmarks():
    print("=" * 60)
    print("       PYTHON TRANSFORMERS BENCHMARKS")
    print("=" * 60)
    print()

    # =========================================================================
    # Load Model and Tokenizer
    # =========================================================================
    print(f"Loading model: {MODEL_NAME}")

    if os.path.isdir(MODEL_PATH):
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertModel.from_pretrained(MODEL_PATH)
    else:
        print(f"Local model not found, downloading {MODEL_NAME}...")
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        model = DistilBertModel.from_pretrained(MODEL_NAME)

    model.eval()  # Set to eval mode
    print("✓ Model loaded\n")

    results = {}

    # =========================================================================
    # 1. Tokenizer Encode (Single Text)
    # =========================================================================
    print("-" * 60)
    print("1. TOKENIZER ENCODE (Single Text)")
    print("-" * 60)

    r = benchmark(lambda: tokenizer.encode(SAMPLE_TEXT), samples=100)
    results["tokenize_single"] = r
    print(f"  Median: {r['median']:.3f} ms")
    print(f"  Min:    {r['min']:.3f} ms")
    print()

    # =========================================================================
    # 2. Tokenizer Encode Batch
    # =========================================================================
    print("-" * 60)
    print("2. TOKENIZER ENCODE_BATCH (8 texts)")
    print("-" * 60)

    r = benchmark(lambda: tokenizer(BATCH_TEXTS, padding=True, return_tensors="pt"), samples=100)
    results["tokenize_batch"] = r
    print(f"  Median: {r['median']:.3f} ms")
    print(f"  Min:    {r['min']:.3f} ms")
    print()

    # =========================================================================
    # 3. Model Forward Pass (Single)
    # =========================================================================
    print("-" * 60)
    print("3. MODEL FORWARD PASS (seq_len=32, batch=1)")
    print("-" * 60)

    input_ids_single = torch.randint(0, 30522, (1, 32))

    with torch.no_grad():
        r = benchmark(lambda: model(input_ids_single), samples=20)
    results["forward_single"] = r
    print(f"  Median: {r['median']:.2f} ms")
    print(f"  Min:    {r['min']:.2f} ms")
    print()

    # =========================================================================
    # 4. Model Forward Pass (Batch)
    # =========================================================================
    print("-" * 60)
    print("4. MODEL FORWARD PASS (seq_len=128, batch=8)")
    print("-" * 60)

    input_ids_batch = torch.randint(0, 30522, (8, 128))

    with torch.no_grad():
        r = benchmark(lambda: model(input_ids_batch), samples=10)
    results["forward_batch"] = r
    print(f"  Median: {r['median']:.2f} ms")
    print(f"  Min:    {r['min']:.2f} ms")
    print()

    # =========================================================================
    # 5. End-to-End Inference
    # =========================================================================
    print("-" * 60)
    print("5. END-TO-END INFERENCE (tokenize + forward)")
    print("-" * 60)

    def e2e_single():
        inputs = tokenizer(SAMPLE_TEXT, return_tensors="pt")
        with torch.no_grad():
            return model(**inputs)

    r = benchmark(e2e_single, samples=20)
    results["e2e_single"] = r
    print(f"  Median: {r['median']:.2f} ms")
    print(f"  Min:    {r['min']:.2f} ms")
    print()

    # =========================================================================
    # 6. Batch Inference with Embeddings
    # =========================================================================
    print("-" * 60)
    print("6. BATCH EMBEDDING (8 texts, mean pooling)")
    print("-" * 60)

    def embed_batch():
        inputs = tokenizer(BATCH_TEXTS, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

    r = benchmark(embed_batch, samples=10)
    results["embed_batch"] = r
    print(f"  Median: {r['median']:.2f} ms")
    print(f"  Min:    {r['min']:.2f} ms")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("                    SUMMARY")
    print("=" * 60)
    print()
    print("| Benchmark              | Median (ms) |")
    print("|------------------------|-------------|")
    for name, r in sorted(results.items()):
        print(f"| {name:<22} | {r['median']:11.2f} |")
    print()

    return results

if __name__ == "__main__":
    run_benchmarks()
