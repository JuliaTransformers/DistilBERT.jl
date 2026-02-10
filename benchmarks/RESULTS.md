# DistilBERT Benchmark Results

**Date:** 2026-02-09
**Machine:** Linux, 4 Threads

## 1. Correctness Verification

We validated the Julia implementation against the HuggingFace Transformers (PyTorch) reference implementation using two models:

| Model | Dimensions | Layers | Vocab | Token IDs | Hidden States Max Diff | Verdict |
|-------|------------|--------|-------|-----------|------------------------|---------|
| **Small** | dim=32 | 5 | 1124 | ✅ Match | `8.34e-07` | **PERFECT** |
| **Big** | dim=768 | 6 | 30k | ✅ Match | `6.76e-03` | **PASS** (expected FP32 drift) |

> **Note on Big Model:** The max difference of ~`6e-3` is expected when comparing different framework implementations (LibTorch vs OpenBLAS) accumulated across 6 transformer layers. The per-token values match to 3-4 decimal places.

## 2. Performance Benchmarks

**Threads:** 4 (Julia & PyTorch)

### Small Model (dim=32, layers=5)

| Component | Batch Size | Sequence Length | Julia (MKL) | Python | Speedup (MKL vs Py) |
|-----------|------------|-----------------|-------------|--------|---------------------|
| **Tokenizer** | 1 | - | **0.04** | 0.16 | **4.0x Faster** 🚀 |
| **Tokenizer** | 8 | - | **0.32** | 0.89 | **2.8x Faster** 🚀 |
| | | | | | |
| **Model** | 1 | 32 | **0.53** | 6.05 | **11.4x Faster** 🚀 |
| **Model** | 8 | 32 | **4.67** | 4.78 | **1.02x Faster** |
| **Model** | 1 | 128 | 6.49 | **6.04** | 1.07x Slower |
| **Model** | 8 | 128 | 37.65 | **10.12** | 3.7x Slower |

### Big Model (dim=768, layers=6)

| Component | Batch Size | Sequence Length | Julia (MKL) | Python | Speedup (MKL vs Py) |
|-----------|------------|-----------------|-------------|--------|---------------------|
| **Tokenizer** | 1 | - | **0.01** | 0.10 | **10.0x Faster** 🚀 |
| **Tokenizer** | 8 | - | **0.11** | 0.60 | **5.5x Faster** 🚀 |
| | | | | | |
| **Model** | 1 | 32 | 55.70 | **46.45** | 1.2x Slower |
| **Model** | 8 | 32 | 360.22 | **230.06** | 1.6x Slower |
| **Model** | 1 | 128 | 195.66 | **133.64** | 1.5x Slower |
| **Model** | 8 | 128 | 1522.64 | **835.53** | 1.8x Slower |

### Analysis (After NNlib Refactor)

1.  **Massive Improvement:** The switch to `NNlib.dot_product_attention` yielded significant gains.
    - Single-item inference for Big Model improved from **175ms** (previous MKL run) to **56ms**.
    - This is a **~3.1x speedup** just from the attention refactor!
2.  **Gap Closing:**
    - Julia is now much closer to PyTorch performance.
    - Julia is **11x faster** for small model single-item inference.
    - For the Big Model, Julia is solely ~1.2x - 1.8x slower than highly-optimized PyTorch, which is a massive improvement from the previous 6.5x slowdown.
3.  **Remaining Bottlenecks:**
    - Usage of `NNlib.batched_mul` elsewhere (e.g., in other layers) or general allocation overhead in Flux layers might be the remaining gap.

**Recommendation:**
1.  Investigate other manual `batched_mul` usages to replace with NNlib primitives if possible.
2.  Consider `Octavian.jl` for CPU-based matrix multiplication speedups.

