# DistilBERT Comparison: Julia vs Python (PyTorch CPU)

**Date:** 2026-02-09
**Environment:** Linux (4 cores)
**Model:** DistilBERT Base

## Executive Summary
**Julia is the clear winner for Real-Time Inference (Single Item)**.
**Python holds the edge for Batch Throughput** at large scales.

| Scenario | Julia (OpenBLAS) | Julia (MKL + @tturbo) | Julia (Octavian) | Python (MKL/oneDNN) |
|----------|------------------|-----------------------|------------------|---------------------|
| **Seq=32, Batch=1** | **0.54** | 0.74 | ~0.60 | 3.77 |
| **Seq=128, Batch=1** | **4.81** | 5.46 | ~5.00 | 5.06 |
| **Seq=32, Batch=8** | 11.75 | **5.82** | ~7.00 | **5.06** |
| **Seq=128, Batch=8**| 54.60 | **46.61** | 53.79 | **11.60** |

## Optimization Analysis
1.  **Latency (Single Item)**: Julia's JIT and low overhead make it superior for `Batch=1`.
2.  **MKL Integration**: `MKL.jl` is the fastest CPU backend we found.
3.  **Alternative Backends**:
    - **Octavian.jl**: Slower than MKL (53ms vs 46ms) for this shape. Good pure-Julia fallback but not a performance booster here.
    - **Fused Kernels**: Naive LoopVectorization was 20x slower.
4.  **The "Python Gap"**: PyTorch likely uses **oneDNN (MKL-DNN)** fused kernels (Assembly-level optimization for `MatMul+Bias+Gelu`). Julia's `MKL.jl` wraps standard BLAS, not oneDNN.

## Recommendation
- **CPU**: Use **MKL.jl**.
- **GPU**: For high throughput (`Batch > 8`, `Seq > 128`), switch to **CUDA.jl**. CPU optimization has barely any runway left.
