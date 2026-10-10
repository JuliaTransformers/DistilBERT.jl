# DistilBERT.jl

A high-performance, pure Julia implementation of the **DistilBERT** model, built with **Flux.jl**.

This project implements the DistilBERT architecture from scratch, featuring a custom WordPiece tokenizer and compatibility with Hugging Face pre-trained weights (`.safetensors`). It has been optimized for performance, achieving near-parity with PyTorch on CPU for large models and significantly outperforming it for small-batch inference.

## 🚀 Features

- **Pure Julia**: No Python dependencies required for inference.
- **High Performance**: Optimized attention mechanism using `NNlib.dot_product_attention`.
- **Type-stable & inferable**: the forward pass is `@inferred` end-to-end.
- **Hugging Face Compatible**:
    - Loads weights directly from `model.safetensors` (SafeTensors > PyTorch `.bin`).
    - Implements `WordPieceTokenizer` matching `BertTokenizer` logic, including
      lowercasing and accent stripping.
- **Task-Specific Heads**:
    - `DistilBertModel` (Base)
    - `DistilBertForSequenceClassification`
    - `DistilBertForTokenClassification`
    - `DistilBertForQuestionAnswering`
    - `DistilBertForMaskedLM`
- **GPU support** (CUDA package extension) with `to_gpu`, `to_half`, and
  device-aware `predict`/`embed`.
- **Verified Accuracy**: Validated against Python's `transformers` library with `< 1e-4` numerical difference.

## 📦 Installation

This package is currently a local project. **Requires Julia 1.12 or later.** Clone the repository and instantiate the environment:

```bash
git clone https://github.com/StartYourStart/DistilBERT.jl.git
cd DistilBERT.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

> **Name collisions:** the package exports generic names (`encode`, `tokenize`,
> `predict`, `embed`, `unmask`). In environments that also load other NLP
> packages, qualify calls (e.g. `DistilBERT.encode(...)`) or import selectively.

## ⚡ Quick Start

### 1. Download Model Weights
You need a folder containing `config.json`, `vocab.txt`, and `model.safetensors` from a DistilBERT model (e.g., `distilbert-base-uncased`).

### 2. Basic Inference
```julia
using DistilBERT

# Load model and tokenizer
model_path = "path/to/your/model_directory"
model = load_model(model_path)
tokenizer = WordPieceTokenizer(joinpath(model_path, "vocab.txt"))

# Disable dropout for inference
Flux.testmode!(model)

# Run inference on a single sentence
text = "DistilBERT is amazing in Julia!"
output = predict(model, tokenizer, text)

println("Output shape: ", size(output))
# (dim, seq_len, 1)
```

### 3. Sentence Embeddings
Easily extract pooled embeddings for downstream tasks:

```julia
# Get [CLS] token embedding
cls_embedding = embed(model, tokenizer, text; pooling=CLSPooling)

# Get Mean-pooled embedding
mean_embedding = embed(model, tokenizer, text; pooling=MeanPooling)

println("Embedding size: ", size(cls_embedding))
# (dim,)
```

Pooling strategies are `CLSPooling` (default), `MeanPooling`, and `MaxPooling`.
Legacy `Symbol`s (`:cls`, `:mean`, `:max`) still work but are deprecated.

### 4. Batch Processing
Efficiently process multiple sentences with automatic padding and masking:

```julia
texts = [
    "Julia is fast.",
    "Machine learning is exciting.",
    "This is a longer sentence to test padding."
]

# Get batch embeddings
batch_embeddings = embed(model, tokenizer, texts; pooling=MeanPooling)

println("Batch shape: ", size(batch_embeddings))
# (dim, batch_size)
```

Return values live on the same device as the model's parameters: CPU `Array`s
for a CPU model, `CuArray`s after `to_gpu(model)`. Use `to_cpu(model)` (or
`Array(output)`) to move results back.

## 📊 Benchmarks

> [!WARNING]
> **Benchmark Variability:** Results may vary significantly depending on system load. Use this just as a general reference.

Hardware: Linux, 4 Threads. Comparison vs PyTorch (Hugging Face).

### Small Model (dim=32)
| Task | Julia (MKL) | Python | Speedup |
|------|-------------|--------|---------|
| **Tokenizer** (Single) | **0.07 ms** | 0.23 ms | **3.3x** 🚀 |
| **Inference** (Batch=1) | **0.52 ms** | 6.22 ms | **12.0x** 🚀 |
| **Inference** (Batch=8) | **5.20 ms** | 8.02 ms | **1.5x** 🚀 |

### Big Model (Standard DistilBERT)
| Task | Julia (MKL) | Python (Torch) | Speedup |
|------|-------------|----------------|---------|
| **Tokenizer** (Single) | **0.01 ms** | 0.14 ms | **14.0x** 🚀 |
| **Inference** (Batch=1) | 36.35 ms | **34.36 ms** | 1.06x (Near Parity) |
| **Inference** (Batch=8) | 211.74 ms | **188.56 ms** | 1.1x |

## ✅ Verification

To ensure the implementation matches the reference PyTorch implementation, run the parity validation script:

```bash
# One-time: instantiate the benchmark environment and hook up the local package
julia --project=benchmarks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

# Verify against a 'small' model (fast)
julia --project=benchmarks benchmarks/validate_parity.jl small

# Verify against a 'big' model
julia --project=benchmarks benchmarks/validate_parity.jl big
```

**Requirements:**
- A Python environment (`.venv`) with `torch` and `transformers` installed is required for running the comparison scripts.
- To use MKL for best performance in Julia: `using MKL` (ensure it's in the benchmark project dependencies).

## 🖥️ GPU Usage

GPU support is **backend-agnostic**: the device layer works with any
`GPUArraysCore.AbstractGPUArray`, and `to_gpu`/`predict`/`embed` follow Flux's
active backend. Load your backend and, if it is not CUDA (Flux's default),
select it:

```julia
using CUDA, DistilBERT, Flux      # CUDA (default)

model = load_model("distilbert-base-uncased")
Flux.testmode!(model)

model_gpu = to_gpu(model)            # move parameters to the GPU
println(on_gpu(model_gpu))           # true

# predict/embed move inputs to the model's device automatically
output = predict(model_gpu, tokenizer, "Hello world!")   # backend array
embedding = embed(model_gpu, tokenizer, ["hi", "there"]; pooling=MeanPooling)

model_cpu = to_cpu(model_gpu)        # move back
model_f16 = to_half(to_gpu(model))   # Float16 (Tensor Cores)
```

```julia
using AMDGPU, Flux                   # AMD GPU
Flux.gpu_backend!("AMDGPU")          # takes effect after restarting Julia
model_gpu = to_gpu(model)

using Metal, Flux                    # Apple GPU
Flux.gpu_backend!("Metal")
model_gpu = to_gpu(model)
```

`to_bfloat16` preserves the model's device (Ampere+ recommended for BF16).

> [!NOTE]
> Only CUDA is exercised by this package's test suite. AMDGPU/Metal support is
> provided by the generic `GPUArraysCore`/`Flux.gpu` path and is untested here.
> `cuda_inputs` and `gpu_memory_info` are CUDA-only helpers.

## Project Structure

- `src/DistilBERT.jl`: Main entry point and exports.
- `src/config.jl`: Validated configuration struct.
- `src/layers.jl`: Transformer layers (Attention, FFN, Embeddings).
- `src/models.jl`: Model definitions and task heads (type-stable, NTuple blocks).
- `src/loading.jl`: Weight loading logic (SafeTensors/Pickle, HF Hub via extension).
- `src/inference.jl`: `predict`, `embed`, `unmask`, pooling strategies.
- `src/performance.jl`: Device/precision utilities.
- `src/tokenizer.jl`: WordPiece tokenizer implementation.
- `ext/DistilBERTCUDAExt.jl`: CUDA device extension.
- `ext/DistilBERTTDExt.jl`: TransformersDownloads (HF Hub) extension.
- `benchmarks/`: Performance benchmarking scripts (`benchmark_julia.jl`, `benchmark_python.py`, `validate_parity.jl`).
- `validation/`: CPU/GPU parity harness against HuggingFace (requires Python).
- `REVIEW.md`: Architecture audit and production-readiness roadmap.
- `models/`: Directory structure for storing downloaded model weights (`config.json`, `vocab.txt`, `model.safetensors`).
