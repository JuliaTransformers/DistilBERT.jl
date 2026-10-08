# Validation Framework

This folder contains parity validation scripts that verify the Julia DistilBERT
implementation produces numerically equivalent outputs to HuggingFace Transformers (PyTorch).

## Structure

```
validation/
├── utils.jl                          # Shared Julia utilities (ValidationUtils module)
├── cpu/
│   ├── python_reference.py           # Generates CPU reference .npy files from Python
│   └── validate_distilbert.jl        # CPU parity validation (Julia vs Python)
└── gpu/
    ├── python_reference_gpu.py       # Generates GPU reference .npy files (CUDA)
    └── validate_distilbert_cuda.jl   # GPU parity + Float16 + throughput benchmark
```

## Prerequisites

### Python side
```bash
pip install transformers torch numpy
```

### Julia side
```julia
using Pkg
Pkg.add(["NPZ", "BenchmarkTools"])
# For GPU:
Pkg.add("CUDA")
```

## Quick Start — CPU Validation

```bash
# From the DistilBERT.jl root:

# 1. (Optional) Pre-generate Python references
python3 validation/cpu/python_reference.py models/big /tmp/distilbert_ref_cpu

# 2. Run Julia validation (auto-generates Python refs if missing)
julia --project=. validation/cpu/validate_distilbert.jl big /tmp/distilbert_ref_cpu
```

## Quick Start — GPU Validation

```bash
# 1. Generate GPU references (requires CUDA GPU + Python)
python3 validation/gpu/python_reference_gpu.py models/big /tmp/distilbert_ref_gpu --dtype fp32

# 2. Run Julia GPU validation
julia --project=. validation/gpu/validate_distilbert_cuda.jl big /tmp/distilbert_ref_gpu
```

## Tolerance Guidelines

| Scenario                      | Expected max_diff | Verdict  |
|-------------------------------|-------------------|----------|
| Small model CPU (dim=32)      | < 1e-5            | Perfect  |
| Big model CPU (dim=768, 6L)   | < 1e-2            | Pass     |
| GPU Float32                   | < 1e-2            | Pass     |
| GPU Float16                   | < 1e-1            | Expected |

The difference for large models accumulates across 6 layers due to non-associativity
of floating-point arithmetic and different BLAS implementations (OpenBLAS vs LibTorch).

## How It Works

```
Python (transformers)          Julia (DistilBERT.jl)
─────────────────────          ─────────────────────
load model weights   ────────►  load same weights
tokenize texts       ────────►  tokenize same texts
run inference        ───────┐
save hidden_states.npy      │   run inference
                             ▼
                         compare(julia_out, python_out)
                         assert max_diff < atol
```

Both sides load the same model weights from the same directory, guaranteeing
weight parity. Any remaining differences are due to floating-point arithmetic.

## Extending to New Models

1. Copy `validation/cpu/python_reference.py` and adapt the model class.
2. Copy `validation/cpu/validate_distilbert.jl` and change the `load_model` call.
3. The `ValidationUtils` module (`validation/utils.jl`) is model-agnostic and reusable.

## CI Integration

```yaml
# .github/workflows/validate.yml (excerpt)
- name: Generate Python references
  run: python3 validation/cpu/python_reference.py models/big /tmp/ref

- name: Run Julia CPU validation
  run: julia --project=. validation/cpu/validate_distilbert.jl big /tmp/ref
```
