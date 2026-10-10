export to_gpu, to_cpu, on_gpu, cuda_inputs, gpu_memory_info
export to_half, to_bfloat16, to_float32
export prepare_inputs, num_parameters, num_parameters_mib

# ============================================================================
# Device Management
#
# Backend-agnostic: works with any `GPUArraysCore.AbstractGPUArray` (CuArray,
# ROCArray, MtlArray, ...) without a package extension. `Flux.gpu` selects the
# active backend (see `Flux.gpu_backend!`). Backend-specific helpers
# (`cuda_inputs`, `gpu_memory_info`) live in ext/DistilBERTCUDAExt.jl.
# ============================================================================

"""
    _to_device(x, ref::AbstractArray)

Move `x` onto the same device as the reference array `ref` (typically a model
weight). For GPU references, `x` is copied into a new `Int32` (integer inputs)
or `Float32` (real inputs) array of the same backend. Inference entry points
(`predict`, `embed`) and the `Embeddings` layer use this so inputs always match
the model's device.
"""
_to_device(x, ref::AbstractArray) = x

function _to_device(x::AbstractArray{<:Integer}, ref::AbstractGPUArray)
    return _copy_to_device(ref, Int32, x)
end

function _to_device(x::AbstractArray{<:Real}, ref::AbstractGPUArray)
    return _copy_to_device(ref, Float32, x)
end

function _copy_to_device(ref::AbstractArray, ::Type{T}, x::AbstractArray) where {T}
    y = similar(ref, T, size(x))
    # Host-to-device copies need a materialised CPU source (`Array`); GPU
    # sources are copied device-to-device. This avoids scalar iteration on
    # ranges and other non-Array host containers (e.g. `1:seq_length`).
    src = (x isa Array || x isa AbstractGPUArray) ? x : Array(x)
    copyto!(y, src)
    return y
end

"""
    to_gpu(model)

Move a model to the active GPU backend (CUDA, AMDGPU, or Metal).

Load the backend package and — if it is not CUDA, Flux's default — select it
with `Flux.gpu_backend!`. Errors if no backend is active.

Only CUDA is covered by this package's test suite; other GPUArrays.jl backends
are supported by construction but untested here.

# Example
```julia
using CUDA, Flux
model = load_model("path/to/distilbert")
model_gpu = to_gpu(model)

# AMDGPU / Metal:
using AMDGPU, Flux
Flux.gpu_backend!("AMDGPU")   # takes effect after restarting Julia
model_gpu = to_gpu(model)
```
"""
function to_gpu(model)
    gpu_model = Flux.gpu(model)
    on_gpu(gpu_model) || error(
        "No active GPU backend: `to_gpu` left the model on the CPU.\n" *
        "Load a GPU backend package (CUDA.jl, AMDGPU.jl, or Metal.jl) and, if " *
        "it is not CUDA (Flux's default), select it with `Flux.gpu_backend!`.")
    return gpu_model
end

"""
    to_cpu(model)

Move a model back to CPU memory.
"""
to_cpu(model) = model |> Flux.cpu

"""
    on_gpu(model) -> Bool

Return `true` if the model's parameters currently reside on a GPU device.
Works for any `GPUArraysCore.AbstractGPUArray` backend (CUDA, AMDGPU, Metal)
without importing the backend package.
"""
function on_gpu(model)
    ps = Flux.params(model)
    isempty(ps) && return false
    return first(ps) isa AbstractGPUArray
end

"""
    cuda_inputs(input_ids, attention_mask=nothing) -> (CuArray, CuArray|Nothing)

Convert CPU integer/float arrays to GPU-ready Int32/Float32 CuArrays.

CUDA embedding kernels require `Int32` indices.
Requires `CUDA.jl` to be loaded (package extension).

# Example
```julia
using CUDA
ids, mask = cuda_inputs(input_ids, attention_mask)
output = model_gpu(ids; mask=mask)
```
"""
function cuda_inputs end

"""
    gpu_memory_info() -> NamedTuple{(:free_gib, :total_gib, :used_gib)}

Return current GPU memory statistics in GiB.
Requires `CUDA.jl` to be loaded.
"""
function gpu_memory_info end

# ============================================================================
# Precision Utilities
# ============================================================================

"""
    to_half(model)

Convert model parameters from `Float32` to `Float16`.

Enables Tensor Core acceleration on NVIDIA GPUs (Volta+).
Combine with `to_gpu` for maximum speedup.

!!! warning
    Float16 has limited dynamic range. Validate numerical outputs after converting,
    especially for longer sequences or deeper models.

# Example
```julia
using CUDA
model_f16 = to_half(to_gpu(model))
```
"""
to_half(model) = Flux.f16(model)

"""
    to_bfloat16(model)

Convert model parameters to `BFloat16` (`Core.BFloat16`, Julia ≥ 1.12).

BFloat16 has the same exponent range as Float32 (more numerically stable than Float16),
supported on CUDA Ampere (30xx / A100) and newer hardware.

The conversion is device-preserving: conversion is done with broadcasting,
so a GPU array stays on the GPU.
"""
function to_bfloat16(model)
    return Flux.fmap(_to_bfloat16, model)
end

_to_bfloat16(x::AbstractArray{Core.BFloat16}) = x
_to_bfloat16(x::AbstractArray{<:AbstractFloat}) = Core.BFloat16.(x)
_to_bfloat16(x) = x

"""
    to_float32(model)

Convert model parameters back to `Float32`.
"""
to_float32(model) = Flux.f32(model)

# ============================================================================
# Model Inspection
# ============================================================================

"""
    num_parameters(model) -> Int

Return the total number of trainable scalar parameters.

# Example
```julia
model = load_model("path/to/distilbert")
println("Parameters: \$(num_parameters(model) ÷ 1_000_000)M")
```
"""
function num_parameters(model)
    return sum(length, Flux.params(model))
end

"""
    num_parameters_mib(model) -> Float64

Return the approximate memory footprint of Float32 model parameters in MiB.
"""
function num_parameters_mib(model)
    return num_parameters(model) * 4.0 / 2^20
end

# ============================================================================
# Input Preparation (CPU-safe)
# ============================================================================

"""
    prepare_inputs(input_ids, attention_mask=nothing) -> (Matrix{Int32}, Matrix{Float32}|Nothing)

Normalize input types for CPU inference:
- `input_ids` is converted to `Int32` (matches CUDA convention, avoids type mismatch).
- `attention_mask` is converted to `Float32` if provided.

For GPU inference use `cuda_inputs` instead.

# Example
```julia
ids, mask = prepare_inputs(raw_ids, raw_mask)
output = model(ids; mask=mask)
```
"""
function prepare_inputs(
    input_ids::AbstractMatrix{<:Integer},
    attention_mask::Union{Nothing,AbstractMatrix}=nothing
)
    ids = Int32.(input_ids)
    mask = isnothing(attention_mask) ? nothing : Float32.(attention_mask)
    return ids, mask
end
