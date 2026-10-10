module DistilBERTCUDAExt

using DistilBERT
using CUDA
using Flux

# ============================================================================
# Device Transfer
# ============================================================================

"""
    DistilBERT.to_gpu(model)

Move a DistilBERT model and all its parameters to the active CUDA device.
Raises an error if CUDA is not functional.
"""
function DistilBERT.to_gpu(model)
    CUDA.functional() || error(
        "CUDA is not available or not functional. " *
        "Ensure a CUDA-capable GPU is present and CUDA.jl is correctly installed.\n" *
        "Run `CUDA.versioninfo()` to diagnose."
    )
    return model |> Flux.gpu
end

# ============================================================================
# Input Preparation
# ============================================================================

"""
    DistilBERT.cuda_inputs(input_ids, attention_mask=nothing)

Transfer tokenized inputs to the current CUDA device as Int32/Float32.
CUDA embedding kernels require `Int32` indices (not `Int64`).

Returns `(cu_input_ids::CuArray{Int32}, cu_attention_mask::CuArray{Float32}|Nothing)`.

# Example
```julia
using CUDA
ids, mask = DistilBERT.cuda_inputs(input_ids, attention_mask)
output = model_gpu(ids; mask=mask)
```
"""
function DistilBERT.cuda_inputs(
    input_ids::AbstractMatrix{<:Integer},
    attention_mask::Union{Nothing,AbstractMatrix{<:Real}}=nothing
)
    cu_ids  = CuArray{Int32}(input_ids)
    cu_mask = isnothing(attention_mask) ? nothing : CuArray{Float32}(attention_mask)
    return cu_ids, cu_mask
end

# ============================================================================
# GPU Memory Diagnostics
# ============================================================================

"""
    DistilBERT.gpu_memory_info() -> NamedTuple

Return current CUDA device memory statistics in GiB.

Fields: `free_gib`, `total_gib`, `used_gib`.

# Example
```julia
using CUDA
info = DistilBERT.gpu_memory_info()
println("Free GPU memory: \$(round(info.free_gib, digits=2)) GiB")
```
"""
function DistilBERT.gpu_memory_info()
    free, total = CUDA.Mem.info()
    return (
        free_gib  = free  / 2^30,
        total_gib = total / 2^30,
        used_gib  = (total - free) / 2^30,
    )
end

# ============================================================================
# Device-Aware Input Transfer
# ============================================================================
# The core `predict`/`embed` API is device-agnostic: it asks `_to_device` to
# place inputs on the same device as the model's parameters. This extension
# supplies the CUDA implementation. No core method is ever overwritten.

"""
    DistilBERT._to_device(x, ref::CuArray) -> CuArray | x

Move `x` to the CUDA device when the reference array `ref` (a model weight)
lives on the GPU, otherwise return `x` unchanged. Integer arrays become
`CuArray{Int32}` (CUDA embedding kernels require `Int32` indices).
"""
function DistilBERT._to_device(x::AbstractArray{<:Integer}, ref::CuArray)
    return CuArray{Int32}(x)
end

function DistilBERT._to_device(x::AbstractArray{<:Real}, ref::CuArray)
    return CuArray{Float32}(x)
end

end # module DistilBERTCUDAExt
