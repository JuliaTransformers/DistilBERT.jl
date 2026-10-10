module DistilBERTCUDAExt

# Package extension: CUDA-specific helpers.
#
# Core device handling is backend-agnostic (GPUArraysCore.AbstractGPUArray +
# Flux.gpu), so `to_gpu`, `to_cpu`, `on_gpu`, `predict`, and `embed` work for
# CUDA, AMDGPU, and Metal through Flux. This extension only provides helpers
# that are inherently CUDA-specific.

using DistilBERT
using CUDA
using Flux

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

end # module DistilBERTCUDAExt
