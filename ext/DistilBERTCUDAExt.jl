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
# CUDA-aware High-Level Predict
# ============================================================================
# Override predict/embed with device-aware versions so callers can just pass
# `device=gpu` instead of manually converting inputs.

"""
    DistilBERT.predict(model, tokenizer, text; device=Flux.gpu)
    DistilBERT.predict(model, tokenizer, texts; device=Flux.gpu, max_length=512)

GPU-accelerated single-text or batch inference.

The model must already reside on the GPU (call `to_gpu(model)` first).
`device` is applied to the input tensors; use `Flux.gpu` (default) for CUDA.

# Example
```julia
using CUDA
model_gpu = DistilBERT.to_gpu(load_model("path/to/model"))
Flux.testmode!(model_gpu)
output = DistilBERT.predict(model_gpu, tokenizer, "Hello world!")
```
"""
function DistilBERT.predict(
    model::DistilBERT.DistilBertModel,
    tokenizer::DistilBERT.WordPieceTokenizer,
    text::String;
    device=Flux.gpu
)
    input_ids    = DistilBERT.encode(tokenizer, text)
    input_matrix = CuArray{Int32}(reshape(input_ids, :, 1))
    return model(input_matrix)
end

function DistilBERT.predict(
    model::DistilBERT.DistilBertModel,
    tokenizer::DistilBERT.WordPieceTokenizer,
    texts::Vector{String};
    device=Flux.gpu,
    max_length::Int=512
)
    input_ids, attention_mask = DistilBERT.encode_batch(tokenizer, texts; max_length)
    cu_ids, cu_mask = DistilBERT.cuda_inputs(input_ids, attention_mask)
    return model(cu_ids; mask=cu_mask)
end

"""
    DistilBERT.embed(model, tokenizer, text; device=Flux.gpu, pooling=:cls)
    DistilBERT.embed(model, tokenizer, texts; device=Flux.gpu, pooling=:cls, max_length=512)

GPU-accelerated sentence embedding.
"""
function DistilBERT.embed(
    model::DistilBERT.DistilBertModel,
    tokenizer::DistilBERT.WordPieceTokenizer,
    text::String;
    device=Flux.gpu,
    pooling::Symbol=:cls
)
    output = DistilBERT.predict(model, tokenizer, text; device)
    cpu_output = Array(output)  # bring back to CPU for pooling utilities

    if pooling == :cls
        return vec(DistilBERT.cls_pooling(cpu_output))
    elseif pooling == :mean
        mask = ones(Float32, size(cpu_output, 2), 1)
        return vec(DistilBERT.mean_pooling(cpu_output, mask))
    elseif pooling == :max
        mask = ones(Float32, size(cpu_output, 2), 1)
        return vec(DistilBERT.max_pooling(cpu_output, mask))
    else
        error("Unknown pooling: $pooling. Use :cls, :mean, or :max")
    end
end

function DistilBERT.embed(
    model::DistilBERT.DistilBertModel,
    tokenizer::DistilBERT.WordPieceTokenizer,
    texts::Vector{String};
    device=Flux.gpu,
    pooling::Symbol=:cls,
    max_length::Int=512
)
    input_ids, attention_mask = DistilBERT.encode_batch(tokenizer, texts; max_length)
    cu_ids, cu_mask = DistilBERT.cuda_inputs(input_ids, attention_mask)
    output = model(cu_ids; mask=cu_mask)
    cpu_output = Array(output)

    if pooling == :cls
        return DistilBERT.cls_pooling(cpu_output)
    elseif pooling == :mean
        return DistilBERT.mean_pooling(cpu_output, attention_mask)
    elseif pooling == :max
        return DistilBERT.max_pooling(cpu_output, attention_mask)
    else
        error("Unknown pooling: $pooling. Use :cls, :mean, or :max")
    end
end

end # module DistilBERTCUDAExt
