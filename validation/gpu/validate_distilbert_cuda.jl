#!/usr/bin/env julia
#=
GPU Parity Validation — DistilBERT.jl (CUDA) vs HuggingFace Transformers (GPU)

Usage:
    julia --project=. validation/gpu/validate_distilbert_cuda.jl [model_dir] [ref_dir]

Prerequisites:
  1. CUDA-capable GPU present
  2. CUDA.jl installed: `using Pkg; Pkg.add("CUDA")`
  3. NPZ.jl installed: `using Pkg; Pkg.add("NPZ")`
  4. Python reference generated:
       python3 validation/gpu/python_reference_gpu.py <model_dir> /tmp/distilbert_ref_gpu

Steps validated:
  - Float32 GPU hidden states vs Python float32
  - Float16 GPU hidden states vs Python float16 (if GPU supports it)
  - Throughput comparison (Julia GPU vs Python GPU)
=#

using CUDA
CUDA.functional() || error("CUDA not functional. Run on a machine with a GPU.")

using Pkg
for pkg in ("NPZ",)
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg; io=devnull)
end

using DistilBERT
using Flux
using NPZ
using Test
using JSON
using Printf
using BenchmarkTools

include(joinpath(@__DIR__, "..", "utils.jl"))
using .ValidationUtils

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
const MODELS_DIR = joinpath(dirname(dirname(@__DIR__)), "models")
model_dir = let arg = get(ARGS, 1, "big")
    arg ∈ ("big","small") ? joinpath(MODELS_DIR, arg) : arg
end
ref_dir = get(ARGS, 2, "/tmp/distilbert_ref_gpu")

isdir(model_dir) || error("Model not found: $model_dir")
mkpath(ref_dir)

# ---------------------------------------------------------------------------
# Step 1: Generate Python GPU references if absent
# ---------------------------------------------------------------------------
if !isfile(joinpath(ref_dir, "hidden_states.npy"))
    @info "Reference files missing — generating via Python (GPU)..."
    py = ValidationUtils.detect_python()
    script = joinpath(@__DIR__, "python_reference_gpu.py")
    run(`$py $script $model_dir $ref_dir`)
end

# ---------------------------------------------------------------------------
# Step 2: Load references
# ---------------------------------------------------------------------------
py_input_ids_bf  = NPZ.npzread(joinpath(ref_dir, "input_ids.npy"))      # int32 (batch, seq)
py_mask_bf       = NPZ.npzread(joinpath(ref_dir, "attention_mask.npy")) # float32 (batch, seq)
py_hidden_bf     = NPZ.npzread(joinpath(ref_dir, "hidden_states.npy"))  # float32 (batch, seq, dim)
py_logits_bf     = NPZ.npzread(joinpath(ref_dir, "token_logits.npy"))   # float32 (batch, seq, vocab)
ref_config       = JSON.parsefile(joinpath(ref_dir, "config.json"))
py_dtype         = isfile(joinpath(ref_dir, "dtype.txt")) ?
                       strip(read(joinpath(ref_dir, "dtype.txt"), String)) : "fp32"

println("="^60)
println(" GPU PARITY VALIDATION")
println(" Model:      $model_dir")
println(" CUDA GPU:   $(CUDA.name(CUDA.device()))")
println(" Py dtype:   $py_dtype")
println("="^60)

# ---------------------------------------------------------------------------
# Step 3: Julia inputs — transpose to Julia column-major (seq, batch)
# ---------------------------------------------------------------------------
jl_ids_cpu  = permutedims(Int32.(py_input_ids_bf), (2, 1))   # (seq, batch)
jl_mask_cpu = permutedims(Float32.(py_mask_bf),     (2, 1))   # (seq, batch)

# Target: (dim, seq, batch)
py_hidden = permutedims(Float32.(py_hidden_bf), (3, 2, 1))
py_logits = permutedims(Float32.(py_logits_bf), (3, 2, 1))

# ---------------------------------------------------------------------------
# Step 4: Load and move Julia models to GPU
# ---------------------------------------------------------------------------
@info "Loading models and transferring to GPU..."
base_model = load_model(model_dir)
mlm_model  = load_model(DistilBertForMaskedLM, model_dir)
Flux.testmode!(base_model)
Flux.testmode!(mlm_model)

base_gpu = to_gpu(base_model)
mlm_gpu  = to_gpu(mlm_model)

let (free, total) = CUDA.Mem.info()
    @info "Models on GPU. Memory: $(round((total - free) / 2^30, digits=2)) GiB used"
end

# ---------------------------------------------------------------------------
# Step 5: Float32 GPU inference
# ---------------------------------------------------------------------------
@info "Running Float32 GPU inference..."
jl_ids_gpu  = CuArray{Int32}(jl_ids_cpu)
jl_mask_gpu = CuArray{Float32}(jl_mask_cpu)

jl_hidden_gpu = base_gpu(jl_ids_gpu; mask=jl_mask_gpu)
jl_logits_gpu = mlm_gpu(jl_ids_gpu;  mask=jl_mask_gpu)

# Bring back to CPU for comparison
jl_hidden_f32 = Array(jl_hidden_gpu)
jl_logits_f32 = Array(jl_logits_gpu)

@testset "Float32 GPU — Hidden States" begin
    atol = py_dtype == "fp32" ? 5f-3 : 2f-2
    @test compare_tensors(jl_hidden_f32, py_hidden; atol, label="hidden_states (f32)")
end

@testset "Float32 GPU — MLM Logits" begin
    atol = py_dtype == "fp32" ? 5f-3 : 2f-2
    @test compare_tensors(jl_logits_f32, py_logits; atol, label="mlm_logits (f32)")
end

# ---------------------------------------------------------------------------
# Step 6: Float16 GPU inference (if device supports it)
# ---------------------------------------------------------------------------
if CUDA.capability(CUDA.device()) >= VersionNumber(7, 0)
    @info "Tensor Cores detected — testing Float16..."
    base_f16 = to_half(base_gpu)
    mlm_f16  = to_half(mlm_gpu)
    Flux.testmode!(base_f16)
    Flux.testmode!(mlm_f16)

    # Float16 models need Float16 inputs
    jl_ids_f16  = CuArray{Int32}(jl_ids_cpu)    # int stays int
    jl_mask_f16 = CuArray{Float16}(jl_mask_cpu)

    jl_hidden_f16_gpu = base_f16(jl_ids_f16; mask=jl_mask_f16)
    jl_hidden_f16     = Float32.(Array(jl_hidden_f16_gpu))

    @testset "Float16 GPU — Hidden States" begin
        # Higher tolerance due to reduced precision
        @test compare_tensors(jl_hidden_f16, py_hidden; atol=1f-1, label="hidden_states (f16)")
    end

    mem_free, mem_total = CUDA.Mem.info()
    @printf "GPU memory after f16 run: %.2f / %.2f GiB used\n" (mem_total - mem_free) / 2^30 mem_total / 2^30
else
    @info "Skipping Float16 test: GPU compute capability < 7.0"
end

# ---------------------------------------------------------------------------
# Step 7: Throughput benchmark (GPU vs CPU)
# ---------------------------------------------------------------------------
println("\n", "-"^60)
println(" THROUGHPUT BENCHMARK  (batch=$(size(jl_ids_cpu, 2)), seq=$(size(jl_ids_cpu, 1)))")
println("-"^60)

# CPU baseline
CUDA.synchronize()
cpu_time = @belapsed begin
    $(base_model)($(jl_ids_cpu); mask=$(jl_mask_cpu))
end samples=20 evals=1 setup=(GC.gc())

# GPU (Float32)
CUDA.synchronize()
gpu_time = @belapsed begin
    CUDA.@sync $(base_gpu)($(jl_ids_gpu); mask=$(jl_mask_gpu))
end samples=20 evals=1 setup=(GC.gc())

@printf "  Julia CPU:          %8.3f ms\n" cpu_time * 1e3
@printf "  Julia GPU (f32):    %8.3f ms  (%.1fx speedup)\n" gpu_time * 1e3 cpu_time / gpu_time

if CUDA.capability(CUDA.device()) >= VersionNumber(7, 0)
    jl_ids_f16_bench  = CuArray{Int32}(jl_ids_cpu)
    jl_mask_f16_bench = CuArray{Float16}(jl_mask_cpu)
    base_f16_bench = to_half(base_gpu)
    Flux.testmode!(base_f16_bench)

    CUDA.synchronize()
    gpu_f16_time = @belapsed begin
        CUDA.@sync $(base_f16_bench)($(jl_ids_f16_bench); mask=$(jl_mask_f16_bench))
    end samples=20 evals=1 setup=(GC.gc())
    @printf "  Julia GPU (f16):    %8.3f ms  (%.1fx speedup vs CPU)\n" gpu_f16_time * 1e3 cpu_time / gpu_f16_time
end

println("\n✅  GPU validation complete.")
