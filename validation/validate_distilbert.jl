#!/usr/bin/env julia
#=
validate_distilbert.jl — DistilBERT.jl vs HuggingFace Python (Float32, apples-to-apples)

Uses TransformersValidation.jl for:
  • Python environment management (uv venv)
  • Float32 reference generation via `reference.py`
  • Tensor comparison with configurable tolerances
  • Performance benchmark (Julia vs Python on same hardware)

Usage:
    julia --project=. validation/validate_distilbert.jl [model_dir_or_hf_id] [options]

Options (environment variables):
    VALIDATION_WITH_BENCHMARK=true|false   (default: true)
    VALIDATION_SAMPLES=N                   (default: 30)
    VALIDATION_OUT_DIR=/tmp/distilbert_ref (default: tempdir)
    VALIDATION_TEXTS="text1|||text2"       (default: 4 built-in texts)

Prerequisites:
    uv installed (https://github.com/astral-sh/uv)
    Julia env: Pkg.develop(path="../POC/TransformersValidation.jl")
=#

using Pkg
# Auto-add TransformersValidation from the POC folder if not already present
tv_path = joinpath(dirname(@__DIR__), "..", "POC", "TransformersValidation.jl")
if !haskey(Pkg.project().dependencies, "TransformersValidation") && isdir(tv_path)
    @info "Adding TransformersValidation.jl from local path..."
    Pkg.develop(path=tv_path; io=devnull)
end

using DistilBERT
using TransformersValidation
using Flux
using Printf

# ============================================================================
# Configuration
# ============================================================================
const MODELS_DIR = joinpath(@__DIR__, "..", "models")

model_arg = get(ARGS, 1, "big")
model_dir = if model_arg ∈ ("big", "small")
    joinpath(MODELS_DIR, model_arg)
else
    model_arg
end
isdir(model_dir) || error("Model directory not found: $model_dir\n" *
    "Run: julia --project=. scripts/download_model.jl")

with_benchmark = lowercase(get(ENV, "VALIDATION_WITH_BENCHMARK", "true")) == "true"
bench_samples  = parse(Int, get(ENV, "VALIDATION_SAMPLES", "30"))
out_dir        = get(ENV, "VALIDATION_OUT_DIR", mktempdir())
texts_env      = get(ENV, "VALIDATION_TEXTS", "")
test_texts     = isempty(texts_env) ? [
    "Hello, world!",
    "DistilBERT is a distilled version of BERT.",
    "The quick brown fox jumps over the lazy dog.",
    "Julia is a high-performance language for numerical computing.",
] : split(texts_env, "|||")

# ============================================================================
# Step 1: Python environment (uv-managed)
# ============================================================================
python_dir = joinpath(tv_path, "python")
println("Setting up Python environment (uv)...")
env = PythonEnvironment(python_dir)
env = ensure_python_env(env)

# ============================================================================
# Step 2: Load Julia model + tokenizer
# ============================================================================
println("\nLoading Julia model from $model_dir...")
model = load_model(model_dir)
Flux.testmode!(model)

vocab_path = joinpath(model_dir, "vocab.txt")
tokenizer  = WordPieceTokenizer(vocab_path)

cfg = model.config
println("  dim=$(cfg.dim), layers=$(cfg.n_layers), heads=$(cfg.n_heads), vocab=$(cfg.vocab_size)")
println("  Parameters: $(num_parameters(model) ÷ 1_000_000)M")

# ============================================================================
# Step 3: Build ReferenceConfig
# ============================================================================
ref_cfg = ReferenceConfig(
    model_dir;
    model_class     = "DistilBertModel",
    tokenizer_class = "DistilBertTokenizer",
    test_texts      = test_texts,
    max_length      = cfg.max_position_embeddings,
    atol            = 5f-3,     # generous for 6-layer accumulated FP32 drift
    logit_atol      = 5f-3,
)

# ============================================================================
# Step 4: Run the full validation (parity + optional benchmark)
# ============================================================================
println("\nRunning validation (Float32, apples-to-apples)...")

# We validate the base model (hidden states)
# Use run_distilbert_validation from TransformersValidation

# Step 4a: Generate Python reference via the generic reference.py script
texts_arg = join(test_texts, "|||")
mkpath(out_dir)

run_python_script(env, "reference.py",
    model_dir,
    out_dir,
    "--texts",           texts_arg,
    "--max-length",      string(ref_cfg.max_length),
    "--model-class",     "DistilBertModel",
    "--tokenizer-class", "DistilBertTokenizer",
    "--mlm-class",       "DistilBertForMaskedLM",
    "--dtype",           "float32";
    capture=false
)

println("\nLoading reference arrays from $out_dir...")
py_ids_bf    = load_numpy_array(joinpath(out_dir, "input_ids.npy"))      # (batch, seq)
py_mask_bf   = load_numpy_array(joinpath(out_dir, "attention_mask.npy")) # (batch, seq)
py_hidden_bf = load_numpy_array(joinpath(out_dir, "hidden_states.npy"))  # (batch, seq, dim)
py_logits_bf = isfile(joinpath(out_dir, "mlm_logits.npy")) ?
    load_numpy_array(joinpath(out_dir, "mlm_logits.npy")) : nothing

# Convert batch-first Python → column-major Julia
jl_ids    = permute_python_to_julia(Int32.(py_ids_bf))    # (seq, batch)
jl_mask   = permute_python_to_julia(Float32.(py_mask_bf)) # (seq, batch)
py_hidden = permute_python_to_julia(py_hidden_bf)          # (dim, seq, batch)
py_logits = isnothing(py_logits_bf) ? nothing : permute_python_to_julia(py_logits_bf)

# Step 4b: Julia inference (Float32)
println("Running Julia inference...")
jl_hidden = Float32.(model(jl_ids; mask=jl_mask))

# Step 4c: MLM model
mlm_model = load_model(DistilBertForMaskedLM, model_dir)
Flux.testmode!(mlm_model)
jl_logits = Float32.(mlm_model(jl_ids; mask=jl_mask))

# ============================================================================
# Step 5: Parity report
# ============================================================================
parities = TensorParity[]

println("\n" * "="^72)
println(" PARITY RESULTS — Float32 (Julia vs HuggingFace Python)")
println("="^72)

p1 = compare_tensors(jl_hidden, py_hidden;
    atol=ref_cfg.atol, name="base model hidden_states")
push!(parities, p1); println("  ", p1)

if !isnothing(py_logits)
    p2 = compare_tensors(jl_logits, py_logits;
        atol=ref_cfg.logit_atol, name="mlm_logits", check_top1=true)
    push!(parities, p2); println("  ", p2)
end

# Tokenizer parity
tok_py_out = run_python_inline(env, """
import json
from transformers import DistilBertTokenizer
tok = DistilBertTokenizer.from_pretrained("$(model_dir)")
texts = $(JSON.json(test_texts))
r = tok(texts, padding=True, truncation=True, max_length=$(ref_cfg.max_length), return_tensors="pt")
print(json.dumps(r["input_ids"].tolist()))
""")

py_tok_ids = JSON.parse(tok_py_out)
match_count = 0
for (i, text) in enumerate(test_texts)
    jl_ids_tok = encode(tokenizer, text;
        add_special_tokens=true, max_length=ref_cfg.max_length, truncation=true)
    # Pad to match Python output length
    py_ids_i = py_tok_ids[i]
    n = length(py_ids_i)
    jl_padded = vcat(jl_ids_tok, fill(tokenizer.vocab[tokenizer.pad_token], max(0, n - length(jl_ids_tok))))
    match_count += (jl_padded[1:min(n,length(jl_padded))] == py_ids_i[1:min(n,length(jl_padded))])
end
tok_parity = TensorParity(
    "tokenizer ids (all texts)",
    (length(test_texts),), (length(test_texts),),
    Float32(length(test_texts) - match_count),
    Float32((length(test_texts) - match_count) / length(test_texts)),
    0f0,
    match_count == length(test_texts))
push!(parities, tok_parity); println("  ", tok_parity)

all_pass = all(p.passed for p in parities)
println()
println(all_pass ? "  ✅  ALL PARITY CHECKS PASSED" : "  ❌  SOME CHECKS FAILED")

# ============================================================================
# Step 6: Performance benchmark
# ============================================================================
if with_benchmark
    println("\n" * "="^72)
    println(" PERFORMANCE BENCHMARK (Float32, same input)")
    println("="^72)

    # Julia timings
    println("\nBenchmarking Julia...")
    jl_bench = benchmark_julia_fn(() -> model(jl_ids; mask=jl_mask);
        samples=bench_samples, label="Julia (Float32, CPU)")
    println("  ", jl_bench)

    # Python timings
    println("\nBenchmarking Python...")
    py_bench = benchmark_python_script(env, "benchmark.py",
        model_dir,
        "--texts",           texts_arg,
        "--max-length",      string(ref_cfg.max_length),
        "--model-class",     "DistilBertModel",
        "--tokenizer-class", "DistilBertTokenizer",
        "--dtype",           "float32";
        samples=bench_samples, label="Python (Float32, CPU)")
    println("  ", py_bench)

    cmp = BenchmarkComparison("DistilBERT inference", jl_bench, py_bench)
    println()
    println("  ", cmp)
    println("="^72)
end

# ============================================================================
# Step 7: Summary
# ============================================================================
println()
println("Model:         $model_dir")
println("Batch size:    $(size(jl_ids, 2))")
println("Sequence len:  $(size(jl_ids, 1))")
println("Float32 only:  yes (apples-to-apples)")
println("Reference dir: $out_dir")
