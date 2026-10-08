#!/usr/bin/env julia
#=
CPU Parity Validation — DistilBERT.jl vs HuggingFace Transformers

Usage:
    julia --project=. validation/cpu/validate_distilbert.jl [model_dir] [ref_dir]

  model_dir: path to a local HuggingFace DistilBERT directory (default: "models/big")
  ref_dir:   directory with .npy reference files produced by python_reference.py
             (default: /tmp/distilbert_ref_cpu)

Workflow:
    1. (Optional) Generate Python references:
           python3 validation/cpu/python_reference.py <model_dir> /tmp/distilbert_ref_cpu
    2. Run this script — it generates its own Python references if they are missing.
    3. Compares hidden states and MLM logits within tolerance.

Requires: transformers torch (Python), NPZ.jl (Julia)
=#

using Pkg
# Add NPZ only if not already present
if !haskey(Pkg.project().dependencies, "NPZ")
    @info "Adding NPZ.jl for numpy I/O..."
    Pkg.add("NPZ"; io=devnull)
end

using DistilBERT
using Flux
using NPZ
using Test
using JSON
using Printf

include(joinpath(@__DIR__, "..", "utils.jl"))
using .ValidationUtils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const MODELS_DIR = joinpath(dirname(dirname(@__DIR__)), "models")

function resolve_model(arg)
    arg == "big"   && return joinpath(MODELS_DIR, "big")
    arg == "small" && return joinpath(MODELS_DIR, "small")
    return arg
end

model_dir = resolve_model(get(ARGS, 1, "big"))
ref_dir   = get(ARGS, 2, "/tmp/distilbert_ref_cpu")

isdir(model_dir) || error("Model directory not found: $model_dir")
mkpath(ref_dir)

# ---------------------------------------------------------------------------
# Step 1: Generate Python references if not present
# ---------------------------------------------------------------------------
ref_hidden = joinpath(ref_dir, "hidden_states.npy")

if !isfile(ref_hidden)
    @info "Reference files not found — generating via Python..."
    py = ValidationUtils.detect_python()
    script = joinpath(@__DIR__, "python_reference.py")
    run(`$py $script $model_dir $ref_dir`)
    @info "Python references saved to $ref_dir"
end

# ---------------------------------------------------------------------------
# Step 2: Load references
# ---------------------------------------------------------------------------
py_input_ids_batchfirst  = NPZ.npzread(joinpath(ref_dir, "input_ids.npy"))      # (batch, seq)
py_attention_mask_bf     = NPZ.npzread(joinpath(ref_dir, "attention_mask.npy")) # (batch, seq)
py_hidden_states_bf      = NPZ.npzread(joinpath(ref_dir, "hidden_states.npy"))  # (batch, seq, dim)
py_token_logits_bf       = NPZ.npzread(joinpath(ref_dir, "token_logits.npy"))   # (batch, seq, vocab)

ref_config = JSON.parsefile(joinpath(ref_dir, "config.json"))
test_texts = String.(ref_config["texts"])

println("="^60)
println(" CPU PARITY VALIDATION")
println(" Model:  $model_dir")
println(" Texts:  $(length(test_texts))")
println("="^60)

# ---------------------------------------------------------------------------
# Step 3: Julia conversion — Julia is column-major, Python is row-major
# Input:  (batch, seq) → transpose → (seq, batch)
# Output: Python (batch, seq, dim) → permute → (dim, seq, batch)
# ---------------------------------------------------------------------------
# input_ids and mask: transpose (batch,seq) → (seq,batch)
jl_input_ids      = permutedims(Int32.(py_input_ids_batchfirst), (2, 1))   # (seq, batch)
jl_attention_mask = permutedims(Float32.(py_attention_mask_bf),  (2, 1))   # (seq, batch)

# Python outputs: (batch, seq, dim) → (dim, seq, batch)
py_hidden = permutedims(Float32.(py_hidden_states_bf), (3, 2, 1))  # (dim, seq, batch)
py_logits = permutedims(Float32.(py_token_logits_bf),  (3, 2, 1))  # (vocab, seq, batch)

# ---------------------------------------------------------------------------
# Step 4: Load Julia models
# ---------------------------------------------------------------------------
@info "Loading base model..."
base_model = load_model(model_dir)
Flux.testmode!(base_model)

@info "Loading MLM model..."
mlm_model = load_model(DistilBertForMaskedLM, model_dir)
Flux.testmode!(mlm_model)

@info "Loading tokenizer..."
vocab_path = joinpath(model_dir, "vocab.txt")
tokenizer  = WordPieceTokenizer(vocab_path)

println("\nModel: dim=$(base_model.config.dim), layers=$(base_model.config.n_layers)")
println("Params: $(ValidationUtils.num_parameters(base_model) ÷ 1_000_000)M\n")

# ---------------------------------------------------------------------------
# Step 5: Julia inference
# ---------------------------------------------------------------------------
@info "Running Julia inference..."
jl_hidden = base_model(jl_input_ids; mask=jl_attention_mask)  # (dim, seq, batch)
jl_logits = mlm_model(jl_input_ids; mask=jl_attention_mask)   # (vocab, seq, batch)

# ---------------------------------------------------------------------------
# Step 6: Tokenizer parity (inline Python)
# ---------------------------------------------------------------------------
@testset "Tokenizer Parity" begin
    py = ValidationUtils.detect_python()
    py_tok_result = ValidationUtils.run_python_script("""
import json
from transformers import DistilBertTokenizer
tok = DistilBertTokenizer.from_pretrained("$(model_dir)")
texts = $(JSON.json(test_texts))
ids = tok(texts, padding=True, truncation=True, max_length=$(ref_config["max_length"]), return_tensors="pt")
print(json.dumps(ids["input_ids"].tolist()))
"""; python=py)

    py_ids_nested = JSON.parse(py_tok_result)  # list of lists
    for (i, text) in enumerate(test_texts)
        jl_ids = encode(tokenizer, text; max_length=ref_config["max_length"], truncation=true, padding=false)
        jl_ids_with_pad = vcat(jl_ids, fill(tokenizer.vocab[tokenizer.pad_token],
                                            length(py_ids_nested[i]) - length(jl_ids)))
        @test jl_ids_with_pad == py_ids_nested[i] broken=(length(jl_ids_with_pad) != length(py_ids_nested[i]))
    end
end

# ---------------------------------------------------------------------------
# Step 7: Hidden-state parity
# ---------------------------------------------------------------------------
@testset "Hidden State Parity" begin
    passed = compare_tensors(jl_hidden, py_hidden; atol=5f-3, label="hidden_states")
    @test passed
end

# ---------------------------------------------------------------------------
# Step 8: MLM logits parity
# ---------------------------------------------------------------------------
@testset "MLM Logits Parity" begin
    passed = compare_tensors(jl_logits, py_logits; atol=5f-3, label="mlm_logits")
    @test passed
end

# ---------------------------------------------------------------------------
# Step 9: Per-token top-1 prediction agreement
# ---------------------------------------------------------------------------
@testset "Top-1 Token Agreement" begin
    # Compare argmax over vocab dimension (dim=1 in Julia)
    jl_top1 = argmax.(eachslice(jl_logits, dims=(2, 3)))  # seq×batch matrix of CartesianIndex
    py_top1 = argmax.(eachslice(py_logits, dims=(2, 3)))

    # Flatten and compare
    jl_ids_flat = [x[1] for x in vec(jl_top1)]
    py_ids_flat = [x[1] for x in vec(py_top1)]
    agreement   = mean(jl_ids_flat .== py_ids_flat)
    @printf "Top-1 token agreement: %.1f%%\n" agreement * 100
    @test agreement >= 0.95
end

println("\n✅  CPU validation complete.")
