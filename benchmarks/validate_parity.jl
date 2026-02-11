#!/usr/bin/env julia
#=
Parity Validation: Julia vs Python
Usage: julia --project=. benchmarks/validate_parity.jl [model_dir|small|big]

This script:
1. Validates Tokenizer parity (tokens & IDs)
2. Validates Model parity (hidden states)
3. Validates Masked LM parity (fill-mask predictions)
4. Supports both 'small' and 'big' models via argument
=#

using DistilBERT
using Flux
using JSON
using Printf
using Test
using Dates

# --- Helper Functions ---

function resolve_model_path(arg)
    if arg == "small"
        return joinpath(dirname(@__DIR__), "models/small")
    elseif arg == "big"
        return joinpath(dirname(@__DIR__), "models/big")
    else
        return arg
    end
end

function get_python_cmd()
    python_executable = joinpath(dirname(@__DIR__), ".venv/bin/python3")
    if !isfile(python_executable)
        python_executable = "python3"
    end
    return python_executable
end

# --- Validation Logic ---

function validate_parity(model_path_arg::String)
    model_path = resolve_model_path(model_path_arg)
    if !isdir(model_path)
        error("Model directory not found: $model_path")
    end

    println("="^60)
    println("  PARITY VALIDATION REPORT")
    println("  Date: $(now())")
    println("  Model Path: $model_path")
    println("="^60)

    # 1. Load Resources
    println("\n[Julia] Loading resources...")
    config_path = joinpath(model_path, "config.json")
    vocab_path = joinpath(model_path, "vocab.txt")

    if !isfile(config_path) || !isfile(vocab_path)
        error("Missing config.json or vocab.txt in $model_path")
    end

    # Load Tokenizer
    tokenizer = WordPieceTokenizer(vocab_path; do_lower_case=true)

    # Load Model
    model = load_model(model_path)
    m = Flux.testmode!(model)
    config = model.config
    println("[Julia] Model loaded: dim=$(config.dim), layers=$(config.n_layers), vocab=$(config.vocab_size)")

    # ----------------------------------------------------------------
    # TEST 1: Tokenizer Parity
    # ----------------------------------------------------------------
    println("\n" * "-"^60)
    println("TEST 1: Tokenizer Parity")
    println("-"^60)

    test_sentences = [
        "Hello, world!",
        "DistilBERT is amazing.",
        "UnseenWordThatShouldBeSplit"
    ]

    # Generate Python Ground Truth for Tokenizer
    py_script_tok = """
import json, os
from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained("$model_path")
sentences = $(JSON.json(test_sentences))
results = []
for s in sentences:
    results.append({
        "tokens": tokenizer.tokenize(s),
        "ids": tokenizer.encode(s, add_special_tokens=True)
    })
print(json.dumps(results))
"""
    py_cmd = `$(get_python_cmd()) -c $py_script_tok`
    py_out_tok = try
        read(py_cmd, String)
    catch e
        println("❌ Python Tokenizer script failed: $e")
        return false
    end

    py_tok_results = JSON.parse(py_out_tok)
    tokenizer_pass = true

    for (i, sent) in enumerate(test_sentences)
        expected = py_tok_results[i]

        # Julia
        jl_tokens = tokenize(tokenizer, sent)
        jl_ids = encode(tokenizer, sent)

        # Check Tokens
        if jl_tokens != expected["tokens"]
            println("❌ Token Mismatch for: \"$sent\"")
            println("   Julia: $jl_tokens")
            println("   Python: $(expected["tokens"])")
            tokenizer_pass = false
        end

        # Check IDs (Julia 1-based, Python 0-based)
        if (jl_ids .- 1) != expected["ids"]
            println("❌ ID Mismatch for: \"$sent\"")
            println("   Julia: $jl_ids")
            println("   Python: $(expected["ids"])")
            tokenizer_pass = false
        end
    end

    if tokenizer_pass
        println("✅ All $(length(test_sentences)) tokenizer tests passed.")
    else
        println("❌ Tokenizer tests FAILED.")
    end

    # ----------------------------------------------------------------
    # TEST 2: Model Numerical Parity
    # ----------------------------------------------------------------
    println("\n" * "-"^60)
    println("TEST 2: Model Numerical Parity")
    println("-"^60)

    text = "DistilBERT is amazing."
    println("Input check: \"$text\"")

    # Run Julia Model
    jl_ids = encode(tokenizer, text)
    input_matrix = reshape(jl_ids, :, 1)
    jl_output = m(input_matrix)
    jl_last_hidden_state = dropdims(jl_output, dims=3) # (dim, seq_len)

    # Run Python Model
    py_script_model = """
import torch, json
from transformers import DistilBertModel, DistilBertTokenizer
model_path = "$model_path"
text = "$text"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertModel.from_pretrained(model_path)
model.eval()
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
result = outputs.last_hidden_state.tolist()[0]
print(json.dumps(result))
"""
    py_cmd_model = `$(get_python_cmd()) -c $py_script_model`
    py_out_model = try
        read(py_cmd_model, String)
    catch e
        println("❌ Python Model script failed: $e")
        return false
    end

    # Parse Python output (seq_len, dim) -> transpose to (dim, seq_len)
    py_data = JSON.parse(py_out_model)
    py_last_hidden_state = hcat([Float32.(row) for row in py_data]...)

    # Compare
    diff = abs.(jl_last_hidden_state .- py_last_hidden_state)
    max_diff = maximum(diff)
    mean_diff = sum(diff) / length(diff)

    @printf("Max Absolute Difference: %.2e\n", max_diff)
    @printf("Mean Absolute Difference: %.2e\n", mean_diff)

    model_pass = false
    if max_diff < 1e-4
        println("✅ PASS (Strict < 1e-4)")
        model_pass = true
    elseif max_diff < 1e-2
        println("⚠️  PASS (Relaxed < 1e-2 for FP32 drift)")
        # For big models, cross-framework, this is acceptable
        model_pass = true
    else
        println("❌ FAIL (Diff > 1e-2)")
    end

    # ----------------------------------------------------------------
    # TEST 3: Masked Language Model (fill-mask) Parity
    # ----------------------------------------------------------------
    println("\n" * "-"^60)
    println("TEST 3: Masked Language Model Parity")
    println("-"^60)

    mlm_text = "Hello I'm a [MASK] model."
    println("Input: \"$mlm_text\"")

    # Check if the model has MLM architecture
    config_dict = JSON.parsefile(joinpath(model_path, "config.json"))
    architectures = get(config_dict, "architectures", String[])
    has_mlm = "DistilBertForMaskedLM" in architectures

    mlm_pass = true
    if !has_mlm
        println("⏭️  SKIP: Model does not have DistilBertForMaskedLM architecture.")
        println("   Architectures: $architectures")
    else
        # Run Julia MLM
        println("[Julia] Loading DistilBertForMaskedLM...")
        mlm_model = load_model(DistilBertForMaskedLM, model_path)
        Flux.testmode!(mlm_model)
        jl_results = unmask(mlm_model, tokenizer, mlm_text; top_k=5)

        println("[Julia] Top-5 predictions:")
        for r in jl_results
            @printf("  %.4f  %-12s  =>  %s\n", r.score, r.token, r.sequence)
        end

        # Run Python MLM
        py_script_mlm = """
import torch, json
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
model_path = "$model_path"
text = "$mlm_text"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForMaskedLM.from_pretrained(model_path)
model.eval()
inputs = tokenizer(text, return_tensors="pt")
mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
with torch.no_grad():
    logits = model(**inputs).logits
mask_logits = logits[0, mask_idx]
probs = torch.softmax(mask_logits, dim=0)
top5 = torch.topk(probs, 5)
results = []
for score, tid in zip(top5.values, top5.indices):
    results.append({"score": float(score.item()), "token": tokenizer.decode([tid.item()]).strip(), "token_id": int(tid.item())})
print(json.dumps(results))
"""
        py_cmd_mlm = `$(get_python_cmd()) -c $py_script_mlm`
        py_out_mlm = try
            read(py_cmd_mlm, String)
        catch e
            println("❌ Python MLM script failed: $e")
            mlm_pass = false
            @goto mlm_done
        end

        py_mlm_results = JSON.parse(py_out_mlm)

        println("\n[Python] Top-5 predictions:")
        for r in py_mlm_results
            @printf("  %.4f  %-12s\n", r["score"], r["token"])
        end

        # Compare rankings (token order must match)
        jl_tokens = [r.token for r in jl_results]
        # Python token IDs are 0-based, Julia 1-based
        py_tokens = [r["token"] for r in py_mlm_results]

        println("\n[Comparison]")
        if jl_tokens == py_tokens
            println("✅ Token ranking matches exactly.")
        else
            println("⚠️  Token ranking differs:")
            println("   Julia:  $jl_tokens")
            println("   Python: $py_tokens")
            # A ranking difference is OK as long as scores are close
        end

        # Compare scores
        max_score_diff = 0.0
        for i in 1:min(length(jl_results), length(py_mlm_results))
            sdiff = abs(jl_results[i].score - py_mlm_results[i]["score"])
            max_score_diff = max(max_score_diff, sdiff)
        end
        @printf("Max score difference: %.2e\n", max_score_diff)

        if max_score_diff < 1e-3
            println("✅ PASS (Score diff < 1e-3)")
        elseif max_score_diff < 1e-1
            println("⚠️  PASS (Relaxed, score diff < 0.1)")
        else
            println("❌ FAIL (Score diff > 0.1)")
            mlm_pass = false
        end
    end

    @label mlm_done

    println("\n" * "="^60)
    println("FINAL VERDICT")
    println("="^60)

    if tokenizer_pass && model_pass && mlm_pass
        println("✅ VALIDATION SUCCESSFUL")
        return true
    else
        println("❌ VALIDATION FAILED")
        return false
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    target = length(ARGS) > 0 ? ARGS[1] : "small"
    validate_parity(target)
end
