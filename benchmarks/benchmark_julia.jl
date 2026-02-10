#!/usr/bin/env julia
using Flux
using Distilbert
using BenchmarkTools
using MKL
using Statistics
using Printf
using LinearAlgebra

const MODELS_DIR = joinpath(dirname(@__DIR__), "models")

function benchmark_julia(model_name::String="big")
    model_path = joinpath(MODELS_DIR, model_name)
    if !isdir(model_path)
        error("Model directory not found: $model_path")
    end

    println("="^60)
    println("       JULIA BENCHMARK (models/$model_name)")
    println("="^60)

    # Force BLAS to use all available threads
    n_threads = Threads.nthreads()
    BLAS.set_num_threads(n_threads)

    println("BLAS Implementation: $(BLAS.get_config())")
    println("BLAS Threads: $(BLAS.get_num_threads())")
    println("Julia Threads: $n_threads")
    println()

    # 1. Load Model
    model = load_model(model_path)
    m = Flux.testmode!(model)
    config = model.config
    vocab_size = config.vocab_size
    println("✓ Model loaded: dim=$(config.dim), hidden=$(config.hidden_dim), layers=$(config.n_layers), vocab=$(vocab_size)")

    # Tokenizer
    vocab_path = joinpath(model_path, "vocab.txt")
    tokenizer = WordPieceTokenizer(vocab_path)
    println("✓ Tokenizer loaded: $(length(tokenizer.vocab)) tokens\n")

    results = Dict()

    # ---------------------------------------------------------
    # PART 1: TOKENIZER BENCHMARK
    # ---------------------------------------------------------
    println("-"^60)
    println("BENCHMARK: Tokenizer")
    println("-"^60)

    s1 = "Hello, this is a test sentence for benchmarking purpose."
    batch_docs = [s1 for _ in 1:8]

    encode(tokenizer, s1) # warmup

    t_short = @benchmark encode($tokenizer, $s1) samples = 50
    t_short_ms = median(t_short.times) / 1e6
    println("  Short Text (Single):  $(round(t_short_ms, digits=3)) ms")
    results["Tokenizer (Single)"] = t_short_ms

    encode_batch(tokenizer, batch_docs) # warmup
    t_batch = @benchmark encode_batch($tokenizer, $batch_docs) samples = 50
    t_batch_ms = median(t_batch.times) / 1e6
    println("  Short Text (Batch=8): $(round(t_batch_ms, digits=3)) ms")
    results["Tokenizer (Batch=8)"] = t_batch_ms
    println()

    # ---------------------------------------------------------
    # PART 2: MODEL SCALING BENCHMARK
    # Julia input shape: (seq_len, batch_size) — column-major
    # ---------------------------------------------------------
    scenarios = [
        (seq=32, batch=1, name="Model (Seq=32, Batch=1)"),
        (seq=32, batch=8, name="Model (Seq=32, Batch=8)"),
        (seq=128, batch=1, name="Model (Seq=128, Batch=1)"),
        (seq=128, batch=8, name="Model (Seq=128, Batch=8)")
    ]

    for s in scenarios
        println("-"^60)
        println("BENCHMARK: $(s.name)")
        println("-"^60)

        # Julia: input shape is (seq_len, batch_size)
        input = rand(1:vocab_size, s.seq, s.batch)
        m(input) # Warmup

        b = @benchmark $m($input) samples = 20
        t_ms = median(b.times) / 1e6

        results[s.name] = t_ms
        @printf("  Median: %.2f ms\n\n", t_ms)
    end

    # ---------------------------------------------------------
    # PART 3: SUMMARY
    # ---------------------------------------------------------
    println("="^60)
    println("JULIA RESULTS SUMMARY ($model_name)")
    println("="^60)
    println("Model: dim=$(config.dim), hidden=$(config.hidden_dim), layers=$(config.n_layers)")
    println()
    keys_ordered = [
        "Tokenizer (Single)", "Tokenizer (Batch=8)",
        "Model (Seq=32, Batch=1)", "Model (Seq=32, Batch=8)",
        "Model (Seq=128, Batch=1)", "Model (Seq=128, Batch=8)"
    ]
    for k in keys_ordered
        if haskey(results, k)
            @printf("%-25s: %.2f ms\n", k, results[k])
        end
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    model_name = length(ARGS) > 0 ? ARGS[1] : "big"
    benchmark_julia(model_name)
end
