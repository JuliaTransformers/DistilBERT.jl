#!/usr/bin/env julia
"""
Benchmark script comparing Julia Distilbert.jl vs Python transformers.

Benchmarks:
1. Tokenizer encode (single text)
2. Tokenizer encode_batch (multiple texts)
3. Model forward pass
4. End-to-end inference
5. Batch inference
"""

using Distilbert
using BenchmarkTools
using Statistics
using Printf

# Configuration
const MODEL_PATH = joinpath(dirname(@__DIR__), "models")
const SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."
const LONG_TEXT = repeat("This is a sample sentence for benchmarking. ", 20)
const BATCH_TEXTS = [
    "Hello world!",
    "This is a test sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "DistilBERT is a smaller version of BERT.",
    "Machine learning models can be fast.",
    "Julia is a high-performance programming language.",
    "Natural language processing is fascinating.",
    "Transformers have revolutionized NLP.",
]

function run_benchmarks()
    println("="^60)
    println("       DISTILBERT.JL BENCHMARKS")
    println("="^60)
    println()

    # =========================================================================
    # Load Model and Tokenizer
    # =========================================================================
    println("Loading model from: $MODEL_PATH")

    if !isdir(MODEL_PATH)
        println("⚠️  Model not found at $MODEL_PATH")
        println("Using synthetic model for benchmarking...")
        config = DistilBertConfig(dim=768, n_heads=12, hidden_dim=3072, n_layers=6, vocab_size=30522)
        model = DistilBertModel(config)

        # Create synthetic vocab
        vocab_file = tempname() * ".txt"
        open(vocab_file, "w") do f
            println(f, "[PAD]")
            println(f, "[UNK]")
            println(f, "[CLS]")
            println(f, "[SEP]")
            println(f, "[MASK]")
            for i in 1:30517
                println(f, "token$i")
            end
        end
        tokenizer = WordPieceTokenizer(vocab_file)
        rm(vocab_file)
    else
        model = load_model(MODEL_PATH)
        vocab_path = joinpath(MODEL_PATH, "vocab.txt")
        tokenizer = WordPieceTokenizer(vocab_path)
    end
    println("✓ Model loaded\n")

    results = Dict{String,NamedTuple{(:median, :mean, :min, :allocs, :memory),Tuple{Float64,Float64,Float64,Int,Float64}}}()

    # =========================================================================
    # 1. Tokenizer Encode (Single Text)
    # =========================================================================
    println("-"^60)
    println("1. TOKENIZER ENCODE (Single Text)")
    println("-"^60)

    # Warmup
    encode(tokenizer, SAMPLE_TEXT)

    b = @benchmark encode($tokenizer, $SAMPLE_TEXT) samples = 100
    results["tokenize_single"] = (
        median=median(b.times) / 1e6,  # ms
        mean=mean(b.times) / 1e6,
        min=minimum(b.times) / 1e6,
        allocs=b.allocs,
        memory=b.memory / 1024,  # KB
    )

    @printf("  Median: %.3f ms\n", results["tokenize_single"].median)
    @printf("  Min:    %.3f ms\n", results["tokenize_single"].min)
    @printf("  Allocs: %d\n", results["tokenize_single"].allocs)
    @printf("  Memory: %.2f KB\n\n", results["tokenize_single"].memory)

    # =========================================================================
    # 2. Tokenizer Encode Batch
    # =========================================================================
    println("-"^60)
    println("2. TOKENIZER ENCODE_BATCH (8 texts)")
    println("-"^60)

    # Warmup
    encode_batch(tokenizer, BATCH_TEXTS)

    b = @benchmark encode_batch($tokenizer, $BATCH_TEXTS) samples = 100
    results["tokenize_batch"] = (
        median=median(b.times) / 1e6,
        mean=mean(b.times) / 1e6,
        min=minimum(b.times) / 1e6,
        allocs=b.allocs,
        memory=b.memory / 1024,
    )

    @printf("  Median: %.3f ms\n", results["tokenize_batch"].median)
    @printf("  Min:    %.3f ms\n", results["tokenize_batch"].min)
    @printf("  Allocs: %d\n", results["tokenize_batch"].allocs)
    @printf("  Memory: %.2f KB\n\n", results["tokenize_batch"].memory)

    # =========================================================================
    # 3. Model Forward Pass (Single)
    # =========================================================================
    println("-"^60)
    println("3. MODEL FORWARD PASS (seq_len=32, batch=1)")
    println("-"^60)

    input_ids = rand(1:30522, 32, 1)

    # Warmup
    model(input_ids)

    b = @benchmark $model($input_ids) samples = 20
    results["forward_single"] = (
        median=median(b.times) / 1e6,
        mean=mean(b.times) / 1e6,
        min=minimum(b.times) / 1e6,
        allocs=b.allocs,
        memory=b.memory / (1024^2),  # MB
    )

    @printf("  Median: %.2f ms\n", results["forward_single"].median)
    @printf("  Min:    %.2f ms\n", results["forward_single"].min)
    @printf("  Allocs: %d\n", results["forward_single"].allocs)
    @printf("  Memory: %.2f MB\n\n", results["forward_single"].memory)

    # =========================================================================
    # 4. Model Forward Pass (Batch)
    # =========================================================================
    println("-"^60)
    println("4. MODEL FORWARD PASS (seq_len=128, batch=8)")
    println("-"^60)

    input_ids_batch = rand(1:30522, 128, 8)

    # Warmup
    model(input_ids_batch)

    b = @benchmark $model($input_ids_batch) samples = 10
    results["forward_batch"] = (
        median=median(b.times) / 1e6,
        mean=mean(b.times) / 1e6,
        min=minimum(b.times) / 1e6,
        allocs=b.allocs,
        memory=b.memory / (1024^2),
    )

    @printf("  Median: %.2f ms\n", results["forward_batch"].median)
    @printf("  Min:    %.2f ms\n", results["forward_batch"].min)
    @printf("  Allocs: %d\n", results["forward_batch"].allocs)
    @printf("  Memory: %.2f MB\n\n", results["forward_batch"].memory)

    # =========================================================================
    # 5. End-to-End Inference
    # =========================================================================
    println("-"^60)
    println("5. END-TO-END INFERENCE (tokenize + forward)")
    println("-"^60)

    # Warmup
    inference(model, tokenizer, SAMPLE_TEXT)

    b = @benchmark inference($model, $tokenizer, $SAMPLE_TEXT) samples = 20
    results["e2e_single"] = (
        median=median(b.times) / 1e6,
        mean=mean(b.times) / 1e6,
        min=minimum(b.times) / 1e6,
        allocs=b.allocs,
        memory=b.memory / (1024^2),
    )

    @printf("  Median: %.2f ms\n", results["e2e_single"].median)
    @printf("  Min:    %.2f ms\n", results["e2e_single"].min)
    @printf("  Allocs: %d\n", results["e2e_single"].allocs)
    @printf("  Memory: %.2f MB\n\n", results["e2e_single"].memory)

    # =========================================================================
    # 6. Batch Inference with Embeddings
    # =========================================================================
    println("-"^60)
    println("6. BATCH EMBEDDING (8 texts, mean pooling)")
    println("-"^60)

    # Warmup
    embed(model, tokenizer, BATCH_TEXTS; pooling=:mean)

    b = @benchmark embed($model, $tokenizer, $BATCH_TEXTS; pooling=:mean) samples = 10
    results["embed_batch"] = (
        median=median(b.times) / 1e6,
        mean=mean(b.times) / 1e6,
        min=minimum(b.times) / 1e6,
        allocs=b.allocs,
        memory=b.memory / (1024^2),
    )

    @printf("  Median: %.2f ms\n", results["embed_batch"].median)
    @printf("  Min:    %.2f ms\n", results["embed_batch"].min)
    @printf("  Allocs: %d\n", results["embed_batch"].allocs)
    @printf("  Memory: %.2f MB\n\n", results["embed_batch"].memory)

    # =========================================================================
    # Summary
    # =========================================================================
    println("="^60)
    println("                    SUMMARY")
    println("="^60)
    println()
    println("| Benchmark              | Median (ms) | Allocs  | Memory    |")
    println("|------------------------|-------------|---------|-----------|")
    for (name, r) in sort(collect(results), by=x -> x[1])
        mem_str = r.memory < 1 ? @sprintf("%.2f KB", r.memory * 1024) : @sprintf("%.2f MB", r.memory)
        @printf("| %-22s | %11.2f | %7d | %9s |\n", name, r.median, r.allocs, mem_str)
    end
    println()

    return results
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
