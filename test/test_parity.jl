# Numeric and tokenizer parity against HuggingFace Transformers.
#
# Uses the committed tiny-model fixture in test/fixtures/tiny-distilbert, so it
# runs hermetically (no Python, no network). Regenerate fixtures with
# validation/fixtures/generate.py when upgrading torch/transformers.

using Test
using DistilBERT
using Flux
using NPZ
using JSON

const PARITY_FIXTURE_DIR = joinpath(@__DIR__, "fixtures", "tiny-distilbert")
const PARITY_GOLDEN_DIR = joinpath(PARITY_FIXTURE_DIR, "golden")

@testset "Python parity (tiny fixture)" begin
    @test isdir(PARITY_FIXTURE_DIR)

    input_ids = Int.(NPZ.npzread(joinpath(PARITY_GOLDEN_DIR, "input_ids.npy")))       # (B, S)
    mask = Float32.(NPZ.npzread(joinpath(PARITY_GOLDEN_DIR, "attention_mask.npy")))   # (B, S)
    hidden = Float32.(NPZ.npzread(joinpath(PARITY_GOLDEN_DIR, "hidden_states.npy")))  # (B, S, D)
    logits = Float32.(NPZ.npzread(joinpath(PARITY_GOLDEN_DIR, "mlm_logits.npy")))     # (B, S, V)
    tok_golden = JSON.parsefile(joinpath(PARITY_GOLDEN_DIR, "tokenizer.json"))
    meta = JSON.parsefile(joinpath(PARITY_GOLDEN_DIR, "meta.json"))

    # ---------------------------------------------------------------- models
    model = load_model(PARITY_FIXTURE_DIR)
    mlm = load_model(DistilBertForMaskedLM, PARITY_FIXTURE_DIR)
    Flux.testmode!(model)
    Flux.testmode!(mlm)

    tokenizer = WordPieceTokenizer(joinpath(PARITY_FIXTURE_DIR, "vocab.txt"))

    @testset "tokenizer goldens" begin
        for case in tok_golden["cases"]
            text = case["text"]
            @test tokenize(tokenizer, text) == case["tokens"]
            # Julia IDs are 1-based; Python IDs are 0-based.
            @test (encode(tokenizer, text) .- 1) == case["ids"]
        end

        pair = tok_golden["pair"]
        jl_pair = encode_pair(tokenizer, pair["text_a"], pair["text_b"])
        @test (jl_pair.input_ids .- 1) == pair["input_ids"]
        @test jl_pair.token_type_ids == pair["token_type_ids"]
        @test jl_pair.attention_mask == Float32.(pair["attention_mask"])
    end

    # ------------------------------------------------ numeric goldens (HF input_ids)
    # Python IDs are 0-based; Julia IDs are 1-based.
    jl_ids = permutedims(input_ids, (2, 1)) .+ 1   # (S, B)
    jl_mask = permutedims(mask, (2, 1))            # (S, B)
    py_hidden = permutedims(hidden, (3, 2, 1))    # (D, S, B)
    py_logits = permutedims(logits, (3, 2, 1))    # (V, S, B)

    jl_hidden = model(jl_ids; mask=jl_mask)
    jl_logits = mlm(jl_ids; mask=jl_mask)

    @test size(jl_hidden) == size(py_hidden)
    @test size(jl_logits) == size(py_logits)

    hidden_diff = maximum(abs.(jl_hidden .- py_hidden))
    logits_diff = maximum(abs.(jl_logits .- py_logits))
    @info "Parity max abs diff" hidden_states=hidden_diff mlm_logits=logits_diff

    @test hidden_diff < 1f-5
    @test logits_diff < 1f-5
    @test all(argmax(jl_logits, dims=1) .== argmax(py_logits, dims=1))

    # ------------------------------------------- end-to-end (Julia tokenization)
    texts = String.(meta["numeric_texts"])
    jl_ids2, jl_mask2 = encode_batch(tokenizer, texts; max_length=meta["max_length"])
    @test permutedims(jl_ids2, (2, 1)) .- 1 == input_ids
    @test permutedims(jl_mask2, (2, 1)) == mask

    jl_hidden2 = model(jl_ids2; mask=jl_mask2)
    jl_logits2 = mlm(jl_ids2; mask=jl_mask2)
    @test maximum(abs.(jl_hidden2 .- py_hidden)) < 1f-5
    @test maximum(abs.(jl_logits2 .- py_logits)) < 1f-5
end
