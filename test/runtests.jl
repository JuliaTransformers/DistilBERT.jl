using Test
using DistilBERT
using Flux
using GPUArraysCore
using Aqua

# Loading CUDA (when available) must load the package extension cleanly.
# This guards against method-overwriting regressions in ext/.
const HAS_CUDA = try
    @eval using CUDA
    true
catch err
    @warn "CUDA.jl could not be loaded; CUDA-specific tests will be skipped." exception = err
    false
end

"""Minimal `AbstractGPUArray` stand-in to test backend-agnostic device dispatch."""
struct FakeGPUArray{T,N,A<:Array{T,N}} <: AbstractGPUArray{T,N}
    data::A
end
Base.size(f::FakeGPUArray) = size(f.data)
Base.similar(f::FakeGPUArray, ::Type{T}, dims::Dims) where {T} =
    FakeGPUArray(similar(f.data, T, dims))
function Base.copyto!(f::FakeGPUArray, src::AbstractArray)
    copyto!(f.data, src)
    return f
end
# Disambiguate against GPUArrays' `copyto!(::AnyGPUArray, ::Array)`
function Base.copyto!(f::FakeGPUArray, src::Array)
    copyto!(f.data, src)
    return f
end

"""Build a HuggingFace-shaped state dict (batch-first weights) for load_weights! tests."""
function make_base_state(config::DistilBertConfig; prefix::String="distilbert.")
    D, H = config.dim, config.hidden_dim
    sd = Dict{String,Any}()
    sd["$(prefix)embeddings.word_embeddings.weight"] =
        randn(Float32, config.vocab_size, D)
    sd["$(prefix)embeddings.position_embeddings.weight"] =
        randn(Float32, config.max_position_embeddings, D)
    sd["$(prefix)embeddings.LayerNorm.weight"] = ones(Float32, D)
    sd["$(prefix)embeddings.LayerNorm.bias"] = zeros(Float32, D)
    for i in 0:(config.n_layers - 1)
        p = "$(prefix)transformer.layer.$i"
        for name in ("attention.q_lin", "attention.k_lin", "attention.v_lin", "attention.out_lin")
            sd["$p.$name.weight"] = randn(Float32, D, D)
            sd["$p.$name.bias"] = zeros(Float32, D)
        end
        sd["$p.sa_layer_norm.weight"] = ones(Float32, D)
        sd["$p.sa_layer_norm.bias"] = zeros(Float32, D)
        sd["$p.ffn.lin1.weight"] = randn(Float32, H, D)
        sd["$p.ffn.lin1.bias"] = zeros(Float32, H)
        sd["$p.ffn.lin2.weight"] = randn(Float32, D, H)
        sd["$p.ffn.lin2.bias"] = zeros(Float32, D)
        sd["$p.output_layer_norm.weight"] = ones(Float32, D)
        sd["$p.output_layer_norm.bias"] = zeros(Float32, D)
    end
    return sd
end

@testset "DistilBERT.jl" begin
    @testset "Config" begin
        config = DistilBertConfig()
        @test config.dim == 768
        @test config.n_layers == 6
        @test config.n_heads == 12
        @test config.vocab_size == 30522

        # Float literals are converted to Float32 fields
        @test DistilBertConfig(dropout=0.1).dropout === 0.1f0
    end

    @testset "Config validation" begin
        @test DistilBertConfig(dim=64, n_heads=4) isa DistilBertConfig
        @test_throws ArgumentError DistilBertConfig(dim=10, n_heads=3)
        @test_throws ArgumentError DistilBertConfig(dim=0)
        @test_throws ArgumentError DistilBertConfig(n_layers=0)
        @test_throws ArgumentError DistilBertConfig(n_heads=0)
        @test_throws ArgumentError DistilBertConfig(hidden_dim=0)
        @test_throws ArgumentError DistilBertConfig(vocab_size=0)
        @test_throws ArgumentError DistilBertConfig(max_position_embeddings=0)
        @test_throws ArgumentError DistilBertConfig(dropout=1.0)
        @test_throws ArgumentError DistilBertConfig(dropout=-0.1)
        @test_throws ArgumentError DistilBertConfig(qa_dropout=1.0)
        @test_throws ArgumentError DistilBertConfig(seq_classif_dropout=1.0)
        @test_throws ArgumentError DistilBertConfig(initializer_range=0.0)
        @test_throws ArgumentError DistilBertConfig(layer_norm_eps=0.0)
    end

    @testset "Forward Pass Shape" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
        model = DistilBertModel(config)

        batch_size = 2
        seq_len = 10
        input_ids = rand(1:100, seq_len, batch_size)

        output = model(input_ids)

        @test size(output) == (64, seq_len, batch_size)
    end

    @testset "Embeddings Layer" begin
        config = DistilBertConfig(dim=64, n_heads=4, vocab_size=100)
        emb = DistilBERT.Embeddings(config)
        x = rand(1:100, 10, 2)
        y = emb(x)
        @test size(y) == (64, 10, 2)
    end

    @testset "Attention Layer" begin
        config = DistilBertConfig(dim=64, n_heads=4)
        attn = DistilBERT.MultiHeadSelfAttention(config)
        x = randn(Float32, 64, 10, 2)  # (dim, seq_len, batch_size)
        y = attn(x)
        @test size(y) == (64, 10, 2)
    end

    @testset "Tokenizer" begin
        # Create a minimal vocab for testing
        vocab_content = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n##ing\ntest\n"
        vocab_file = tempname()
        write(vocab_file, vocab_content)

        tokenizer = WordPieceTokenizer(vocab_file; do_lower_case=true)

        # Test basic tokenization
        tokens = tokenize(tokenizer, "hello")
        @test "hello" in tokens

        # Test encoding adds special tokens
        ids = encode(tokenizer, "hello")
        @test length(ids) >= 3  # At least [CLS], hello, [SEP]
        @test ids[1] == tokenizer.vocab["[CLS]"]
        @test ids[end] == tokenizer.vocab["[SEP]"]

        rm(vocab_file)
    end

    @testset "Pooling Strategies" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
        model = DistilBertModel(config)

        batch_size = 2
        seq_len = 10
        input_ids = rand(1:100, seq_len, batch_size)
        output = model(input_ids)

        # Create a mock attention mask
        attention_mask = ones(Float32, seq_len, batch_size)
        attention_mask[8:10, 2] .= 0.0f0  # Padding in second batch

        # CLS pooling
        cls_out = DistilBERT.cls_pooling(output)
        @test size(cls_out) == (64, batch_size)
        @test cls_out == output[:, 1, :]

        # Mean pooling
        mean_out = DistilBERT.mean_pooling(output, attention_mask)
        @test size(mean_out) == (64, batch_size)

        # Max pooling
        max_out = DistilBERT.max_pooling(output, attention_mask)
        @test size(max_out) == (64, batch_size)
    end

    @testset "Task-Specific Heads" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)

        batch_size = 2
        seq_len = 10
        input_ids = rand(1:100, seq_len, batch_size)

        # Sequence Classification
        @testset "Sequence Classification" begin
            num_labels = 3
            model = DistilBertForSequenceClassification(config, num_labels)
            logits = model(input_ids)
            @test size(logits) == (num_labels, batch_size)
        end

        # Token Classification
        @testset "Token Classification" begin
            num_labels = 5
            model = DistilBertForTokenClassification(config, num_labels)
            logits = model(input_ids)
            @test size(logits) == (num_labels, seq_len, batch_size)
        end

        # Question Answering
        @testset "Question Answering" begin
            model = DistilBertForQuestionAnswering(config)
            result = model(input_ids)
            @test haskey(result, :start_logits)
            @test haskey(result, :end_logits)
            @test size(result.start_logits) == (seq_len, batch_size)
            @test size(result.end_logits) == (seq_len, batch_size)
        end

        # Masked Language Modeling
        @testset "Masked Language Modeling" begin
            model = DistilBertForMaskedLM(config)
            logits = model(input_ids)
            @test size(logits) == (config.vocab_size, seq_len, batch_size)
        end
    end

    @testset "Type Stability" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
        model = DistilBertModel(config)
        Flux.testmode!(model)

        x = rand(1:100, 10, 2)
        mask = ones(Float32, 10, 2)

        out_no_mask = @inferred model(x)
        @test out_no_mask isa Array{Float32,3}

        out_mask = @inferred model(x; mask=mask)
        @test out_mask isa Array{Float32,3}

        seq = DistilBertForSequenceClassification(config, 3)
        tok = DistilBertForTokenClassification(config, 5)
        qa = DistilBertForQuestionAnswering(config)
        mlm = DistilBertForMaskedLM(config)
        Flux.testmode!(seq)
        Flux.testmode!(tok)
        Flux.testmode!(qa)
        Flux.testmode!(mlm)

        @test (@inferred seq(x; mask=mask)) isa Matrix{Float32}
        @test (@inferred tok(x; mask=mask)) isa Array{Float32,3}
        qa_out = @inferred qa(x; mask=mask)
        @test qa_out.start_logits isa AbstractMatrix{Float32}
        @test qa_out.end_logits isa AbstractMatrix{Float32}
        @test (@inferred mlm(x; mask=mask)) isa Array{Float32,3}
    end

    @testset "Pooling Strategies (enum)" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
        model = DistilBertModel(config)
        Flux.testmode!(model)

        dir = mktempdir()
        open(joinpath(dir, "vocab.txt"), "w") do f
            for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
                println(f, t)
            end
        end
        tokenizer = WordPieceTokenizer(dir)

        single_cls = embed(model, tokenizer, "hello world"; pooling=CLSPooling)
        single_mean = embed(model, tokenizer, "hello world"; pooling=MeanPooling)
        single_max = embed(model, tokenizer, "hello world"; pooling=MaxPooling)
        @test length(single_cls) == 64
        @test length(single_mean) == 64
        @test length(single_max) == 64

        batch = embed(model, tokenizer, ["hello", "world"]; pooling=MeanPooling)
        @test size(batch) == (64, 2)

        # Legacy Symbol API still works (deprecated)
        sym_cls = @test_deprecated embed(model, tokenizer, "hello world"; pooling=:cls)
        sym_mean = @test_deprecated embed(model, tokenizer, "hello world"; pooling=:mean)
        @test sym_cls == single_cls
        @test sym_mean == single_mean

        @test_throws ErrorException embed(model, tokenizer, "hello world"; pooling=:bogus)
    end

    @testset "Tokenizer source handling" begin
        dir = mktempdir()
        open(joinpath(dir, "vocab.txt"), "w") do f
            for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"]
                println(f, t)
            end
        end

        # Directory source resolves vocab.txt
        tok_dir = WordPieceTokenizer(dir)
        @test tokenize(tok_dir, "hello") == ["hello"]

        # Direct file source still works
        tok_file = WordPieceTokenizer(joinpath(dir, "vocab.txt"))
        @test tokenize(tok_file, "hello") == ["hello"]

        # Unknown source without TransformersDownloads errors with guidance
        if !DistilBERT._td_available()
            @test_throws ErrorException WordPieceTokenizer("definitely-not-a-real-hf-model-id")
        end

        rm(dir; recursive=true)
    end

    @testset "Tokenizer fidelity & truncation" begin
        dir = mktempdir()
        open(joinpath(dir, "vocab.txt"), "w") do f
            for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                "cafe", "naive", "test", "##ing", "hello", "world"]
                println(f, t)
            end
        end
        tok = WordPieceTokenizer(dir)

        # NFKD accent stripping matches HF BertTokenizer (do_lower_case=true)
        @test tokenize(tok, "café") == ["cafe"]
        @test tokenize(tok, "naïve") == ["naive"]
        @test tokenize(tok, "TESTING") == ["test", "##ing"]

        # do_lower_case=false preserves case and accents
        tok_cs = WordPieceTokenizer(dir; do_lower_case=false)
        @test tokenize(tok_cs, "TESTING") == ["[UNK]"]
        @test tokenize(tok_cs, "café") == ["[UNK]"]

        # Multi-byte tokens: WordPiece continuation must slice whole characters
        uni_dir = mktempdir()
        open(joinpath(uni_dir, "vocab.txt"), "w") do f
            for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "café", "##é"]
                println(f, t)
            end
        end
        tok_uni = WordPieceTokenizer(uni_dir; do_lower_case=false)
        @test tokenize(tok_uni, "café") == ["café"]
        @test tokenize(tok_uni, "caféé") == ["café", "##é"]
        rm(uni_dir; recursive=true)

        # Truncation strategies enforce max_length
        long_a = "test test test test"
        long_b = "hello world test test"
        for strat in (LongestFirst, OnlyFirst, OnlySecond)
            p = encode_pair(tok, long_a, long_b; max_length=8, truncation=strat)
            @test length(p.input_ids) == 8
            @test length(p.token_type_ids) == length(p.input_ids)
            @test length(p.attention_mask) == length(p.input_ids)
        end

        # LongestFirst can trim both sequences down to the specials only
        p0 = encode_pair(tok, long_a, "hello"; max_length=3, truncation=LongestFirst)
        @test length(p0.input_ids) == 3

        # Deprecated Symbol API matches the enum API
        psym = @test_deprecated encode_pair(tok, long_a, long_b; max_length=8, truncation=:only_second)
        penum = encode_pair(tok, long_a, long_b; max_length=8, truncation=OnlySecond)
        @test psym.input_ids == penum.input_ids
        @test_throws ErrorException encode_pair(tok, long_a, long_b; max_length=8, truncation=:bogus)

        # encode_batch validates its padding strategy
        @test_throws ArgumentError encode_batch(tok, ["hello"]; padding=:bogus)

        rm(dir; recursive=true)
    end

    @testset "Gradient Tests" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)

        @testset "DistilBertModel gradients" begin
            model = DistilBertModel(config)
            x = rand(1:100, 10, 2)

            # Use Flux.gradient which internally uses Zygote
            grads = Flux.gradient(model) do m
                sum(m(x))
            end

            @test grads[1] !== nothing
        end

        @testset "SequenceClassification gradients" begin
            model = DistilBertForSequenceClassification(config, 3)
            x = rand(1:100, 10, 2)

            grads = Flux.gradient(model) do m
                sum(m(x))
            end

            @test grads[1] !== nothing
        end

        @testset "TokenClassification gradients" begin
            model = DistilBertForTokenClassification(config, 5)
            x = rand(1:100, 10, 2)

            grads = Flux.gradient(model) do m
                sum(m(x))
            end

            @test grads[1] !== nothing
        end

        @testset "QuestionAnswering gradients" begin
            model = DistilBertForQuestionAnswering(config)
            x = rand(1:100, 10, 2)

            grads = Flux.gradient(model) do m
                r = m(x)
                sum(r.start_logits) + sum(r.end_logits)
            end

            @test grads[1] !== nothing
        end

        @testset "MaskedLM gradients (incl. tied bias)" begin
            model = DistilBertForMaskedLM(config)
            x = rand(1:100, 10, 2)

            grads = Flux.gradient(model) do m
                sum(m(x))
            end

            @test grads[1] !== nothing
            # The projection bias must receive a gradient (was excluded from trainables)
            @test grads[1].vocab_projector_bias !== nothing
        end
    end

    @testset "Precision Utilities" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
        model = DistilBertModel(config)

        @testset "to_bfloat16 preserves array type and device" begin
            m16 = DistilBERT.to_bfloat16(model)
            w = m16.embeddings.word_embeddings.weight
            @test eltype(w) === Core.BFloat16
            @test w isa Matrix{Core.BFloat16}
            # layer norm / dense parameters converted too
            @test eltype(m16.transformer[1].attention.q_lin.weight) === Core.BFloat16
        end

        @testset "to_bfloat16 idempotent" begin
            m16 = DistilBERT.to_bfloat16(model)
            m16b = DistilBERT.to_bfloat16(m16)
            @test m16b.embeddings.word_embeddings.weight === m16.embeddings.word_embeddings.weight
        end

        @testset "num_parameters counts MLM projection bias" begin
            mlm = DistilBertForMaskedLM(config)
            # vocab_projector_bias must be part of the trainable parameter tree
            @test any(p -> p === mlm.vocab_projector_bias, Flux.params(mlm))
            @test DistilBERT.num_parameters(mlm) >=
                  DistilBERT.num_parameters(mlm.distilbert) + length(mlm.vocab_projector_bias)
        end
    end

    @testset "Backend-agnostic device dispatch" begin
        ref = FakeGPUArray(zeros(Float32, 4, 4))

        ids = DistilBERT._to_device(reshape(1:5, 5, 1), ref)
        @test ids isa FakeGPUArray{Int32}
        @test ids.data == reshape(Int32.(1:5), 5, 1)

        mask = DistilBERT._to_device(ones(Float32, 5, 2), ref)
        @test mask isa FakeGPUArray{Float32}
        @test mask.data == ones(Float32, 5, 2)

        # Position ids (a UnitRange) transfer to the reference backend too
        pos = DistilBERT._to_device(1:4, ref)
        @test pos isa FakeGPUArray{Int32}
        @test pos.data == Int32.(1:4)

        # CPU fallback is the identity
        cpu_ref = zeros(Float32, 2, 2)
        x = reshape(1:3, 3, 1)
        @test DistilBERT._to_device(x, cpu_ref) === x
    end

    @testset "CUDA extension" begin
        if !HAS_CUDA
            @test_skip "CUDA.jl not installed"
        else
            # Loading CUDA must load the weakdep extension without overwrite errors
            @test Base.get_extension(DistilBERT, :DistilBERTCUDAExt) !== nothing

            if CUDA.functional()
                config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
                model = DistilBertModel(config)
                Flux.testmode!(model)

                @testset "to_gpu / on_gpu / to_cpu" begin
                    gm = DistilBERT.to_gpu(model)
                    @test DistilBERT.on_gpu(gm)
                    @test gm.embeddings.word_embeddings.weight isa CuArray
                    cm = DistilBERT.to_cpu(gm)
                    @test !DistilBERT.on_gpu(cm)
                end

                @testset "device-aware predict/embed" begin
                    vocab_file = tempname() * ".txt"
                    open(vocab_file, "w") do f
                        for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]
                            println(f, t)
                        end
                    end
                    tokenizer = WordPieceTokenizer(vocab_file)
                    gm = DistilBERT.to_gpu(model)

                    out = DistilBERT.predict(gm, tokenizer, "hello world")
                    @test out isa CuArray

                    emb = DistilBERT.embed(gm, tokenizer, "hello world"; pooling=:mean)
                    @test emb isa CuArray
                    @test length(emb) == 64

                    emb_batch = DistilBERT.embed(gm, tokenizer, ["hello", "world"]; pooling=:mean)
                    @test emb_batch isa CuArray
                    @test size(emb_batch) == (64, 2)

                    rm(vocab_file)
                end

                @testset "to_bfloat16 preserves CuArray device" begin
                    gm = DistilBERT.to_gpu(model)
                    gb = DistilBERT.to_bfloat16(gm)
                    w = gb.embeddings.word_embeddings.weight
                    @test w isa CuArray{Core.BFloat16}
                end
            else
                @test_skip "CUDA not functional on this machine"
            end
        end
    end

    @testset "Edge Cases" begin
        # Create a simple vocab for testing
        vocab_file = tempname() * ".txt"
        tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "hello", "world", "test", ".", ",", "a", "the"]
        open(vocab_file, "w") do f
            for t in tokens
                println(f, t)
            end
        end
        tokenizer = WordPieceTokenizer(vocab_file)

        @testset "Empty string" begin
            ids = encode(tokenizer, "")
            # Should at least have [CLS] and [SEP]
            @test length(ids) >= 2
            @test ids[1] == tokenizer.vocab["[CLS]"]
            @test ids[end] == tokenizer.vocab["[SEP]"]
        end

        @testset "Whitespace only" begin
            ids = encode(tokenizer, "   ")
            @test length(ids) >= 2
        end

        @testset "Long sequence truncation" begin
            long_text = join(repeat(["hello world"], 100), " ")
            ids = encode(tokenizer, long_text; max_length=20, truncation=true)
            @test length(ids) == 20
            @test ids[end] == tokenizer.vocab["[SEP]"]  # SEP preserved
        end

        @testset "Padding" begin
            ids = encode(tokenizer, "hello"; max_length=20, padding=true)
            @test length(ids) == 20
            pad_id = tokenizer.vocab["[PAD]"]
            @test ids[end] == pad_id
        end

        @testset "add_special_tokens=false" begin
            ids = encode(tokenizer, "hello"; add_special_tokens=false)
            cls_id = tokenizer.vocab["[CLS]"]
            sep_id = tokenizer.vocab["[SEP]"]
            @test !(cls_id in ids)
            @test !(sep_id in ids)
        end

        @testset "encode_pair" begin
            result = encode_pair(tokenizer, "hello", "world")
            @test haskey(result, :input_ids)
            @test haskey(result, :token_type_ids)
            @test haskey(result, :attention_mask)
            @test length(result.input_ids) == length(result.token_type_ids)
            @test length(result.input_ids) == length(result.attention_mask)
            # First tokens should be type 0, later tokens type 1
            @test result.token_type_ids[1] == 0
            @test 1 in result.token_type_ids  # type 1 should exist
        end

        @testset "All-padding batch handling" begin
            config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
            model = DistilBertModel(config)

            # Minimal input with padding
            input_ids = fill(1, 5, 2)  # PAD token ID
            mask = zeros(Float32, 5, 2)
            mask[1:2, :] .= 1.0f0  # Only first 2 tokens are real

            output = model(input_ids; mask=mask)
            @test size(output) == (64, 5, 2)
            @test !any(isnan, output)
        end

        rm(vocab_file)
    end

    @testset "API robustness" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=1, vocab_size=100)
        model = DistilBertModel(config)
        @test occursin("DistilBertModel", sprint(show, model))
        @test occursin("DistilBertConfig", sprint(show, config))

        dir = mktempdir()
        open(joinpath(dir, "vocab.txt"), "w") do f
            for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
                println(f, t)
            end
        end
        tok = WordPieceTokenizer(dir)

        # Empty batches are rejected with a clear error
        @test_throws ArgumentError encode_batch(tok, String[])

        # max_length=1 truncation leaves room for [SEP]
        ids = encode(tok, "a much longer sentence than one token";
            max_length=1, truncation=true)
        @test length(ids) == 1
        @test ids[1] == tok.vocab["[SEP]"]

        rm(dir; recursive=true)
    end

    @testset "Weight loading (synthetic state dicts)" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
        sd = make_base_state(config)

        # Base model: word embeddings are (vocab, dim) in HF layout and must be
        # transposed; Dense weights are (out, in) in both layouts.
        base = DistilBertModel(config)
        DistilBERT.load_weights!(base, sd)
        @test base.embeddings.word_embeddings.weight ==
              permutedims(sd["distilbert.embeddings.word_embeddings.weight"], (2, 1))
        @test base.embeddings.position_embeddings.weight ==
              permutedims(sd["distilbert.embeddings.position_embeddings.weight"], (2, 1))
        @test base.transformer[1].attention.q_lin.weight ==
              sd["distilbert.transformer.layer.0.attention.q_lin.weight"]
        @test base.transformer[2].ffn.lin1.weight ==
              sd["distilbert.transformer.layer.1.ffn.lin1.weight"]

        # Sequence classification head
        seq_sd = copy(sd)
        seq_sd["pre_classifier.weight"] = randn(Float32, 64, 64)
        seq_sd["pre_classifier.bias"] = zeros(Float32, 64)
        seq_sd["classifier.weight"] = randn(Float32, 3, 64)
        seq_sd["classifier.bias"] = zeros(Float32, 3)
        seq = DistilBertForSequenceClassification(config, 3)
        DistilBERT.load_weights!(seq, seq_sd)
        @test seq.classifier.weight == seq_sd["classifier.weight"]
        @test seq.pre_classifier.bias == seq_sd["pre_classifier.bias"]

        # Token classification head
        tok_sd = copy(sd)
        tok_sd["classifier.weight"] = randn(Float32, 5, 64)
        tok_sd["classifier.bias"] = zeros(Float32, 5)
        tok = DistilBertForTokenClassification(config, 5)
        DistilBERT.load_weights!(tok, tok_sd)
        @test tok.classifier.weight == tok_sd["classifier.weight"]

        # Question answering head
        qa_sd = copy(sd)
        qa_sd["qa_outputs.weight"] = randn(Float32, 2, 64)
        qa_sd["qa_outputs.bias"] = zeros(Float32, 2)
        qa = DistilBertForQuestionAnswering(config)
        DistilBERT.load_weights!(qa, qa_sd)
        @test qa.qa_outputs.weight == qa_sd["qa_outputs.weight"]

        # MaskedLM head (tied projection + bias)
        mlm_sd = copy(sd)
        mlm_sd["vocab_transform.weight"] = randn(Float32, 64, 64)
        mlm_sd["vocab_transform.bias"] = zeros(Float32, 64)
        mlm_sd["vocab_layer_norm.weight"] = ones(Float32, 64)
        mlm_sd["vocab_layer_norm.bias"] = zeros(Float32, 64)
        mlm_sd["vocab_projector.bias"] = randn(Float32, 100)
        mlm = DistilBertForMaskedLM(config)
        DistilBERT.load_weights!(mlm, mlm_sd)
        @test mlm.vocab_transform.weight == mlm_sd["vocab_transform.weight"]
        @test mlm.vocab_projector_bias == mlm_sd["vocab_projector.bias"]
    end

    @testset "load_model dispatch (init_random)" begin
        dir = mktempdir()
        open(joinpath(dir, "config.json"), "w") do f
            print(f, """{"dim": 64, "n_heads": 4, "hidden_dim": 256, "n_layers": 1, "vocab_size": 100}""")
        end

        @test load_model(dir; init_random=true) isa DistilBertModel
        @test load_model(DistilBertForSequenceClassification, dir;
            num_labels=3, init_random=true) isa DistilBertForSequenceClassification
        @test load_model(DistilBertForTokenClassification, dir;
            num_labels=5, init_random=true) isa DistilBertForTokenClassification
        @test load_model(DistilBertForQuestionAnswering, dir;
            init_random=true) isa DistilBertForQuestionAnswering
        @test load_model(DistilBertForMaskedLM, dir;
            init_random=true) isa DistilBertForMaskedLM

        # Missing config.json is a clear error (local, no TransformersDownloads)
        if !DistilBERT._td_available()
            @test_throws ErrorException load_model(mktempdir(); init_random=true)
        end

        rm(dir; recursive=true)
    end

    include("test_parity.jl")

    @testset "Aqua quality" begin
        Aqua.test_all(DistilBERT;
            # Env-dependent check: fails to locate Project.toml in some deps
            persistent_tasks=false)
    end
end
