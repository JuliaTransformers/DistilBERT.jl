using Distilbert
using Flux
using Test

@testset "DistilBERT Tests" begin
    @testset "Config" begin
        config = DistilBertConfig()
        @test config.dim == 768
        @test config.n_layers == 6
    end

    @testset "Forward Pass Shape" begin
        config = DistilBertConfig(dim=64, n_heads=4, hidden_dim=256, n_layers=2, vocab_size=100)
        model = DistilBertModel(config)

        batch_size = 2
        seq_len = 10
        input_ids = rand(1:100, seq_len, batch_size)

        output = model(input_ids)

        # Expected output shape: (dim, seq_len, batch_size)
        @test size(output) == (64, seq_len, batch_size)
    end

    @testset "Layers" begin
        config = DistilBertConfig(dim=64, n_heads=4)

        # Embeddings
        emb = Distilbert.Embeddings(config)
        x = rand(1:100, 10, 2)
        y = emb(x)
        @test size(y) == (64, 10, 2)

        # Attention
        attn = Distilbert.MultiHeadSelfAttention(config)
        # Input to attention is (dim, seq_len, batch_size)
        y_attn = attn(y)
        @test size(y_attn) == (64, 10, 2)
    end
end
