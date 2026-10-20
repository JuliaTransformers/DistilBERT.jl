export DistilBertConfig

"""
    DistilBertConfig

Configuration for the DistilBERT model.

# Fields
- `vocab_size::Int`: Vocabulary size (default: 30522)
- `dim::Int`: Dimensionality of the encoder layers and the pooler layer (default: 768)
- `n_layers::Int`: Number of hidden layers in the Transformer encoder (default: 6)
- `n_heads::Int`: Number of attention heads for each attention layer in the Transformer encoder (default: 12)
- `hidden_dim::Int`: Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder (default: 3072)
- `dropout::Float32`: The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler (default: 0.1)
- `max_position_embeddings::Int`: The maximum sequence length that this model might ever be used with (default: 512)
- `initializer_range::Float32`: The standard deviation of the truncated_normal_initializer for initializing all weight matrices (default: 0.02)
- `qa_dropout::Float32`: Dropout probability for the QA head (default: 0.1)
- `seq_classif_dropout::Float32`: Dropout probability for the sequence classification head (default: 0.2)
"""
@kwdef struct DistilBertConfig
    vocab_size::Int = 30522
    dim::Int = 768
    n_layers::Int = 6
    n_heads::Int = 12
    hidden_dim::Int = 3072
    dropout::Float32 = 0.1f0
    max_position_embeddings::Int = 512
    initializer_range::Float32 = 0.02f0
    qa_dropout::Float32 = 0.1f0
    seq_classif_dropout::Float32 = 0.2f0
    layer_norm_eps::Float32 = 1f-12

    function DistilBertConfig(
        vocab_size::Integer,
        dim::Integer,
        n_layers::Integer,
        n_heads::Integer,
        hidden_dim::Integer,
        dropout::Real,
        max_position_embeddings::Integer,
        initializer_range::Real,
        qa_dropout::Real,
        seq_classif_dropout::Real,
        layer_norm_eps::Real,
    )
        vocab_size > 0 || throw(ArgumentError("vocab_size must be positive, got $vocab_size"))
        dim > 0 || throw(ArgumentError("dim must be positive, got $dim"))
        n_layers > 0 || throw(ArgumentError("n_layers must be positive, got $n_layers"))
        n_heads > 0 || throw(ArgumentError("n_heads must be positive, got $n_heads"))
        dim % n_heads == 0 || throw(ArgumentError(
            "dim ($dim) must be divisible by n_heads ($n_heads)"))
        hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive, got $hidden_dim"))
        max_position_embeddings > 0 || throw(ArgumentError(
            "max_position_embeddings must be positive, got $max_position_embeddings"))
        0 <= dropout < 1 || throw(ArgumentError("dropout must be in [0, 1), got $dropout"))
        0 <= qa_dropout < 1 || throw(ArgumentError("qa_dropout must be in [0, 1), got $qa_dropout"))
        0 <= seq_classif_dropout < 1 || throw(ArgumentError(
            "seq_classif_dropout must be in [0, 1), got $seq_classif_dropout"))
        initializer_range > 0 || throw(ArgumentError(
            "initializer_range must be positive, got $initializer_range"))
        layer_norm_eps > 0 || throw(ArgumentError(
            "layer_norm_eps must be positive, got $layer_norm_eps"))

        return new(
            vocab_size, dim, n_layers, n_heads, hidden_dim, dropout,
            max_position_embeddings, initializer_range, qa_dropout,
            seq_classif_dropout, layer_norm_eps,
        )
    end
end

function Base.show(io::IO, c::DistilBertConfig)
    print(io, "DistilBertConfig(dim=$(c.dim), layers=$(c.n_layers), heads=$(c.n_heads), hidden=$(c.hidden_dim), vocab=$(c.vocab_size))")
end
