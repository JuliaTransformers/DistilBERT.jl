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
struct DistilBertConfig
    vocab_size::Int
    dim::Int
    n_layers::Int
    n_heads::Int
    hidden_dim::Int
    dropout::Float32
    max_position_embeddings::Int
    initializer_range::Float32
    qa_dropout::Float32
    seq_classif_dropout::Float32
    layer_norm_eps::Float32
end

function DistilBertConfig(;
    vocab_size=30522,
    dim=768,
    n_layers=6,
    n_heads=12,
    hidden_dim=3072,
    dropout=0.1f0,
    max_position_embeddings=512,
    initializer_range=0.02f0,
    qa_dropout=0.1f0,
    seq_classif_dropout=0.2f0,
    layer_norm_eps=1f-12
)
    return DistilBertConfig(
        vocab_size, dim, n_layers, n_heads, hidden_dim, dropout,
        max_position_embeddings, initializer_range, qa_dropout, seq_classif_dropout, layer_norm_eps
    )
end

function Base.show(io::IO, c::DistilBertConfig)
    print(io, "DistilBertConfig(dim=$(c.dim), layers=$(c.n_layers), heads=$(c.n_heads), hidden=$(c.hidden_dim), vocab=$(c.vocab_size))")
end
