module Distilbert

using Flux
using NNlib
using JSON
using Pickle
using SafeTensors

export DistilBertConfig, DistilBertModel

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
    sinusoidal_pos_embds::Bool
    tie_weights::Bool
    output_hidden_states::Bool
    output_attentions::Bool
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
    sinusoidal_pos_embds=false,
    tie_weights=true,
    output_hidden_states=false,
    output_attentions=false
)
    return DistilBertConfig(
        vocab_size, dim, n_layers, n_heads, hidden_dim, dropout,
        max_position_embeddings, initializer_range, qa_dropout, seq_classif_dropout,
        sinusoidal_pos_embds, tie_weights, output_hidden_states, output_attentions
    )
end


struct Embeddings
    word_embeddings::Embedding
    position_embeddings::Embedding
    LayerNorm::LayerNorm
    dropout::Dropout
end

Flux.@layer Embeddings

function Embeddings(config::DistilBertConfig)
    return Embeddings(
        Embedding(config.vocab_size => config.dim),
        Embedding(config.max_position_embeddings => config.dim),
        LayerNorm(config.dim),
        Dropout(config.dropout)
    )
end

function (m::Embeddings)(input_ids)
    seq_length = size(input_ids, 1)

    words_embeddings = m.word_embeddings(input_ids) # (dim, seq_len, batch_size)

    # Position embeddings: (dim, seq_len) -> broadcast to (dim, seq_len, batch_size)
    # We use 1:seq_length for Julia indices
    pos_ids = 1:seq_length
    position_embeddings = m.position_embeddings(pos_ids) # (dim, seq_len)

    embeddings = words_embeddings .+ position_embeddings
    embeddings = m.LayerNorm(embeddings)
    embeddings = m.dropout(embeddings)

    return embeddings
end

struct MultiHeadSelfAttention
    n_heads::Int
    dim::Int
    head_dim::Int
    q_lin::Dense
    k_lin::Dense
    v_lin::Dense
    out_lin::Dense
    dropout::Dropout
end

Flux.@layer MultiHeadSelfAttention

function MultiHeadSelfAttention(config::DistilBertConfig)
    head_dim = config.dim ÷ config.n_heads
    return MultiHeadSelfAttention(
        config.n_heads,
        config.dim,
        head_dim,
        Dense(config.dim => config.dim),
        Dense(config.dim => config.dim),
        Dense(config.dim => config.dim),
        Dense(config.dim => config.dim),
        Dropout(config.dropout)
    )
end

function (m::MultiHeadSelfAttention)(x, mask=nothing)
    # x shape: (dim, seq_len, batch_size)
    dim, seq_len, batch_size = size(x)

    q = m.q_lin(x)
    k = m.k_lin(x)
    v = m.v_lin(x)

    # Reshape for multi-head attention
    # (dim, seq_len, batch_size) -> (head_dim, n_heads, seq_len, batch_size)
    q = reshape(q, m.head_dim, m.n_heads, seq_len, batch_size)
    k = reshape(k, m.head_dim, m.n_heads, seq_len, batch_size)
    v = reshape(v, m.head_dim, m.n_heads, seq_len, batch_size)

    # Permute for attention: (head_dim, seq_len, n_heads, batch_size)
    q = permutedims(q, (1, 3, 2, 4))
    k = permutedims(k, (1, 3, 2, 4))
    v = permutedims(v, (1, 3, 2, 4))

    # Attention scores
    # Reshape to (head_dim, seq_len, n_heads * batch_size)
    q_reshaped = reshape(q, m.head_dim, seq_len, m.n_heads * batch_size)
    k_reshaped = reshape(k, m.head_dim, seq_len, m.n_heads * batch_size)

    # scores: (seq_len, seq_len, n_heads * batch_size)
    scores = batched_mul(permutedims(q_reshaped, (2, 1, 3)), k_reshaped)

    scores = scores ./ sqrt(Float32(m.head_dim))

    if mask !== nothing
        # Masking logic to be implemented
    end

    weights = softmax(scores, dims=2)
    weights = m.dropout(weights)

    v_reshaped = reshape(v, m.head_dim, seq_len, m.n_heads * batch_size)

    context = batched_mul(v_reshaped, permutedims(weights, (2, 1, 3)))

    # Reshape back: (head_dim, seq_len, n_heads, batch_size)
    context = reshape(context, m.head_dim, seq_len, m.n_heads, batch_size)

    # Permute back: (head_dim, n_heads, seq_len, batch_size) -> (dim, seq_len, batch_size)
    context = permutedims(context, (1, 3, 2, 4))
    context = reshape(context, dim, seq_len, batch_size)

    output = m.out_lin(context)
    return output
end

struct FeedForward
    lin1::Dense
    lin2::Dense
    dropout::Dropout
end

Flux.@layer FeedForward

function FeedForward(config::DistilBertConfig)
    return FeedForward(
        Dense(config.dim => config.hidden_dim, gelu),
        Dense(config.hidden_dim => config.dim),
        Dropout(config.dropout)
    )
end

function (m::FeedForward)(x)
    return m.lin2(m.dropout(m.lin1(x)))
end


struct TransformerBlock
    attention::MultiHeadSelfAttention
    sa_layer_norm::LayerNorm
    ffn::FeedForward
    output_layer_norm::LayerNorm
end

Flux.@layer TransformerBlock

function TransformerBlock(config::DistilBertConfig)
    return TransformerBlock(
        MultiHeadSelfAttention(config),
        LayerNorm(config.dim),
        FeedForward(config),
        LayerNorm(config.dim)
    )
end

function (m::TransformerBlock)(x, mask=nothing)
    # Self-Attention
    sa_output = m.attention(x, mask)
    sa_output = m.sa_layer_norm(sa_output .+ x)

    # Feed Forward
    ffn_output = m.ffn(sa_output)
    output = m.output_layer_norm(ffn_output .+ sa_output)

    return output
end

struct DistilBertModel
    config::DistilBertConfig
    embeddings::Embeddings
    transformer::Chain
end

Flux.@layer DistilBertModel

function DistilBertModel(config::DistilBertConfig)
    return DistilBertModel(
        config,
        Embeddings(config),
        Chain([TransformerBlock(config) for _ in 1:config.n_layers]...)
    )
end

function (m::DistilBertModel)(input_ids; mask=nothing)
    x = m.embeddings(input_ids)

    for block in m.transformer
        x = block(x, mask)
    end

    return x
end

function load_model(path::String)
    config_path = joinpath(path, "config.json")
    if !isfile(config_path)
        error("config.json not found in $path")
    end

    config_dict = JSON.parsefile(config_path)

    # Helper to safer parsing
    function get_key(d, k, default)
        return get(d, string(k), default)
    end

    # We map JSON fields to our struct fields
    config = DistilBertConfig(
        vocab_size=get_key(config_dict, "vocab_size", 30522),
        dim=get_key(config_dict, "dim", 768),
        n_layers=get_key(config_dict, "n_layers", 6),
        n_heads=get_key(config_dict, "n_heads", 12),
        hidden_dim=get_key(config_dict, "hidden_dim", 3072),
        dropout=Float32(get_key(config_dict, "dropout", 0.1)),
        max_position_embeddings=get_key(config_dict, "max_position_embeddings", 512),
        initializer_range=Float32(get_key(config_dict, "initializer_range", 0.02)),
        qa_dropout=Float32(get_key(config_dict, "qa_dropout", 0.1)),
        seq_classif_dropout=Float32(get_key(config_dict, "seq_classif_dropout", 0.2)),
        sinusoidal_pos_embds=get_key(config_dict, "sinusoidal_pos_embds", false),
        tie_weights=get_key(config_dict, "tie_weights", true),
        output_hidden_states=get_key(config_dict, "output_hidden_states", false),
        output_attentions=get_key(config_dict, "output_attentions", false)
    )

    model = DistilBertModel(config)

    # Load weights
    safetensors_path = joinpath(path, "model.safetensors")
    pytorch_bin_path = joinpath(path, "pytorch_model.bin")

    state_dict = nothing

    if isfile(safetensors_path)
        println("Loading weights from $safetensors_path using SafeTensors...")
        state_dict = SafeTensors.load_safetensors(safetensors_path)
    elseif isfile(pytorch_bin_path)
        println("Loading weights from $pytorch_bin_path using Pickle...")
        state_dict = Pickle.load(open(pytorch_bin_path))
    else
        warn("No model weights found. Returning random initialized model.")
        return model
    end

    load_weights!(model, state_dict)

    return model
end

function load_weights!(model::DistilBertModel, state_dict)
    function load_dense!(dense::Dense, prefix::String)
        w_key = prefix * ".weight"
        b_key = prefix * ".bias"

        if haskey(state_dict, w_key)
            w = state_dict[w_key]
            copy!(dense.weight, Float32.(w))
        end

        if haskey(state_dict, b_key)
            b = state_dict[b_key]
            copy!(dense.bias, Float32.(b))
        end
    end

    function load_layernorm!(ln::LayerNorm, prefix::String)
        w_key = prefix * ".weight"
        b_key = prefix * ".bias"

        if haskey(state_dict, w_key)
            copy!(ln.diag.scale, Float32.(state_dict[w_key]))
        end
        if haskey(state_dict, b_key)
            copy!(ln.diag.bias, Float32.(state_dict[b_key]))
        end
    end

    function load_embedding!(emb::Embedding, key::String)
        if haskey(state_dict, key)
            w = state_dict[key]
            copy!(emb.weight, permutedims(Float32.(w), (2, 1)))
        end
    end

    # 1. Embeddings
    load_embedding!(model.embeddings.word_embeddings, "embeddings.word_embeddings.weight")
    load_embedding!(model.embeddings.position_embeddings, "embeddings.position_embeddings.weight")
    load_layernorm!(model.embeddings.LayerNorm, "embeddings.LayerNorm")

    # 2. Transformer Blocks
    for i in 1:model.config.n_layers
        layer_prefix = "transformer.layer.$(i-1)"
        block = model.transformer[i]

        load_dense!(block.attention.q_lin, "$layer_prefix.attention.q_lin")
        load_dense!(block.attention.k_lin, "$layer_prefix.attention.k_lin")
        load_dense!(block.attention.v_lin, "$layer_prefix.attention.v_lin")
        load_dense!(block.attention.out_lin, "$layer_prefix.attention.out_lin")
        load_layernorm!(block.sa_layer_norm, "$layer_prefix.sa_layer_norm")

        load_dense!(block.ffn.lin1, "$layer_prefix.ffn.lin1")
        load_dense!(block.ffn.lin2, "$layer_prefix.ffn.lin2")
        load_layernorm!(block.output_layer_norm, "$layer_prefix.output_layer_norm")
    end

    println("Weights loaded successfully.")
end

end # module Distilbert
