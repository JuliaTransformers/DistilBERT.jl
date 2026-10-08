struct Embeddings
    word_embeddings::Embedding
    position_embeddings::Embedding
    layer_norm::LayerNorm
    dropout::Dropout
end

Flux.@layer Embeddings

function Embeddings(config::DistilBertConfig)
    return Embeddings(
        Embedding(config.vocab_size => config.dim),
        Embedding(config.max_position_embeddings => config.dim),
        LayerNorm(config.dim; eps=config.layer_norm_eps),
        Dropout(config.dropout)
    )
end

function (m::Embeddings)(input_ids::AbstractMatrix{<:Integer})
    seq_length = size(input_ids, 1)

    words_embeddings = m.word_embeddings(input_ids)           # (dim, seq_len, batch_size)
    position_embeddings = m.position_embeddings(1:seq_length) # (dim, seq_len)

    # Broadcasting adds position_embeddings over the batch dimension
    embeddings = m.layer_norm(words_embeddings .+ position_embeddings)
    return m.dropout(embeddings)
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

Flux.@layer MultiHeadSelfAttention trainable = (q_lin, k_lin, v_lin, out_lin)

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

# ---------------------------------------------------------------------------
# mask is pre-converted by the model's forward pass via `_make_nnlib_mask`
# so each block receives a ready-to-use Bool array (or nothing).
# ---------------------------------------------------------------------------

"""
    _make_nnlib_mask(mask) -> Union{Nothing, AbstractArray{Bool,4}}

Convert a `(seq_len, batch_size)` attention mask to the 4-D Bool array
format expected by `NNlib.dot_product_attention`:
  `(kv_len, 1, 1, batch_size)` where `true` means *keep* (not masked).

Calling this once per forward pass and sharing the result across all
transformer blocks avoids N×reshape + comparison allocations.
"""
_make_nnlib_mask(::Nothing) = nothing
function _make_nnlib_mask(mask::AbstractMatrix{<:Real})
    # reshape is zero-copy; materialise the Bool comparison once
    return reshape(mask .!= 0, size(mask, 1), 1, 1, size(mask, 2))
end
# Accept already-converted masks (idempotent, enables re-use)
_make_nnlib_mask(mask::AbstractArray{Bool,4}) = mask

function (m::MultiHeadSelfAttention)(x::AbstractArray{<:Real,3};
    mask::Union{Nothing,AbstractArray}=nothing)
    # x: (dim, seq_len, batch_size)
    # mask is expected to already be in NNlib 4-D Bool format
    # (converted once by `DistilBertModel` before the block loop)
    q = m.q_lin(x)
    k = m.k_lin(x)
    v = m.v_lin(x)

    # NNlib dispatches to cuDNN fused attention on CUDA when available
    context, _ = dot_product_attention(q, k, v;
        nheads=m.n_heads,
        mask=mask,
        fdrop=m.dropout)

    return m.out_lin(context)
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

function (m::FeedForward)(x::AbstractArray{<:Real,3})
    return m.dropout(m.lin2(m.lin1(x)))
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
        LayerNorm(config.dim; eps=config.layer_norm_eps),
        FeedForward(config),
        LayerNorm(config.dim; eps=config.layer_norm_eps)
    )
end

function (m::TransformerBlock)(x::AbstractArray{<:Real,3};
    mask::Union{Nothing,AbstractArray}=nothing)
    # mask is already in NNlib 4-D Bool format; pass through unchanged
    sa_output = m.attention(x; mask)
    sa_output = m.sa_layer_norm(sa_output .+ x)   # pre-norm residual

    ffn_output = m.ffn(sa_output)
    return m.output_layer_norm(ffn_output .+ sa_output)
end
