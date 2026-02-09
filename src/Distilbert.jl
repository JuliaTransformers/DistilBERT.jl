module Distilbert

using Flux
using NNlib
using JSON
using Pickle
using SafeTensors
using LoopVectorization

# Include Tokenizer submodule
include("Tokenizer.jl")
include("custom_mul.jl")
using .Tokenizer: WordPieceTokenizer, tokenize, encode, encode_pair, encode_batch, load_vocab

export DistilBertConfig, DistilBertModel, load_model
export WordPieceTokenizer, tokenize, encode, encode_pair, encode_batch, load_vocab

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

function (m::Embeddings)(input_ids::AbstractMatrix{<:Integer})
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

function (m::MultiHeadSelfAttention)(x::AbstractArray{<:Real,3}, mask::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
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

    # Fast attention scores using LoopVectorization
    # Computes Q^T * K and scales, keeping (head_dim, n_heads, seq_len, batch_size) layout
    # Output: (seq_len, seq_len, n_heads, batch_size)
    scale = 1.0f0 / sqrt(Float32(m.head_dim))
    scores = fast_attention_scores(q, k, scale)

    # Handle masking
    if mask !== nothing
        # Optimized mask expansion using broadcasting
        # mask shape: (seq_len, batch_size) -> expand to match scores
        # Reshape mask to (1, seq_len, 1, batch_size)
        mask_reshaped = reshape(mask, 1, seq_len, 1, batch_size)
        # We can broadcast this directly against scores (seq, seq, heads, batch)
        # scores is (s1, s2, h, b). Mask applies to s2.
        # mask_reshaped is (1, s2, 1, b).
        # Broadcast will match dims.
        scores = scores .+ (1.0f0 .- mask_reshaped) .* -1.0f9
    end

    # Apply softmax. Reshape to treat heads*batch as one dim
    scores_flat = reshape(scores, seq_len, seq_len, m.n_heads * batch_size)
    weights_flat = softmax(scores_flat, dims=2)
    weights_flat = m.dropout(weights_flat)

    # Reshape back for context computation
    weights = reshape(weights_flat, seq_len, seq_len, m.n_heads, batch_size)

    # Fast context computation using LoopVectorization
    # Computes V * Weights^T
    # Output: (head_dim, n_heads, seq_len, batch_size)
    context = fast_attention_context(v, weights)

    # Final reshape for output linear layer
    # combine heads: (head_dim, n_heads, seq_len, batch_size) -> (dim, seq_len, batch_size)
    # Memory layout of (head_dim, n_heads, seq_len, batch_size) allows direct reshape to (dim, seq_len, batch_size)
    # because head_dim varies fastest, then n_heads.
    context_flat = reshape(context, dim, seq_len, batch_size)

    output = m.out_lin(context_flat)
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

function (m::FeedForward)(x::AbstractArray{<:Real,3})
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

function (m::TransformerBlock)(x::AbstractArray{<:Real,3}, mask::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
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

function (m::DistilBertModel)(input_ids::AbstractMatrix{<:Integer}; mask::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
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
        @debug "Loading weights from $safetensors_path using SafeTensors..."
        state_dict = SafeTensors.load_safetensors(safetensors_path)
    elseif isfile(pytorch_bin_path)
        @debug "Loading weights from $pytorch_bin_path using Pickle..."
        state_dict = Pickle.load(open(pytorch_bin_path))
    else
        @warn "No model weights found. Returning randomly initialized model."
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

    @debug "Weights loaded successfully."
end

# ============================================================================
# High-Level Inference API
# ============================================================================

"""
    inference(model, tokenizer, text) -> Matrix{Float32}

Run inference on a single text string.

# Arguments
- `model::DistilBertModel`: The DistilBERT model (automatically set to test mode)
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `text::String`: Input text

# Returns
- `output::Array{Float32,3}`: Hidden states of shape (dim, seq_len, 1)

# Example
```julia
model = load_model("path/to/model")
tokenizer = WordPieceTokenizer("path/to/vocab.txt")
output = inference(model, tokenizer, "Hello world!")
```
"""
function inference(model::DistilBertModel, tokenizer::WordPieceTokenizer, text::String)
    Flux.testmode!(model)
    input_ids = encode(tokenizer, text)
    input_matrix = reshape(input_ids, :, 1)
    return model(input_matrix)
end

"""
    inference(model, tokenizer, texts; max_length=512) -> Matrix{Float32}

Run batch inference on multiple texts with automatic padding and masking.

# Arguments
- `model::DistilBertModel`: The DistilBERT model (automatically set to test mode)
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `texts::Vector{String}`: Input texts
- `max_length::Int`: Maximum sequence length (default: 512)

# Returns
- `output::Array{Float32,3}`: Hidden states of shape (dim, seq_len, batch_size)

# Example
```julia
model = load_model("path/to/model")
tokenizer = WordPieceTokenizer("path/to/vocab.txt")
output = inference(model, tokenizer, ["Hello world!", "How are you?"])
```
"""
function inference(model::DistilBertModel, tokenizer::WordPieceTokenizer,
    texts::Vector{String}; max_length::Int=512)
    Flux.testmode!(model)
    input_ids, attention_mask = encode_batch(tokenizer, texts; max_length=max_length)
    return model(input_ids; mask=attention_mask)
end

export inference

# ============================================================================
# Pooling Strategies
# ============================================================================

"""
    cls_pooling(output) -> Matrix{Float32}

Extract the [CLS] token representation (first token) from model output.

# Arguments
- `output::Array{Float32,3}`: Model output of shape (dim, seq_len, batch_size)

# Returns
- `Matrix{Float32}`: Shape (dim, batch_size)
"""
function cls_pooling(output::AbstractArray{<:Real,3})
    return output[:, 1, :]
end

"""
    mean_pooling(output, attention_mask) -> Matrix{Float32}

Compute mean of token embeddings, weighted by attention mask.

# Arguments
- `output::Array{Float32,3}`: Model output of shape (dim, seq_len, batch_size)
- `attention_mask::Matrix{Float32}`: Mask of shape (seq_len, batch_size)

# Returns
- `Matrix{Float32}`: Shape (dim, batch_size)
"""
function mean_pooling(output::AbstractArray{<:Real,3}, attention_mask::AbstractMatrix{<:Real})
    # output: (dim, seq_len, batch_size)
    # mask: (seq_len, batch_size) -> expand to (1, seq_len, batch_size)
    mask_expanded = reshape(attention_mask, 1, size(attention_mask)...)

    # Mask the output and sum
    masked_output = output .* mask_expanded
    sum_embeddings = dropdims(sum(masked_output, dims=2), dims=2)  # (dim, batch_size)

    # Count non-padding tokens per batch
    sum_mask = sum(attention_mask, dims=1)  # (1, batch_size)
    sum_mask = max.(sum_mask, 1.0f0)  # Avoid division by zero

    return sum_embeddings ./ sum_mask
end

"""
    max_pooling(output, attention_mask) -> Matrix{Float32}

Compute max of token embeddings, excluding padding tokens.

# Arguments
- `output::Array{Float32,3}`: Model output of shape (dim, seq_len, batch_size)
- `attention_mask::Matrix{Float32}`: Mask of shape (seq_len, batch_size)

# Returns
- `Matrix{Float32}`: Shape (dim, batch_size)
"""
function max_pooling(output::AbstractArray{<:Real,3}, attention_mask::AbstractMatrix{<:Real})
    # Set padding positions to very negative values so they don't affect max
    mask_expanded = reshape(attention_mask, 1, size(attention_mask)...)
    masked_output = output .* mask_expanded .+ (1.0f0 .- mask_expanded) .* -1.0f9

    return dropdims(maximum(masked_output, dims=2), dims=2)
end

export cls_pooling, mean_pooling, max_pooling

# ============================================================================
# Task-Specific Heads
# ============================================================================

"""
    DistilBertForSequenceClassification

DistilBERT model with a classification head for sequence classification tasks.
"""
struct DistilBertForSequenceClassification
    distilbert::DistilBertModel
    pre_classifier::Dense
    classifier::Dense
    dropout::Dropout
end

Flux.@layer DistilBertForSequenceClassification

"""
    DistilBertForSequenceClassification(config, num_labels)

Create a sequence classification model.

# Arguments
- `config::DistilBertConfig`: Model configuration
- `num_labels::Int`: Number of classification labels
"""
function DistilBertForSequenceClassification(config::DistilBertConfig, num_labels::Int)
    return DistilBertForSequenceClassification(
        DistilBertModel(config),
        Dense(config.dim => config.dim, relu),
        Dense(config.dim => num_labels),
        Dropout(config.seq_classif_dropout)
    )
end

function (m::DistilBertForSequenceClassification)(input_ids::AbstractMatrix{<:Integer}; mask=nothing)
    hidden_states = m.distilbert(input_ids; mask=mask)
    pooled_output = cls_pooling(hidden_states)  # (dim, batch_size)
    pooled_output = m.pre_classifier(pooled_output)
    pooled_output = m.dropout(pooled_output)
    logits = m.classifier(pooled_output)
    return logits  # (num_labels, batch_size)
end

"""
    DistilBertForTokenClassification

DistilBERT model with a token classification head (e.g., NER, POS tagging).
"""
struct DistilBertForTokenClassification
    distilbert::DistilBertModel
    classifier::Dense
    dropout::Dropout
end

Flux.@layer DistilBertForTokenClassification

"""
    DistilBertForTokenClassification(config, num_labels)

Create a token classification model.

# Arguments
- `config::DistilBertConfig`: Model configuration
- `num_labels::Int`: Number of token labels
"""
function DistilBertForTokenClassification(config::DistilBertConfig, num_labels::Int)
    return DistilBertForTokenClassification(
        DistilBertModel(config),
        Dense(config.dim => num_labels),
        Dropout(config.dropout)
    )
end

function (m::DistilBertForTokenClassification)(input_ids::AbstractMatrix{<:Integer}; mask=nothing)
    hidden_states = m.distilbert(input_ids; mask=mask)  # (dim, seq_len, batch_size)
    hidden_states = m.dropout(hidden_states)
    logits = m.classifier(hidden_states)
    return logits  # (num_labels, seq_len, batch_size)
end

"""
    DistilBertForQuestionAnswering

DistilBERT model with a span prediction head for extractive QA.
"""
struct DistilBertForQuestionAnswering
    distilbert::DistilBertModel
    qa_outputs::Dense
end

Flux.@layer DistilBertForQuestionAnswering

"""
    DistilBertForQuestionAnswering(config)

Create a question answering model.
"""
function DistilBertForQuestionAnswering(config::DistilBertConfig)
    return DistilBertForQuestionAnswering(
        DistilBertModel(config),
        Dense(config.dim => 2)  # start_logits and end_logits
    )
end

function (m::DistilBertForQuestionAnswering)(input_ids::AbstractMatrix{<:Integer}; mask=nothing)
    hidden_states = m.distilbert(input_ids; mask=mask)  # (dim, seq_len, batch_size)
    logits = m.qa_outputs(hidden_states)  # (2, seq_len, batch_size)
    start_logits = logits[1, :, :]  # (seq_len, batch_size)
    end_logits = logits[2, :, :]    # (seq_len, batch_size)
    return (start_logits=start_logits, end_logits=end_logits)
end

export DistilBertForSequenceClassification, DistilBertForTokenClassification, DistilBertForQuestionAnswering

# ============================================================================
# Sentence Embeddings
# ============================================================================

"""
    embed(model, tokenizer, text; pooling=:cls) -> Vector{Float32}

Get sentence embedding for a single text.

# Arguments
- `model::DistilBertModel`: The DistilBERT model
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `text::String`: Input text
- `pooling::Symbol`: Pooling strategy - `:cls`, `:mean`, or `:max` (default: `:cls`)

# Returns
- `Vector{Float32}`: Sentence embedding of shape (dim,)
"""
function embed(model::DistilBertModel, tokenizer::WordPieceTokenizer, text::String; pooling::Symbol=:cls)
    output = inference(model, tokenizer, text)

    if pooling == :cls
        return vec(cls_pooling(output))
    elseif pooling == :mean
        # For single text, all tokens are valid
        mask = ones(Float32, size(output, 2), 1)
        return vec(mean_pooling(output, mask))
    elseif pooling == :max
        mask = ones(Float32, size(output, 2), 1)
        return vec(max_pooling(output, mask))
    else
        error("Unknown pooling strategy: $pooling. Use :cls, :mean, or :max")
    end
end

"""
    embed(model, tokenizer, texts; pooling=:cls, max_length=512) -> Matrix{Float32}

Get sentence embeddings for multiple texts.

# Arguments
- `model::DistilBertModel`: The DistilBERT model
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `texts::Vector{String}`: Input texts
- `pooling::Symbol`: Pooling strategy - `:cls`, `:mean`, or `:max` (default: `:cls`)
- `max_length::Int`: Maximum sequence length (default: 512)

# Returns
- `Matrix{Float32}`: Sentence embeddings of shape (dim, batch_size)
"""
function embed(model::DistilBertModel, tokenizer::WordPieceTokenizer, texts::Vector{String};
    pooling::Symbol=:cls, max_length::Int=512)
    Flux.testmode!(model)
    input_ids, attention_mask = encode_batch(tokenizer, texts; max_length=max_length)
    output = model(input_ids; mask=attention_mask)

    if pooling == :cls
        return cls_pooling(output)
    elseif pooling == :mean
        return mean_pooling(output, attention_mask)
    elseif pooling == :max
        return max_pooling(output, attention_mask)
    else
        error("Unknown pooling strategy: $pooling. Use :cls, :mean, or :max")
    end
end

export embed

end # module Distilbert
