export DistilBertModel, DistilBertForSequenceClassification, DistilBertForTokenClassification, DistilBertForQuestionAnswering, DistilBertForMaskedLM

"""
    DistilBertModel

The bare DistilBERT Model transformer outputting raw hidden-states without any specific head on top.
"""
struct DistilBertModel
    config::DistilBertConfig
    embeddings::Embeddings
    transformer::Chain
end

Flux.@layer DistilBertModel

function Base.show(io::IO, m::DistilBertModel)
    print(io, "DistilBertModel($(m.config))")
end

function DistilBertModel(config::DistilBertConfig)
    return DistilBertModel(
        config,
        Embeddings(config),
        Chain([TransformerBlock(config) for _ in 1:config.n_layers]...)
    )
end

function (m::DistilBertModel)(input_ids::AbstractMatrix{<:Integer}; mask::Union{Nothing,AbstractMatrix}=nothing)
    x = m.embeddings(input_ids)

    for block in m.transformer
        x = block(x; mask=mask)
    end

    return x
end

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

function (m::DistilBertForSequenceClassification)(input_ids::AbstractMatrix{<:Integer}; mask::Union{Nothing,AbstractMatrix}=nothing)
    hidden_states = m.distilbert(input_ids; mask=mask)
    pooled_output = hidden_states[:, 1, :]  # CLS token: (dim, batch_size)
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

function (m::DistilBertForTokenClassification)(input_ids::AbstractMatrix{<:Integer}; mask::Union{Nothing,AbstractMatrix}=nothing)
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

function (m::DistilBertForQuestionAnswering)(input_ids::AbstractMatrix{<:Integer}; mask::Union{Nothing,AbstractMatrix}=nothing)
    hidden_states = m.distilbert(input_ids; mask=mask)  # (dim, seq_len, batch_size)
    logits = m.qa_outputs(hidden_states)  # (2, seq_len, batch_size)
    start_logits = logits[1, :, :]  # (seq_len, batch_size)
    end_logits = logits[2, :, :]    # (seq_len, batch_size)
    return (start_logits=start_logits, end_logits=end_logits)
end

"""
    DistilBertForMaskedLM

DistilBERT model with a masked language modeling head on top.
"""
struct DistilBertForMaskedLM
    distilbert::DistilBertModel
    vocab_transform::Dense       # dim → dim (with gelu activation)
    vocab_layer_norm::LayerNorm  # dim
    vocab_projector_bias::Vector{Float32}  # (vocab_size,)
end

Flux.@layer DistilBertForMaskedLM trainable = (distilbert, vocab_transform, vocab_layer_norm)

"""
    DistilBertForMaskedLM(config)

Create a masked language modeling model.
"""
function DistilBertForMaskedLM(config::DistilBertConfig)
    return DistilBertForMaskedLM(
        DistilBertModel(config),
        Dense(config.dim => config.dim, gelu),
        LayerNorm(config.dim; eps=config.layer_norm_eps),
        zeros(Float32, config.vocab_size)
    )
end

function (m::DistilBertForMaskedLM)(input_ids::AbstractMatrix{<:Integer}; mask::Union{Nothing,AbstractMatrix}=nothing)
    hidden_states = m.distilbert(input_ids; mask=mask)  # (dim, seq_len, batch_size)

    # MLM head
    prediction_logits = m.vocab_transform(hidden_states)       # (dim, seq_len, batch_size)
    prediction_logits = m.vocab_layer_norm(prediction_logits)  # (dim, seq_len, batch_size)

    # Tied projection: use word_embeddings weight (stored as (dim, vocab_size) in Flux)
    # We need (vocab_size, dim) × (dim, ...) = (vocab_size, ...)
    embed_weight = m.distilbert.embeddings.word_embeddings.weight  # (dim, vocab_size)

    # Reshape for batched matmul: (dim, seq_len, batch) → multiply by embed_weight'
    d, s, b = size(prediction_logits)
    flat = reshape(prediction_logits, d, s * b)          # (dim, seq_len*batch)
    logits_flat = embed_weight' * flat                    # (vocab_size, seq_len*batch)
    logits = reshape(logits_flat, :, s, b)                # (vocab_size, seq_len, batch)

    # Add bias
    logits = logits .+ m.vocab_projector_bias

    return logits  # (vocab_size, seq_len, batch_size)
end

