export predict, embed, unmask

# ============================================================================
# High-Level Inference API
# ============================================================================

"""
    predict(model, tokenizer, text) -> Matrix{Float32}

Run inference on a single text string.

> **Note:** Call `Flux.testmode!(model)` before inference to disable dropout.
> Call `Flux.trainmode!(model)` to re-enable it for training.

# Arguments
- `model::DistilBertModel`: The DistilBERT model
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `text::String`: Input text

# Returns
- `output::Array{Float32,3}`: Hidden states of shape (dim, seq_len, 1)

# Example
```julia
model = load_model("path/to/model")
tokenizer = WordPieceTokenizer("path/to/vocab.txt")
Flux.testmode!(model)
output = predict(model, tokenizer, "Hello world!")
```
"""
function predict(model::DistilBertModel, tokenizer::WordPieceTokenizer, text::String)
    input_ids = encode(tokenizer, text)
    input_matrix = reshape(input_ids, :, 1)
    return model(input_matrix)
end

"""
    predict(model, tokenizer, texts; max_length=512) -> Matrix{Float32}

Run batch inference on multiple texts with automatic padding and masking.

> **Note:** Call `Flux.testmode!(model)` before inference to disable dropout.

# Arguments
- `model::DistilBertModel`: The DistilBERT model
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `texts::Vector{String}`: Input texts
- `max_length::Int`: Maximum sequence length (default: 512)

# Returns
- `output::Array{Float32,3}`: Hidden states of shape (dim, seq_len, batch_size)

# Example
```julia
model = load_model("path/to/model")
tokenizer = WordPieceTokenizer("path/to/vocab.txt")
Flux.testmode!(model)
output = predict(model, tokenizer, ["Hello world!", "How are you?"])
```
"""
function predict(model::DistilBertModel, tokenizer::WordPieceTokenizer,
    texts::Vector{String}; max_length::Int=512)
    input_ids, attention_mask = encode_batch(tokenizer, texts; max_length=max_length)
    return model(input_ids; mask=attention_mask)
end


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
    output = predict(model, tokenizer, text)

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


"""
    unmask(model, tokenizer, text; top_k=5) -> Vector{NamedTuple}

Predict the masked token(s) in a text, similar to HuggingFace's `pipeline('fill-mask')`.
Alias for the fill-mask pipeline.

> **Note:** Call `Flux.testmode!(model)` before inference to disable dropout.

# Arguments
- `model::DistilBertForMaskedLM`: The masked language model
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `text::String`: Input text containing `[MASK]` token(s)
- `top_k::Int`: Number of top predictions to return per mask (default: 5)

# Returns
- `Vector{NamedTuple}`: Each entry has fields:
  - `score::Float32`: Probability of the predicted token
  - `token::String`: The predicted token
  - `token_id::Int`: The token ID
  - `sequence::String`: The full text with `[MASK]` replaced

# Example
```julia
model = load_model(DistilBertForMaskedLM, "models/big")
tokenizer = WordPieceTokenizer("models/big/vocab.txt")
Flux.testmode!(model)

results = unmask(model, tokenizer, "Hello I'm a [MASK] model.")
for r in results
    println("\$(r.score)  \$(r.token)  =>  \$(r.sequence)")
end
```
"""
function unmask(model::DistilBertForMaskedLM, tokenizer::WordPieceTokenizer, text::String;
    top_k::Int=5)
    # The tokenizer splits [MASK] into "[", "mask", "]" due to punctuation handling.
    # Fix: split text on [MASK], tokenize each segment, stitch with mask token ID.

    mask_token_id = tokenizer.vocab[tokenizer.mask_token]
    cls_id = tokenizer.vocab[tokenizer.cls_token]
    sep_id = tokenizer.vocab[tokenizer.sep_token]

    # Split text on [MASK] (case-insensitive)
    mask_pattern = r"\[MASK\]"i
    segments = split(text, mask_pattern)

    if length(segments) < 2
        error("No [MASK] token found in input text. Use [MASK] to indicate the token to predict.")
    end

    # Tokenize each segment (without special tokens) and stitch with mask IDs
    input_ids = Int[cls_id]
    mask_positions = Int[]

    for (i, segment) in enumerate(segments)
        # Tokenize this segment without special tokens
        seg_ids = encode(tokenizer, String(segment); add_special_tokens=false)
        append!(input_ids, seg_ids)

        # Insert mask token between segments (not after the last one)
        if i < length(segments)
            push!(input_ids, mask_token_id)
            push!(mask_positions, length(input_ids))
        end
    end
    push!(input_ids, sep_id)

    input_matrix = reshape(input_ids, :, 1)

    # Run the model
    logits = model(input_matrix)  # (vocab_size, seq_len, 1)

    id_to_token = tokenizer.ids_to_tokens

    # Process predictions for the first [MASK] position
    mask_pos = mask_positions[1]
    mask_logits = logits[:, mask_pos, 1]  # (vocab_size,)

    # Apply softmax to get probabilities
    probs = NNlib.softmax(mask_logits)

    # Get top-k predictions
    top_indices = partialsortperm(probs, 1:min(top_k, length(probs)); rev=true)

    results = map(top_indices) do idx
        token = get(id_to_token, idx, "[UNK]")
        # Reconstruct the sequence with the predicted token
        filled_ids = copy(input_ids)
        filled_ids[mask_pos] = idx
        # Decode back to text (skip [CLS] and [SEP])
        filled_tokens = [get(id_to_token, id, "[UNK]") for id in filled_ids[2:end-1]]
        sequence = _detokenize(filled_tokens)
        (score=probs[idx], token=token, token_id=idx, sequence=sequence)
    end

    return results
end

"""
    _detokenize(tokens) -> String

Reconstruct a string from WordPiece tokens, joining '##' subword pieces.
"""
function _detokenize(tokens::Vector{String})
    if isempty(tokens)
        return ""
    end

    parts = String[]
    for token in tokens
        if startswith(token, "##")
            # Subword continuation — append without space
            push!(parts, token[3:end])
        else
            if !isempty(parts)
                push!(parts, " ")
            end
            push!(parts, token)
        end
    end

    return join(parts)
end

