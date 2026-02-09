module Tokenizer

using Unicode

export WordPieceTokenizer, tokenize, encode, encode_pair, encode_batch, load_vocab

struct WordPieceTokenizer
    vocab::Dict{String,Int}
    ids_to_tokens::Dict{Int,String}
    unk_token::String
    sep_token::String
    pad_token::String
    cls_token::String
    mask_token::String
    do_lower_case::Bool
end

function WordPieceTokenizer(vocab_file::String;
    do_lower_case=true,
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    mask_token="[MASK]")
    vocab = load_vocab(vocab_file)
    ids_to_tokens = Dict(v => k for (k, v) in vocab)
    return WordPieceTokenizer(vocab, ids_to_tokens, unk_token, sep_token, pad_token, cls_token, mask_token, do_lower_case)
end

function load_vocab(vocab_file::String)
    vocab = Dict{String,Int}()
    open(vocab_file, "r") do f
        for (i, line) in enumerate(eachline(f))
            token = strip(line)
            if !isempty(token)
                vocab[token] = i
            end
        end
    end
    return vocab
end

function is_punctuation(char::Char)
    return ispunct(char) || (char in ['-', '_', '.', ',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '"', '\'', '`', '~', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '/', '\\', '|'])
end

function basic_tokenize(text::String, do_lower_case::Bool)
    if do_lower_case
        text = lowercase(text)
    end

    tokens = String[]
    buffer = IOBuffer()

    for char in text
        if isspace(char)
            if buffer.size > 0
                push!(tokens, String(take!(buffer)))
            end
        elseif is_punctuation(char)
            if buffer.size > 0
                push!(tokens, String(take!(buffer)))
            end
            push!(tokens, string(char))  # More efficient than String([char])
        else
            write(buffer, char)
        end
    end

    if buffer.size > 0
        push!(tokens, String(take!(buffer)))
    end

    return tokens
end

function wordpiece_tokenize(token::String, vocab::Dict{String,Int}, unk_token::String)
    # Use character indices instead of collecting into array
    len = length(token)  # Number of characters (not bytes)
    output_tokens = String[]
    start_char = 1

    while start_char <= len
        end_char = len
        found = false

        while start_char <= end_char
            # Use SubString to avoid allocation - creates a view of the original string
            # We need character indices, so use thisind/nextind for proper Unicode handling
            start_byte = thisind(token, start_char)
            end_byte = thisind(token, end_char)
            # Get the byte range for the substring
            if end_char < len
                end_byte = prevind(token, nextind(token, end_byte))
            else
                end_byte = lastindex(token)
            end

            substr_view = SubString(token, start_byte, end_byte)

            # Check with/without ## prefix
            if start_char > 1
                substr_with_prefix = "##" * substr_view
                if haskey(vocab, substr_with_prefix)
                    push!(output_tokens, substr_with_prefix)
                    start_char = end_char + 1
                    found = true
                    break
                end
            else
                if haskey(vocab, substr_view)
                    push!(output_tokens, String(substr_view))
                    start_char = end_char + 1
                    found = true
                    break
                end
            end
            end_char -= 1
        end

        if !found
            return [unk_token]
        end
    end

    return output_tokens
end

function tokenize(tokenizer::WordPieceTokenizer, text::String)
    basic_tokens = basic_tokenize(text, tokenizer.do_lower_case)
    wordpiece_tokens = String[]

    for token in basic_tokens
        subtokens = wordpiece_tokenize(token, tokenizer.vocab, tokenizer.unk_token)
        append!(wordpiece_tokens, subtokens)
    end

    return wordpiece_tokens
end

"""
    encode(tokenizer, text; kwargs...) -> Vector{Int}

Encode a single text to token IDs.

# Keyword Arguments
- `add_special_tokens::Bool=true`: Whether to add [CLS] and [SEP] tokens
- `max_length::Union{Nothing,Int}=nothing`: Maximum sequence length
- `truncation::Bool=false`: Whether to truncate sequences longer than max_length
- `padding::Bool=false`: Whether to pad sequences to max_length

# Returns
- `Vector{Int}`: Token IDs
"""
function encode(tokenizer::WordPieceTokenizer, text::String;
    add_special_tokens::Bool=true,
    max_length::Union{Nothing,Int}=nothing,
    truncation::Bool=false,
    padding::Bool=false)
    tokens = tokenize(tokenizer, text)

    # Add special tokens
    if add_special_tokens
        tokens = [tokenizer.cls_token; tokens; tokenizer.sep_token]
    end

    # Convert to IDs
    unk_id = get(tokenizer.vocab, tokenizer.unk_token, 0)
    ids = [get(tokenizer.vocab, t, unk_id) for t in tokens]

    # Handle truncation
    if max_length !== nothing && truncation && length(ids) > max_length
        if add_special_tokens
            # Keep [SEP] at the end
            ids = ids[1:max_length-1]
            push!(ids, get(tokenizer.vocab, tokenizer.sep_token, 0))
        else
            ids = ids[1:max_length]
        end
    end

    # Handle padding
    if max_length !== nothing && padding && length(ids) < max_length
        pad_id = get(tokenizer.vocab, tokenizer.pad_token, 0)
        while length(ids) < max_length
            push!(ids, pad_id)
        end
    end

    return ids
end

"""
    encode_pair(tokenizer, text_a, text_b; kwargs...) -> NamedTuple

Encode a sentence pair (e.g., for question answering or NLI tasks).

# Keyword Arguments
- `max_length::Union{Nothing,Int}=nothing`: Maximum total sequence length
- `truncation::Symbol=:longest_first`: Truncation strategy (:longest_first, :only_first, :only_second)
- `padding::Bool=false`: Whether to pad to max_length

# Returns
NamedTuple with:
- `input_ids::Vector{Int}`: Token IDs
- `token_type_ids::Vector{Int}`: 0 for first sentence, 1 for second
- `attention_mask::Vector{Float32}`: 1.0 for real tokens, 0.0 for padding
"""
function encode_pair(tokenizer::WordPieceTokenizer, text_a::String, text_b::String;
    max_length::Union{Nothing,Int}=nothing,
    truncation::Symbol=:longest_first,
    padding::Bool=false)
    tokens_a = tokenize(tokenizer, text_a)
    tokens_b = tokenize(tokenizer, text_b)

    # Format: [CLS] tokens_a [SEP] tokens_b [SEP]
    # Token types: 0 0 0 ... 0 0 1 1 ... 1 1
    special_tokens_count = 3  # [CLS], [SEP], [SEP]

    # Truncate if needed
    if max_length !== nothing
        max_tokens = max_length - special_tokens_count

        if length(tokens_a) + length(tokens_b) > max_tokens
            if truncation == :longest_first
                # Truncate the longer sequence first
                while length(tokens_a) + length(tokens_b) > max_tokens
                    if length(tokens_a) > length(tokens_b)
                        pop!(tokens_a)
                    else
                        pop!(tokens_b)
                    end
                end
            elseif truncation == :only_first
                excess = length(tokens_a) + length(tokens_b) - max_tokens
                tokens_a = tokens_a[1:max(1, length(tokens_a) - excess)]
            elseif truncation == :only_second
                excess = length(tokens_a) + length(tokens_b) - max_tokens
                tokens_b = tokens_b[1:max(1, length(tokens_b) - excess)]
            end
        end
    end

    # Build sequence
    tokens = [tokenizer.cls_token; tokens_a; tokenizer.sep_token; tokens_b; tokenizer.sep_token]

    # Token type IDs: 0 for segment A (including [CLS] and first [SEP]), 1 for segment B
    type_a_len = 1 + length(tokens_a) + 1  # [CLS] + tokens_a + [SEP]
    type_b_len = length(tokens_b) + 1       # tokens_b + [SEP]
    token_type_ids = [fill(0, type_a_len); fill(1, type_b_len)]

    # Convert to IDs
    unk_id = get(tokenizer.vocab, tokenizer.unk_token, 0)
    input_ids = [get(tokenizer.vocab, t, unk_id) for t in tokens]

    # Attention mask (all 1s for now)
    attention_mask = ones(Float32, length(input_ids))

    # Padding
    if max_length !== nothing && padding && length(input_ids) < max_length
        pad_id = get(tokenizer.vocab, tokenizer.pad_token, 0)
        pad_len = max_length - length(input_ids)
        append!(input_ids, fill(pad_id, pad_len))
        append!(token_type_ids, fill(0, pad_len))  # Padding gets type 0
        append!(attention_mask, zeros(Float32, pad_len))
    end

    return (input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
end

"""
    encode_batch(tokenizer, texts; max_length=512, padding=:longest)

Encode multiple texts into a batch matrix with padding and attention mask.

# Arguments
- `texts::Vector{String}`: Input texts to encode
- `max_length::Int`: Maximum sequence length (default: 512)
- `padding::Symbol`: Padding strategy - `:longest` or `:max_length` (default: `:longest`)

# Returns
- `input_ids::Matrix{Int}`: Shape (seq_len, batch_size) - token IDs
- `attention_mask::Matrix{Float32}`: Shape (seq_len, batch_size) - 1.0 for real tokens, 0.0 for padding
"""
function encode_batch(tokenizer::WordPieceTokenizer, texts::Vector{String};
    max_length::Int=512, padding::Symbol=:longest)
    # Encode all texts
    all_ids = [encode(tokenizer, t) for t in texts]

    # Determine target length based on padding strategy
    max_actual_len = maximum(length.(all_ids))
    target_len = if padding == :longest
        min(max_length, max_actual_len)
    else
        max_length
    end

    # Create output matrices
    pad_id = get(tokenizer.vocab, tokenizer.pad_token, 0)
    batch_size = length(texts)
    input_ids = fill(pad_id, target_len, batch_size)
    attention_mask = zeros(Float32, target_len, batch_size)

    for (i, ids) in enumerate(all_ids)
        len = min(length(ids), target_len)
        input_ids[1:len, i] = ids[1:len]
        attention_mask[1:len, i] .= 1.0f0
    end

    return input_ids, attention_mask
end

end
