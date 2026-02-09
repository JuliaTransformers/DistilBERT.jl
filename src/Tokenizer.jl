module Tokenizer

using Unicode

export WordPieceTokenizer, tokenize, encode, encode_batch, load_vocab

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

function encode(tokenizer::WordPieceTokenizer, text::String; max_length=nothing, pad_to_max_length=false)
    tokens = tokenize(tokenizer, text)

    # Add special tokens
    tokens = [tokenizer.cls_token; tokens; tokenizer.sep_token]

    # Convert to IDs
    unk_id = get(tokenizer.vocab, tokenizer.unk_token, 0)
    ids = [get(tokenizer.vocab, t, unk_id) for t in tokens]

    if max_length !== nothing
        if length(ids) > max_length
            ids = ids[1:max_length-1]
            push!(ids, get(tokenizer.vocab, tokenizer.sep_token, 0))
        elseif pad_to_max_length
            pad_id = get(tokenizer.vocab, tokenizer.pad_token, 0)
            while length(ids) < max_length
                push!(ids, pad_id)
            end
        end
    end

    return ids
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
