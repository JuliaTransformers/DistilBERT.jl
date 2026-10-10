
using Unicode

export WordPieceTokenizer, tokenize, encode, encode_pair, encode_batch, load_vocab
export AbstractTruncation, LongestFirst, OnlyFirst, OnlySecond

"""
    WordPieceTokenizer

WordPiece tokenizer compatible with BERT-style tokenization.

# Fields
- `vocab::Dict{String,Int}`: Vocabulary mapping from token to ID
- `ids_to_tokens::Dict{Int,String}`: Mapping from ID to token
- `unk_token::String`: Unknown token (default: "[UNK]")
- `sep_token::String`: Separator token (default: "[SEP]")
- `pad_token::String`: Padding token (default: "[PAD]")
- `cls_token::String`: Classification token (default: "[CLS]")
- `mask_token::String`: Mask token (default: "[MASK]")
- `do_lower_case::Bool`: Whether to lowercase input text and strip accents
  (matches HuggingFace `BertTokenizer` with `do_lower_case=true`)
"""
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

"""
    WordPieceTokenizer(source; kwargs...)

Create a WordPieceTokenizer from:

- a direct `vocab.txt` file path,
- a local model directory containing `vocab.txt`, or
- a HuggingFace Hub model ID (requires `TransformersDownloads.jl`).

# Arguments
- `source::String`: Path to the vocabulary file or model directory, or an HF Hub ID.
- `do_lower_case::Bool=true`: Whether to lowercase input text.

# Examples
```julia
# From a vocab file:
tok = WordPieceTokenizer("models/distilbert-base-uncased/vocab.txt")

# From a model directory:
tok = WordPieceTokenizer("models/distilbert-base-uncased")

# From HuggingFace Hub (requires TransformersDownloads.jl):
using TransformersDownloads
tok = WordPieceTokenizer("distilbert-base-uncased")
```
"""
function WordPieceTokenizer(source::String;
    do_lower_case=true,
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    mask_token="[MASK]")
    vocab_path = if isfile(source)
        source
    elseif isdir(source)
        joinpath(source, "vocab.txt")
    elseif _td_available()
        _td_vocab(source)
    else
        error("\"$source\" is not a file or directory. Load TransformersDownloads.jl for HF Hub support.")
    end
    isfile(vocab_path) || error("vocab.txt not found in $source")

    vocab = load_vocab(vocab_path)
    ids_to_tokens = Dict(v => k for (k, v) in vocab)
    return WordPieceTokenizer(vocab, ids_to_tokens, unk_token, sep_token, pad_token, cls_token, mask_token, do_lower_case)
end

"""
    load_vocab(vocab_file::String)

Load vocabulary from a file.
"""
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

"""
    _strip_accents(text) -> String

Remove combining marks (Unicode category `Mn`) after NFKD normalization,
matching HuggingFace `BertTokenizer._run_strip_accents`.
"""
function _strip_accents(text::String)
    normalized = Unicode.normalize(text, :NFKD)
    return filter(c -> Base.Unicode.category_abbrev(c) != "Mn", normalized)
end

function basic_tokenize(text::String, do_lower_case::Bool)
    if do_lower_case
        # HF `BertTokenizer` strips accents and lowercases when do_lower_case=true
        text = lowercase(_strip_accents(text))
    end

    tokens = String[]
    buffer = IOBuffer()

    for char in text
        if isspace(char)
            if position(buffer) > 0
                push!(tokens, String(take!(buffer)))
            end
        elseif ispunct(char)
            if position(buffer) > 0
                push!(tokens, String(take!(buffer)))
            end
            push!(tokens, string(char))  # More efficient than String([char])
        else
            write(buffer, char)
        end
    end

    if position(buffer) > 0
        push!(tokens, String(take!(buffer)))
    end

    return tokens
end

function wordpiece_tokenize(token::String, vocab::Dict{String,Int}, unk_token::String)
    len = length(token)
    output_tokens = String[]
    start_char = 1

    while start_char <= len
        end_char = len
        found = false

        while start_char <= end_char
            # Byte indices of the start of characters start_char and end_char.
            # `SubString` treats these as character indices and includes the
            # full character at `end_byte`, so multibyte tokens stay intact.
            start_byte = nextind(token, 0, start_char)
            end_byte = nextind(token, 0, end_char)

            # First-piece probes hash a SubString (allocation-free). Continuation
            # probes need the "##" prefix, which cannot be represented as a
            # contiguous SubString, so the key is materialised.
            if start_char > 1
                key = "##" * SubString(token, start_byte, end_byte)
                if haskey(vocab, key)
                    push!(output_tokens, key)
                    start_char = end_char + 1
                    found = true
                    break
                end
            else
                key = SubString(token, start_byte, end_byte)
                if haskey(vocab, key)
                    push!(output_tokens, String(key))
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

"""
    tokenize(tokenizer, text) -> Vector{String}

Tokenize a text string into WordPiece tokens.

# Arguments
- `tokenizer::WordPieceTokenizer`: The tokenizer
- `text::String`: Input text

# Returns
- `Vector{String}`: List of tokens
"""
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
        append!(ids, fill(pad_id, max_length - length(ids)))
    end

    return ids
end

# ============================================================================
# Truncation Strategies
# ============================================================================

"""
    AbstractTruncation

Supertype for truncation strategies accepted by [`encode_pair`](@ref):

- [`LongestFirst`](@ref) (default): trim the longer sequence first.
- [`OnlyFirst`](@ref): trim only the first sequence.
- [`OnlySecond`](@ref): trim only the second sequence.

Pass the *type* as the `truncation` keyword, e.g. `truncation=OnlySecond`.
Passing a `Symbol` (`:longest_first`, `:only_first`, `:only_second`) is deprecated.
"""
abstract type AbstractTruncation end

"""Truncation strategy: trim the longer sequence first."""
struct LongestFirst <: AbstractTruncation end

"""Truncation strategy: trim only the first sequence."""
struct OnlyFirst <: AbstractTruncation end

"""Truncation strategy: trim only the second sequence."""
struct OnlySecond <: AbstractTruncation end

const _TRUNCATION_FROM_SYMBOL = Dict(
    :longest_first => LongestFirst,
    :only_first    => OnlyFirst,
    :only_second   => OnlySecond,
)

_resolve_truncation(::Type{T}) where {T<:AbstractTruncation} = T
function _resolve_truncation(truncation::Symbol)
    Base.depwarn(
        "`truncation=$truncation` is deprecated; pass `LongestFirst`, `OnlyFirst`, or `OnlySecond`.",
        :encode_pair)
    haskey(_TRUNCATION_FROM_SYMBOL, truncation) && return _TRUNCATION_FROM_SYMBOL[truncation]
    error("Unknown truncation strategy: $truncation. Use LongestFirst, OnlyFirst, or OnlySecond.")
end

"""
    encode_pair(tokenizer, text_a, text_b; kwargs...) -> NamedTuple

Encode a sentence pair (e.g., for question answering or NLI tasks).

# Keyword Arguments
- `max_length::Union{Nothing,Int}=nothing`: Maximum total sequence length
- `truncation`: `LongestFirst` (default), `OnlyFirst`, or `OnlySecond`.
  Legacy `Symbol`s (`:longest_first`, `:only_first`, `:only_second`) still work
  but are deprecated.
- `padding::Bool=false`: Whether to pad to max_length

# Returns
NamedTuple with:
- `input_ids::Vector{Int}`: Token IDs
- `token_type_ids::Vector{Int}`: 0 for first sentence, 1 for second
- `attention_mask::Vector{Float32}`: 1.0 for real tokens, 0.0 for padding
"""
function encode_pair(tokenizer::WordPieceTokenizer, text_a::String, text_b::String;
    max_length::Union{Nothing,Int}=nothing,
    truncation=LongestFirst,
    padding::Bool=false)
    t = _resolve_truncation(truncation)
    tokens_a = tokenize(tokenizer, text_a)
    tokens_b = tokenize(tokenizer, text_b)

    # Format: [CLS] tokens_a [SEP] tokens_b [SEP]
    # Token types: 0 0 0 ... 0 0 1 1 ... 1 1
    special_tokens_count = 3  # [CLS], [SEP], [SEP]

    # Truncate if needed
    if max_length !== nothing
        max_tokens = max_length - special_tokens_count

        if length(tokens_a) + length(tokens_b) > max_tokens
            excess = length(tokens_a) + length(tokens_b) - max_tokens
            if t === LongestFirst
                # HF `_truncate_seq_pair`: trim the longer sequence one token at
                # a time until within budget (ties trim the second sequence).
                while length(tokens_a) + length(tokens_b) > max_tokens
                    if length(tokens_a) > length(tokens_b)
                        tokens_a = tokens_a[1:end-1]
                    else
                        tokens_b = tokens_b[1:end-1]
                    end
                end
            elseif t === OnlyFirst
                tokens_a = tokens_a[1:max(0, length(tokens_a) - excess)]
            elseif t === OnlySecond
                tokens_b = tokens_b[1:max(0, length(tokens_b) - excess)]
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
    padding in (:longest, :max_length) || throw(ArgumentError(
        "padding must be :longest or :max_length, got :$padding"))

    # Encode all texts
    all_ids = [encode(tokenizer, t; max_length=max_length, truncation=true) for t in texts]

    # Determine target length based on padding strategy
    max_actual_len = maximum(length, all_ids)
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

