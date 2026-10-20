export load_model

"""
    load_model(source; init_random=false)
    load_model(::Type{DistilBertForMaskedLM}, source; init_random=false)
    load_model(::Type{DistilBertForSequenceClassification}, source; num_labels=2, init_random=false)
    load_model(::Type{DistilBertForTokenClassification}, source; num_labels, init_random=false)
    load_model(::Type{DistilBertForQuestionAnswering}, source; init_random=false)

Load a pre-trained DistilBERT model.
`source` can be:
- A **local directory** containing `config.json` + `model.safetensors`
- A **HuggingFace Hub model ID** (e.g. `"distilbert-base-uncased"`) — requires
  `TransformersDownloads.jl` to be in the active environment.

Weight format priority: SafeTensors > PyTorch `.bin` (Pickle).

# Examples
```julia
# From a local directory:
model = load_model("models/distilbert-base-uncased")

# Directly from HuggingFace Hub (downloads and caches automatically):
model = load_model("distilbert-base-uncased")

# Sequence classification:
model = load_model(DistilBertForSequenceClassification, "distilbert-base-uncased"; num_labels=2)
```
"""
function load_model(source::String; kwargs...)
    return load_model(DistilBertModel, source; kwargs...)
end

function load_model(::Type{T}, source::String; init_random::Bool=false, kwargs...) where {T}
    config, state_dict = _load_config_and_weights(source; init_random)
    model = _build_model(T, config; kwargs...)
    state_dict !== nothing && load_weights!(model, state_dict)
    return model
end

# Model-specific construction; typo'd keywords surface as MethodErrors.
_build_model(::Type{DistilBertModel}, config::DistilBertConfig) =
    DistilBertModel(config)

_build_model(::Type{DistilBertForSequenceClassification}, config::DistilBertConfig; num_labels::Int=2) =
    DistilBertForSequenceClassification(config, num_labels)

_build_model(::Type{DistilBertForTokenClassification}, config::DistilBertConfig; num_labels::Int) =
    DistilBertForTokenClassification(config, num_labels)

_build_model(::Type{DistilBertForQuestionAnswering}, config::DistilBertConfig) =
    DistilBertForQuestionAnswering(config)

_build_model(::Type{DistilBertForMaskedLM}, config::DistilBertConfig) =
    DistilBertForMaskedLM(config)

# ============================================================================
# Internal: Config & Weight Loading
# Delegates to TransformersDownloads.jl when it is loaded (preferred), or
# falls back to direct file access for fully local / offline use.
# ============================================================================

"""
    _parse_distilbert_config(config_dict) -> DistilBertConfig

Convert a raw config dictionary (from JSON) to a `DistilBertConfig`.
Isolated here so both the local-path and HF-Hub code paths share the logic.
"""
function _parse_distilbert_config(config_dict::AbstractDict)
    DistilBertConfig(
        vocab_size             = get(config_dict, "vocab_size",             30522),
        dim                    = get(config_dict, "dim",                    768),
        n_layers               = get(config_dict, "n_layers",               6),
        n_heads                = get(config_dict, "n_heads",                12),
        hidden_dim             = get(config_dict, "hidden_dim",             3072),
        dropout                = Float32(get(config_dict, "dropout",                0.1)),
        max_position_embeddings= get(config_dict, "max_position_embeddings",512),
        initializer_range      = Float32(get(config_dict, "initializer_range",     0.02)),
        qa_dropout             = Float32(get(config_dict, "qa_dropout",             0.1)),
        seq_classif_dropout    = Float32(get(config_dict, "seq_classif_dropout",    0.2)),
        layer_norm_eps         = Float32(get(config_dict, "layer_norm_eps",         1e-12)),
    )
end

"""
    _load_config_and_weights(source; init_random=false)

Load config and state dict from `source` (local dir or HF Hub ID).

Uses `TransformersDownloads.jl` when available (enables HF Hub, caching,
sharded safetensors). Falls back to direct file I/O for offline / local use.
"""
function _load_config_and_weights(source::String; init_random::Bool=false)
    # ── Try TransformersDownloads (Hub + local unified) ──────────────────────
    if _td_available()
        return _load_via_td(source; init_random)
    end

    # ── Fallback: local directory only ───────────────────────────────────────
    isdir(source) || error(
        "\"$source\" is not a local directory and TransformersDownloads.jl is not loaded.\n" *
        "For HuggingFace Hub support, add TransformersDownloads to your environment:\n" *
        "  Pkg.develop(path=\"../TransformersDownloads.jl\")\n" *
        "  using TransformersDownloads"
    )

    config_path = joinpath(source, "config.json")
    isfile(config_path) || error("config.json not found in $source")
    config_dict = JSON.parsefile(config_path)
    config = _parse_distilbert_config(config_dict)

    if init_random
        @warn "init_random=true: returning randomly initialized model (no weights loaded)."
        return config, nothing
    end

    state_dict = _load_weights_local(source)
    return config, state_dict
end

# ----------------------------------------------------------------------------
# TransformersDownloads integration hooks
#
# The core package only declares these functions; ext/DistilBERTTDExt.jl
# implements them when TransformersDownloads.jl is loaded. Using
# `Base.get_extension` keeps the core free of `Main`/UUID lookups and makes
# the dependency act as a proper package extension.
# ----------------------------------------------------------------------------

"""Return true if the TransformersDownloads extension is loaded."""
function _td_available()
    return Base.get_extension(@__MODULE__, :DistilBERTTDExt) !== nothing
end

function _td_load_config end
function _td_load_weights end
function _td_vocab end

function _load_via_td(source::String; init_random::Bool=false)
    config_dict = _td_load_config(source)
    config      = _parse_distilbert_config(config_dict)

    if init_random
        @warn "init_random=true: returning randomly initialized model."
        return config, nothing
    end

    @debug "Loading weights via TransformersDownloads for: $source"
    state_dict = _td_load_weights(source)
    return config, state_dict
end

"""Direct file-based weight loading (offline / no TransformersDownloads)."""
function _load_weights_local(dir::String)
    st_path  = joinpath(dir, "model.safetensors")
    bin_path = joinpath(dir, "pytorch_model.bin")

    if isfile(st_path)
        @debug "Loading weights: $st_path (SafeTensors)"
        return SafeTensors.load_safetensors(st_path)
    elseif isfile(bin_path)
        @debug "Loading weights: $bin_path (Pickle)"
        return open(bin_path, "r") do f
            Pickle.load(f)
        end
    else
        error(
            "No weights found in $dir.\n" *
            "Expected: model.safetensors  or  pytorch_model.bin"
        )
    end
end

# ============================================================================
# Weight Loading Helpers
# ============================================================================

function load_dense!(dense::Dense, prefix::String, state_dict, used_keys::Set{String}, key_prefix::String)
    w_key = key_prefix * prefix * ".weight"
    b_key = key_prefix * prefix * ".bias"

    if haskey(state_dict, w_key)
        push!(used_keys, w_key)
        w = state_dict[w_key]
        copy!(dense.weight, Float32.(w))
    else
        @warn "Missing weight: $w_key"
    end

    if haskey(state_dict, b_key)
        push!(used_keys, b_key)
        b = state_dict[b_key]
        copy!(dense.bias, Float32.(b))
    else
        @warn "Missing weight: $b_key"
    end
end

function load_layernorm!(ln::LayerNorm, prefix::String, state_dict, used_keys::Set{String}, key_prefix::String)
    w_key = key_prefix * prefix * ".weight"
    b_key = key_prefix * prefix * ".bias"

    if haskey(state_dict, w_key)
        push!(used_keys, w_key)
        copy!(ln.diag.scale, Float32.(state_dict[w_key]))
    else
        @warn "Missing weight: $w_key"
    end
    if haskey(state_dict, b_key)
        push!(used_keys, b_key)
        copy!(ln.diag.bias, Float32.(state_dict[b_key]))
    else
        @warn "Missing weight: $b_key"
    end
end

function load_embedding!(emb::Embedding, key::String, state_dict, used_keys::Set{String}, key_prefix::String)
    full_key = key_prefix * key
    if haskey(state_dict, full_key)
        push!(used_keys, full_key)
        w = state_dict[full_key]
        copy!(emb.weight, permutedims(Float32.(w), (2, 1)))
    else
        @warn "Missing weight: $full_key"
    end
end

function _detect_key_prefix(state_dict)
    first_key = first(keys(state_dict))
    if startswith(first_key, "distilbert.")
        @info "Detected 'distilbert.' prefix in weight keys — stripping for base model loading."
        return "distilbert."
    end
    return ""
end

function _load_base_weights!(model::DistilBertModel, state_dict, used_keys::Set{String}, key_prefix::String)
    # 1. Embeddings
    load_embedding!(model.embeddings.word_embeddings, "embeddings.word_embeddings.weight", state_dict, used_keys, key_prefix)
    load_embedding!(model.embeddings.position_embeddings, "embeddings.position_embeddings.weight", state_dict, used_keys, key_prefix)
    load_layernorm!(model.embeddings.layer_norm, "embeddings.LayerNorm", state_dict, used_keys, key_prefix)

    # 2. Transformer Blocks
    for i in 1:model.config.n_layers
        layer_prefix = "transformer.layer.$(i-1)"
        block = model.transformer[i]

        load_dense!(block.attention.q_lin, "$layer_prefix.attention.q_lin", state_dict, used_keys, key_prefix)
        load_dense!(block.attention.k_lin, "$layer_prefix.attention.k_lin", state_dict, used_keys, key_prefix)
        load_dense!(block.attention.v_lin, "$layer_prefix.attention.v_lin", state_dict, used_keys, key_prefix)
        load_dense!(block.attention.out_lin, "$layer_prefix.attention.out_lin", state_dict, used_keys, key_prefix)
        load_layernorm!(block.sa_layer_norm, "$layer_prefix.sa_layer_norm", state_dict, used_keys, key_prefix)

        load_dense!(block.ffn.lin1, "$layer_prefix.ffn.lin1", state_dict, used_keys, key_prefix)
        load_dense!(block.ffn.lin2, "$layer_prefix.ffn.lin2", state_dict, used_keys, key_prefix)
        load_layernorm!(block.output_layer_norm, "$layer_prefix.output_layer_norm", state_dict, used_keys, key_prefix)
    end
end

function _report_unused_keys(state_dict, used_keys)
    unused_keys = setdiff(keys(state_dict), used_keys)
    if !isempty(unused_keys)
        @info "Unused keys in state_dict ($(length(unused_keys))): $(collect(unused_keys))"
    end
    @debug "Weights loaded successfully ($(length(used_keys)) keys loaded)."
end

function load_weights!(model::DistilBertModel, state_dict)
    used_keys = Set{String}()
    key_prefix = _detect_key_prefix(state_dict)
    _load_base_weights!(model, state_dict, used_keys, key_prefix)
    _report_unused_keys(state_dict, used_keys)
end

function load_weights!(model::DistilBertForSequenceClassification, state_dict)
    used_keys = Set{String}()
    key_prefix = _detect_key_prefix(state_dict)
    _load_base_weights!(model.distilbert, state_dict, used_keys, key_prefix)

    # Load head weights
    load_dense!(model.pre_classifier, "pre_classifier", state_dict, used_keys, "")
    load_dense!(model.classifier, "classifier", state_dict, used_keys, "")

    _report_unused_keys(state_dict, used_keys)
end

function load_weights!(model::DistilBertForTokenClassification, state_dict)
    used_keys = Set{String}()
    key_prefix = _detect_key_prefix(state_dict)
    _load_base_weights!(model.distilbert, state_dict, used_keys, key_prefix)

    # Load head weights
    load_dense!(model.classifier, "classifier", state_dict, used_keys, "")

    _report_unused_keys(state_dict, used_keys)
end

function load_weights!(model::DistilBertForQuestionAnswering, state_dict)
    used_keys = Set{String}()
    key_prefix = _detect_key_prefix(state_dict)
    _load_base_weights!(model.distilbert, state_dict, used_keys, key_prefix)

    # Load head weights
    load_dense!(model.qa_outputs, "qa_outputs", state_dict, used_keys, "")

    _report_unused_keys(state_dict, used_keys)
end

function load_weights!(model::DistilBertForMaskedLM, state_dict)
    used_keys = Set{String}()
    key_prefix = _detect_key_prefix(state_dict)
    _load_base_weights!(model.distilbert, state_dict, used_keys, key_prefix)

    # Load MLM head weights
    load_dense!(model.vocab_transform, "vocab_transform", state_dict, used_keys, "")
    load_layernorm!(model.vocab_layer_norm, "vocab_layer_norm", state_dict, used_keys, "")

    # Load vocab_projector bias (weight is tied to word embeddings)
    bias_key = "vocab_projector.bias"
    if haskey(state_dict, bias_key)
        push!(used_keys, bias_key)
        copy!(model.vocab_projector_bias, Float32.(state_dict[bias_key]))
    else
        @warn "Missing weight: $bias_key"
    end

    _report_unused_keys(state_dict, used_keys)
end

