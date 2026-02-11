export load_model

"""
    load_model(path; init_random=false)
    load_model(::Type{DistilBertForMaskedLM}, path; init_random=false)
    load_model(::Type{DistilBertForSequenceClassification}, path; num_labels=2, init_random=false)
    load_model(::Type{DistilBertForTokenClassification}, path; num_labels, init_random=false)
    load_model(::Type{DistilBertForQuestionAnswering}, path; init_random=false)

Load a pre-trained DistilBERT model from a directory.
The directory must contain:
- `config.json`: Model configuration
- `model.safetensors` (preferred) or `pytorch_model.bin`: Model weights

# Arguments
- `path::String`: Path to the directory containing model files
- `init_random::Bool=false`: If true, return a randomly initialized model when no weights are found.

# Examples
```julia
# Base model
model = load_model("path/to/model")

# Sequence classification (e.g., sentiment analysis)
model = load_model(DistilBertForSequenceClassification, "path/to/model"; num_labels=2)

# Question answering
model = load_model(DistilBertForQuestionAnswering, "path/to/model")
```
"""
function load_model(path::String; init_random::Bool=false)
    config, state_dict = _load_config_and_weights(path; init_random)
    model = DistilBertModel(config)

    if state_dict !== nothing
        load_weights!(model, state_dict)
    end

    return model
end

function load_model(::Type{DistilBertForSequenceClassification}, path::String;
    num_labels::Int=2, init_random::Bool=false)
    config, state_dict = _load_config_and_weights(path; init_random)
    model = DistilBertForSequenceClassification(config, num_labels)

    if state_dict !== nothing
        load_weights!(model, state_dict)
    end

    return model
end

function load_model(::Type{DistilBertForTokenClassification}, path::String;
    num_labels::Int, init_random::Bool=false)
    config, state_dict = _load_config_and_weights(path; init_random)
    model = DistilBertForTokenClassification(config, num_labels)

    if state_dict !== nothing
        load_weights!(model, state_dict)
    end

    return model
end

function load_model(::Type{DistilBertForQuestionAnswering}, path::String;
    init_random::Bool=false)
    config, state_dict = _load_config_and_weights(path; init_random)
    model = DistilBertForQuestionAnswering(config)

    if state_dict !== nothing
        load_weights!(model, state_dict)
    end

    return model
end

# ============================================================================
# Internal: Config & Weight Loading
# ============================================================================

function _load_config_and_weights(path::String; init_random::Bool=false)
    config_path = joinpath(path, "config.json")
    if !isfile(config_path)
        error("config.json not found in $path")
    end

    config_dict = JSON.parsefile(config_path)

    config = DistilBertConfig(
        vocab_size=get(config_dict, "vocab_size", 30522),
        dim=get(config_dict, "dim", 768),
        n_layers=get(config_dict, "n_layers", 6),
        n_heads=get(config_dict, "n_heads", 12),
        hidden_dim=get(config_dict, "hidden_dim", 3072),
        dropout=Float32(get(config_dict, "dropout", 0.1)),
        max_position_embeddings=get(config_dict, "max_position_embeddings", 512),
        initializer_range=Float32(get(config_dict, "initializer_range", 0.02)),
        qa_dropout=Float32(get(config_dict, "qa_dropout", 0.1)),
        seq_classif_dropout=Float32(get(config_dict, "seq_classif_dropout", 0.2)),
        layer_norm_eps=Float32(get(config_dict, "layer_norm_eps", 1e-12))
    )

    # Load weights
    safetensors_path = joinpath(path, "model.safetensors")
    pytorch_bin_path = joinpath(path, "pytorch_model.bin")

    state_dict = nothing

    if isfile(safetensors_path)
        @debug "Loading weights from $safetensors_path using SafeTensors..."
        state_dict = SafeTensors.load_safetensors(safetensors_path)
    elseif isfile(pytorch_bin_path)
        @debug "Loading weights from $pytorch_bin_path using Pickle..."
        state_dict = open(pytorch_bin_path, "r") do f
            Pickle.load(f)
        end
    else
        if init_random
            @warn "No model weights found in $path. Returning randomly initialized model."
        else
            error("No model weights found in $path. Pass init_random=true to use random weights.")
        end
    end

    return config, state_dict
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

function load_model(::Type{DistilBertForMaskedLM}, path::String;
    init_random::Bool=false)
    config, state_dict = _load_config_and_weights(path; init_random)
    model = DistilBertForMaskedLM(config)

    if state_dict !== nothing
        load_weights!(model, state_dict)
    end

    return model
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

