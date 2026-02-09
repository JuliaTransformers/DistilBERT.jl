
# In src/custom_mul.jl

"""
    fast_attention_scores(Q, K, scale)

Matches data layout (head_dim, n_heads, seq_len, batch_size).
Computes Q^T * K for each head/batch.
Result shape: (seq_len, seq_len, n_heads, batch_size)
"""
function fast_attention_scores(Q::AbstractArray{T,4}, K::AbstractArray{T,4}, scale::T) where T<:Real
    D, H, S, B = size(Q)
    # Result: (S, S, H, B).
    # Dim 1: query pos (s1)
    # Dim 2: key pos (s2)
    Scores = Array{T}(undef, S, S, H, B)

    @tturbo for b in 1:B
        for h in 1:H
            for s2 in 1:S
                for s1 in 1:S
                    val = zero(T)
                    for d in 1:D
                        val += Q[d, h, s1, b] * K[d, h, s2, b]
                    end
                    Scores[s1, s2, h, b] = val * scale
                end
            end
        end
    end
    return Scores
end

"""
    fast_attention_context(V, Weights)

Matches data layout.
V: (head_dim, n_heads, seq_len, batch_size)
Weights: (seq_len, seq_len, n_heads, batch_size) [Attention Scores after Softmax]
Result: (head_dim, n_heads, seq_len, batch_size)
"""
function fast_attention_context(V::AbstractArray{T,4}, W::AbstractArray{T,4}) where T<:Real
    D, H, S, B = size(V)
    # W is (S_query, S_key, H, B) ?
    # Standard: Context = V * W^T.
    # V is D x S_key. W is S_query x S_key (softmax over key).
    # Matmul: V * W^T -> (D x S_key) * (S_key x S_query) -> D x S_query.
    # My Weights input W is (S_query, S_key), same as Scores?
    # In Distilbert.jl: weights = softmax(scores, dims=2).
    # if Scores is (S_query, S_key), softmax sums over S_key.
    # So W[s1, s2] corresponds to Query=s1, Key=s2.

    # We want Ctx[:, s1] = sum_s2 V[:, s2] * W[s1, s2].

    Ctx = Array{T}(undef, D, H, S, B)

    @turbo for b in 1:B
        for h in 1:H
            for s1 in 1:S # Query pos (output col)
                for d in 1:D
                    val = zero(T)
                    for s2 in 1:S # Key pos (summation)
                        val += V[d, h, s2, b] * W[s1, s2, h, b]
                    end
                    Ctx[d, h, s1, b] = val
                end
            end
        end
    end
    return Ctx
end
