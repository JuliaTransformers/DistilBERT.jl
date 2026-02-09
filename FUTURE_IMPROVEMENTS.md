# Future Improvements for Distilbert.jl

## Performance

### 🟠 Attention Computation Allocations (Medium Priority)
**File:** `src/Distilbert.jl` - `MultiHeadSelfAttention`

The attention function creates many intermediate arrays through `reshape` and `permutedims`:
```julia
q = reshape(q, m.head_dim, m.n_heads, seq_len, batch_size)
q = permutedims(q, (1, 3, 2, 4))
# ... more allocations
```

**Recommendations:**
- Use `NNlib.batched_transpose` instead of `permutedims` where applicable
- Pre-allocate buffers for hot paths using `Zygote.Buffer`
- Consider fusing operations with custom CUDA kernels for GPU
- Profile with `@allocated` to identify worst offenders

**Caution:** Changes must preserve Zygote autodiff compatibility.

---

### 🟢 GPU Support (Low Priority - Future Feature)
Currently the library works on CPU. For production use:
- Add CUDA.jl integration
- Ensure all operations use GPU-compatible types
- Add `gpu(model)` / `cpu(model)` utilities
- Benchmark against Python on GPU

---

## Features

### 🟠 Task-Specific Heads (Medium Priority)
Add pre-built classification heads for common tasks:

```julia
# Sequence classification (sentiment, etc.)
struct DistilBertForSequenceClassification
    distilbert::DistilBertModel
    pre_classifier::Dense
    classifier::Dense
    dropout::Dropout
end

# Token classification (NER, POS tagging)
struct DistilBertForTokenClassification
    distilbert::DistilBertModel
    classifier::Dense
    dropout::Dropout
end

# Question Answering
struct DistilBertForQuestionAnswering
    distilbert::DistilBertModel
    qa_outputs::Dense
end
```

---

### 🟠 Pooling Strategies (Medium Priority)
Add common pooling methods for sequence-level representations:

```julia
# CLS token pooling (default BERT style)
cls_pooling(output) = output[:, 1, :]

# Mean pooling over non-padding tokens
function mean_pooling(output, attention_mask)
    # Mask and average
end

# Max pooling
function max_pooling(output, attention_mask)
    # Mask and max
end
```

---

### 🟢 Sentence Embeddings (Low Priority)
High-level API for getting sentence embeddings:

```julia
function embed(model, tokenizer, text::String; pooling=:cls)
    output = inference(model, tokenizer, text)
    return apply_pooling(output, pooling)
end
```

---

## Tokenizer

### 🟢 Special Tokens Handling (Low Priority)
Add support for:
- `add_special_tokens` parameter in encode
- Token type IDs for sentence pairs
- Truncation strategies (`:longest_first`, `:only_first`, `:only_second`)

---

### 🟢 Vocab Caching (Low Priority)
Cache loaded vocabularies globally to avoid re-parsing:

```julia
const VOCAB_CACHE = Dict{String, Dict{String,Int}}()

function load_vocab_cached(vocab_file::String)
    path = abspath(vocab_file)
    get!(VOCAB_CACHE, path) do
        load_vocab(vocab_file)
    end
end
```

---

## Documentation

### 🟢 Docstrings for All Public Functions
Add docstrings to:
- `load_model`
- `DistilBertConfig`
- `tokenize`
- `encode`
- `encode_batch`

### 🟢 API Reference Documentation
Generate docs using Documenter.jl:
- Installation guide
- Quick start tutorial
- API reference
- Benchmarks vs Python

---

## Testing

### 🟢 Gradient Tests
Add tests verifying gradients flow correctly:

```julia
@testset "Gradients" begin
    model = DistilBertModel(DistilBertConfig(dim=64, n_heads=4, n_layers=2))
    x = rand(1:100, 10, 2)

    grads = gradient(model) do m
        sum(m(x))
    end

    @test grads[1] !== nothing
end
```

### 🟢 Edge Case Tests
- Empty strings
- Very long sequences (truncation)
- Unicode edge cases
- All-padding batches

---

## Compatibility

### 🟢 Other Model Weights (Low Priority)
Support loading weights from:
- ONNX format
- Different weight key prefixes (some HuggingFace models vary)
- Sharded checkpoints for large models

---

## Summary

| Priority | Category | Item |
|----------|----------|------|
| 🟠 Medium | Performance | Attention allocations optimization |
| 🟠 Medium | Features | Task-specific heads |X
| 🟠 Medium | Features | Pooling strategies |X
| 🟢 Low | Performance | GPU support |
| 🟢 Low | Features | Sentence embeddings API |X
| 🟢 Low | Tokenizer | Special tokens handling |X
| 🟢 Low | Tokenizer | Vocab caching |
| 🟢 Low | Docs | Docstrings |
| 🟢 Low | Docs | Documenter.jl site |
| 🟢 Low | Testing | Gradient tests |X
| 🟢 Low | Testing | Edge case tests |X
| 🟢 Low | Compat | Other weight formats |
