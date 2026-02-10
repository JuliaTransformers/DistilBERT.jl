# Future Improvements for DistilBERT.jl

## Performance

### 🟠 GPU Support (Medium Priority)
Currently the library works on CPU. For production use:
- Add CUDA.jl integration
- Ensure all operations use GPU-compatible types
- Add `gpu(model)` / `cpu(model)` utilities
- Benchmark against Python on GPU

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

## Features

### 🟠 Load Task-Specific Head Weights (Medium Priority)
`load_model` currently only loads the base `DistilBertModel`. Add support for loading
task-specific heads (`DistilBertForSequenceClassification`, etc.) with their head weights.

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
| 🟠 Medium | Performance | GPU support |
| 🟠 Medium | Features | Load task-specific head weights |
| 🟢 Low | Tokenizer | Vocab caching |
| 🟢 Low | Compat | Other weight formats |
