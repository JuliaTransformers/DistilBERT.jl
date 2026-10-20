# DistilBERT.jl

A pure Julia implementation of the DistilBERT model, built with Flux.jl.

## Quick Start

```julia
using DistilBERT, Flux

model = load_model("models/distilbert-base-uncased")   # dir or HF Hub ID
tokenizer = WordPieceTokenizer("models/distilbert-base-uncased/vocab.txt")
Flux.testmode!(model)                                  # disable dropout

output = predict(model, tokenizer, "Hello world!")     # (dim, seq_len, 1)
embedding = embed(model, tokenizer, "Hello world!"; pooling=MeanPooling)
```

Return values live on the same device as the model parameters.

## API Reference

```@contents
```

### Model & Config

```@docs
DistilBertConfig
DistilBertModel
load_model
```

### Tokenizer

```@docs
WordPieceTokenizer
tokenize
load_vocab
encode
encode_pair
encode_batch
```

### Inference & Embeddings

```@docs
predict
embed
unmask
```

### Pooling

```@docs
AbstractPooling
CLSPooling
MeanPooling
MaxPooling
DistilBERT.cls_pooling
DistilBERT.mean_pooling
DistilBERT.max_pooling
```

### Truncation

```@docs
AbstractTruncation
LongestFirst
OnlyFirst
OnlySecond
```

### Task-Specific Heads

```@docs
DistilBertForSequenceClassification
DistilBertForTokenClassification
DistilBertForQuestionAnswering
DistilBertForMaskedLM
```

### Device & Precision

GPU support is backend-agnostic: the device layer works with any
`GPUArraysCore.AbstractGPUArray`, and `to_gpu` uses Flux's active backend
(CUDA by default; select AMDGPU/Metal with `Flux.gpu_backend!`). `predict` and
`embed` automatically move inputs to the model's device.

```@docs
to_gpu
to_cpu
on_gpu
to_half
to_bfloat16
to_float32
prepare_inputs
num_parameters
num_parameters_mib
```
