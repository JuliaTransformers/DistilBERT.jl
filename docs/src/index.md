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

CUDA support is provided by a package extension: load `CUDA.jl` to enable it.
`to_gpu`/`to_cpu` move parameters, and `predict`/`embed` automatically follow
the model's device.

```@docs
to_gpu
to_cpu
on_gpu
cuda_inputs
gpu_memory_info
to_half
to_bfloat16
to_float32
prepare_inputs
num_parameters
num_parameters_mib
```
