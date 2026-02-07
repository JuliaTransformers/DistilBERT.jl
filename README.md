# Distilbert.jl

A pure Julia implementation of the DistilBERT model, built with Flux.jl.

This project implements the DistilBERT model architecture from scratch in Julia, including a custom WordPiece tokenizer. It is designed to be compatible with pre-trained weights from Hugging Face (converted to `.safetensors` format).

## Features

- **Pure Julia Implementation**: No Python dependencies for inference.
- **Flux.jl Integration**: Built using standard Flux layers and custom structs for Transformer components.
- **WordPiece Tokenizer**: Fully implemented tokenizer in Julia (matching `BertTokenizer` logic).
- **Weight Loading**: Supports loading weights from Hugging Face `model.safetensors` files.
- **Verified Accuracy**: Validated against Python's `transformers` library with < 1e-6 numerical difference.

## Installation

This package is currently a local project. To use it, clone the repository and instantiate the environment.

```bash
git clone https://github.com/StartYourStart/Distilbert.jl.git
cd Distilbert.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

### 1. Download Model Weights
You need the `config.json`, `vocab.txt`, and `model.safetensors` from a DistilBERT model (e.g., `distilbert-base-uncased`). Place them in a `files/` directory in the root of the project.

### 2. Run Inference
```julia
using Distilbert
using Flux

# Load model and tokenizer
model_path = "files/" # Directory containing config.json and model.safetensors
vocab_path = "files/vocab.txt"

println("Loading model...")
model = Distilbert.load_model(model_path)
tokenizer = Distilbert.Tokenizer.WordPieceTokenizer(vocab_path; do_lower_case=true)

# Tokenize input
text = "DistilBERT is amazing."
input_ids = Distilbert.Tokenizer.encode(tokenizer, text)

# Prepare input for Flux (Seq Length x Batch Size)
input_matrix = reshape(input_ids, :, 1)

# Run model
# Using testmode! effectively disables dropout for deterministic inference
testmode!(model)
output = model(input_matrix)

println("Output shape: ", size(output))
# (Hidden Dim, Seq Length, Batch Size)
```

## Verification

To ensure the implementation matches the reference PyTorch implementation, we provide a verification suite.

### Prerequisites (Python)
You need a Python environment with `torch` and `transformers` installed to run the verification scripts (which compare Julia output to Python output).

```bash
# Create a venv (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers safetensors numpy
```

### Running Verification Tests

Run the end-to-end verification script:

```bash
julia --project=. test/verify_end_to_end.jl
```

This script will:
1. Load the Julia model and tokenizer.
2. Run a sample sentence through the Julia model.
3. Automatically run a Python script (using the installed `.venv` or system python) to get reference outputs from Hugging Face `transformers`.
4. Compare Embeddings and Final Hidden States.
5. Report max and mean differences (should be `< 1e-6`).

### Component Tests
- `test/verify_tokenizer.jl`: Verifies tokenization logic against Python.
- `test/verify_weights_tensor_loading.jl`: Verifies tensor transposition logic.

## Project Structure

- `src/Distilbert.jl`: Main module containing model architecture (Embeddings, TransformerBlock, etc.) and weight loading logic.
- `src/Tokenizer.jl`: WordPiece tokenizer implementation.
- `test/`: Verification and test scripts.
