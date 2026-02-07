using Test
using Distilbert
include("../src/Tokenizer.jl")
using .Tokenizer
using JSON
using Flux

# Paths
files_dir = joinpath(@__DIR__, "../files")
vocab_file = joinpath(files_dir, "vocab.txt")

# 1. Load Julia Model and Tokenizer
println("Loading Julia model...")
model = Distilbert.load_model(files_dir)
testmode!(model)
tokenizer = WordPieceTokenizer(vocab_file; do_lower_case=true)

# 2. Input Data
text = "DistilBERT is amazing."
println("Input text: \"$text\"")

# Tokenize and Encode (Julia)
input_ids = encode(tokenizer, text)
println("Julia Input IDs: $input_ids")

# Prepare input for Flux model
# Shape: (seq_len, batch_size) -> (N, 1)
input_matrix = reshape(input_ids, :, 1)

# Run Julia Embeddings only
println("Running Julia embeddings...")
julia_embeddings = model.embeddings(input_matrix)
# (dim, seq_len, 1)

# Run Julia Forward Pass
println("Running Julia forward pass...")
julia_output = model(input_matrix)
println("Julia output shape: ", size(julia_output))


# 3. Run Python Logic
println("Running Python reference model...")

python_script = """
import torch
from transformers import DistilBertModel, DistilBertTokenizer
import json
import os

files_dir = "$files_dir"
text = "$text"

# Load Model and Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(files_dir)
model = DistilBertModel.from_pretrained(files_dir)
model.eval()

# Tokenize
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Forward Pass
with torch.no_grad():
    # Get Embeddings
    # embeddings = model.embeddings(input_ids)
    embedding_output = model.embeddings(input_ids)

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

# Output results
print(json.dumps({
    "input_ids": input_ids.tolist()[0],
    "embedding_output": embedding_output.tolist()[0], # (seq_len, dim)
    "last_hidden_state": last_hidden_state.tolist()[0]
}))
"""

# Detect Python Executable
python_executable = joinpath(@__DIR__, "../.venv/bin/python3")
if !isfile(python_executable)
    # Fallback or check user provided
    python_executable = "python3"
end

cmd = `$python_executable -c $python_script`
python_out = read(cmd, String)
py_results = JSON.parse(python_out)

py_input_ids = Int.(py_results["input_ids"])
py_hidden_state = hcat([Float32.(row) for row in py_results["last_hidden_state"]]...)
py_embeddings = hcat([Float32.(row) for row in py_results["embedding_output"]]...)

println("Python Input IDs: $py_input_ids")

# 4. Compare
@testset "End-to-End Verification" begin
    # Verify Inputs match
    # Julia IDs are 1-based (from Tokenizer), Python are 0-based.
    @test (input_ids .- 1) == py_input_ids

    # Verify Embeddings
    # julia_embeddings: (dim, seq_len, 1) -> (dim, seq_len)
    jl_emb_flat = dropdims(julia_embeddings, dims=3)

    diff_emb = abs.(jl_emb_flat .- py_embeddings)
    max_diff_emb = maximum(diff_emb)
    mean_diff_emb = sum(diff_emb) / length(diff_emb)
    println("Embeddings Max Diff: $max_diff_emb")
    println("Embeddings Mean Diff: $mean_diff_emb")
    @test max_diff_emb < 1e-4

    # Verify Outputs match
    jl_out_flat = dropdims(julia_output, dims=3)

    diff = abs.(jl_out_flat .- py_hidden_state)
    max_diff = maximum(diff)
    mean_diff = sum(diff) / length(diff)

    println("Output Max difference: $max_diff")
    println("Output Mean difference: $mean_diff")

    @test max_diff < 1e-4
end
