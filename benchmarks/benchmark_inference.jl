using Distilbert
using Flux
using Statistics
using Printf

# Paths
files_dir = joinpath(@__DIR__, "../files")
vocab_file = joinpath(files_dir, "vocab.txt")

# 1. Load Julia Model and Tokenizer
include("../src/Tokenizer.jl")
using .Tokenizer

println("Loading Julia model...")
model = Distilbert.load_model(files_dir)
testmode!(model)
tokenizer = Tokenizer.WordPieceTokenizer(vocab_file; do_lower_case=true)

# 2. Input Data
text = "DistilBERT is amazing. This is a benchmark test to compare performance."
println("Input text: \"$text\"")

# Tokenize and Encode
input_ids = Tokenizer.encode(tokenizer, text)
input_matrix = reshape(input_ids, :, 1)

# Warmup
println("Warming up Julia model...")
for _ in 1:5
    model(input_matrix)
end

# Benchmark Julia
n_iters = 100
println("Benchmarking Julia model ($n_iters iterations)...")
t_start = time()
for _ in 1:n_iters
    model(input_matrix)
end
t_end = time()
avg_time_jl = (t_end - t_start) / n_iters
println(@sprintf("Julia Average Time: %.4f ms", avg_time_jl * 1000))

# Benchmark Python
println("\nBenchmarking Python model...")
python_script = """
import torch
from transformers import DistilBertModel, DistilBertTokenizer
import time
import os

files_dir = "$files_dir"
text = "$text"
n_iters = $n_iters

# Load Model and Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(files_dir)
model = DistilBertModel.from_pretrained(files_dir)
model.eval()

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Warmup
with torch.no_grad():
    for _ in range(5):
        model(**inputs)

# Benchmark
start_time = time.time()
with torch.no_grad():
    for _ in range(n_iters):
        model(**inputs)
end_time = time.time()

avg_time = (end_time - start_time) / n_iters
print(f"{avg_time * 1000:.4f}")
"""

# Run Python script
python_executable = joinpath(@__DIR__, "../.venv/bin/python3")
if !isfile(python_executable)
    python_executable = "python3"
end

cmd = `$python_executable -c $python_script`
py_out = read(cmd, String)
avg_time_py = parse(Float64, strip(py_out))

println(@sprintf("Python Average Time: %.4f ms", avg_time_py))

# Comparison
ratio = avg_time_jl / (avg_time_py / 1000) # avg_time_py is in ms ?? No, it's printed as ms but parsed as number. Wait.
# Python script prints: f"{avg_time * 1000:.4f}" -> so it's in ms.
# Julia avg_time_jl is in seconds.
# Let's convert Julia to ms for comparison.
avg_time_jl_ms = avg_time_jl * 1000

println("-"^30)
println(@sprintf("Julia:  %.4f ms", avg_time_jl_ms))
println(@sprintf("Python: %.4f ms", avg_time_py))
if avg_time_jl_ms < avg_time_py
    println(@sprintf("Julia is %.2fx faster", avg_time_py / avg_time_jl_ms))
else
    println(@sprintf("Python is %.2fx faster", avg_time_jl_ms / avg_time_py))
end
println("-"^30)
