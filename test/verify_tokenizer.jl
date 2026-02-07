using Test
using Distilbert
using JSON

# Include the tokenizer directly or via Distilbert module if exported
include("../src/Tokenizer.jl")
using .Tokenizer

# Paths
files_dir = joinpath(@__DIR__, "../files")
vocab_file = joinpath(files_dir, "vocab.txt")

# Initialize Tokenizer
tokenizer = WordPieceTokenizer(vocab_file; do_lower_case=true)

# Test Cases
test_sentences = [
    "Hello, world!",
    "DistilBERT is amazing.",
    "This is a test of the tokenization logic.",
    "UnseenWordThatShouldBeSplit"
]

println("Running Tokenizer Verification...")

# Generate expected outputs using Python
python_script = """
from transformers import DistilBertTokenizer
import json
import os

files_dir = "$files_dir"
tokenizer = DistilBertTokenizer.from_pretrained(files_dir)

sentences = [
    "Hello, world!",
    "DistilBERT is amazing.",
    "This is a test of the tokenization logic.",
    "UnseenWordThatShouldBeSplit"
]

results = []
for sent in sentences:
    tokens = tokenizer.tokenize(sent)
    ids = tokenizer.encode(sent, add_special_tokens=True)
    results.append({"tokens": tokens, "ids": ids})

print(json.dumps(results))
"""

# Run Python script
python_executable = joinpath(@__DIR__, "../.venv/bin/python3")
if !isfile(python_executable)
    python_executable = "python3"
end

cmd = `$python_executable -c $python_script`
python_out = read(cmd, String)
expected_results = JSON.parse(python_out)

@testset "Tokenizer Tests" begin
    for (i, sent) in enumerate(test_sentences)
        expected = expected_results[i]
        expected_tokens = expected["tokens"]
        expected_ids = expected["ids"]

        # Tokenize (without special tokens)
        jl_tokens = tokenize(tokenizer, sent)

        println("Sentence: $sent")
        println("  Julia Tokens: $jl_tokens")
        println("  Pythn Tokens: $expected_tokens")

        @test jl_tokens == expected_tokens

        # Encode (with special tokens)
        jl_ids = encode(tokenizer, sent)

        println("  Julia IDs: $jl_ids")
        println("  Pythn IDs: $expected_ids")

        # Note: Validating Python 0-based vs Julia 1-based indexing for IDs?
        # The `encode` function in `Tokenizer.jl` returns the IDs from the vocab map.
        # If `load_vocab` stored 1-based indices (i), then jl_ids will be 1-based relative to file lines.
        # Python's transformers uses 0-based indices from the file.
        # So we expect jl_ids to be equal to expected_ids .+ 1 if we used i.
        # Let's check `load_vocab` implementation:
        # vocab[token] = i
        # So yes, they are 1-based.
        # Thus, we should compare jl_ids .- 1 with expected_ids.

        @test (jl_ids .- 1) == expected_ids
    end
end
