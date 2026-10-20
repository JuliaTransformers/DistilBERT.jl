"""
    module ValidationUtils

Shared utilities for parity validation between Julia and Python (HuggingFace Transformers).

Provides:
- `run_python_script(code)` — run a Python snippet and capture stdout
- `compare_tensors(a, b; atol, rtol)` — compare two arrays with tolerance
- `load_npy(path)` / `save_npy(value, path)` — numpy-compatible array I/O via NPZ
- `@validate` — macro for labelled tolerance checks
- `ReferenceCase` — struct for parameterised test cases
"""
module ValidationUtils

using Test
using Printf
using LinearAlgebra

export run_python_script, compare_tensors, load_npy, save_npy
export ReferenceCase, run_reference_cases
export @validate

# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

"""
    run_python_script(code; python="python3", timeout=60) -> String

Execute `code` as a Python one-liner (via `-c`) and return trimmed stdout.
Raises an error with stderr if the process exits non-zero.
"""
function run_python_script(code::String; python::String="python3", timeout::Int=60)
    buf = IOBuffer()
    err = IOBuffer()
    try
        run(pipeline(
            Cmd([python, "-c", code]; ignorestatus=false),
            stdout=buf, stderr=err
        ))
    catch e
        error_text = String(take!(err))
        error("Python script failed:\n$error_text\nOriginal error: $e")
    end
    return strip(String(take!(buf)))
end

"""
    detect_python(candidates=["python3","python"]) -> String

Return the first Python executable that can import `transformers`.
Raises an error if none found.
"""
function detect_python(candidates=["python3", "python"])
    for py in candidates
        try
            out = run_python_script("import transformers; print('ok')"; python=py)
            out == "ok" && return py
        catch
        end
    end
    error("No Python with `transformers` found. Install with: pip install transformers torch")
end

# ---------------------------------------------------------------------------
# Tensor comparison
# ---------------------------------------------------------------------------

"""
    compare_tensors(julia_out, python_out; atol=1f-4, rtol=1f-5, label="") -> Bool

Compare two float arrays element-wise. Prints a coloured report and returns
`true` if within tolerance.

Both arrays are permuted to `(batch, seq, dim)` order for display purposes only;
comparison is element-wise after broadcasting to the same shape.
"""
function compare_tensors(
    a::AbstractArray,
    b::AbstractArray;
    atol::Real=1f-4,
    rtol::Real=1f-5,
    label::String=""
)
    header = isempty(label) ? "" : "[$label] "
    a32 = Float32.(a)
    b32 = Float32.(b)

    if size(a32) != size(b32)
        @error "$(header)Shape mismatch: Julia=$(size(a32))  Python=$(size(b32))"
        return false
    end

    diff  = abs.(a32 .- b32)
    max_d = maximum(diff)
    mean_d = mean(diff)
    pass  = max_d <= atol

    status = pass ? "✅ PASS" : "❌ FAIL"
    @printf "%s %s  max_diff=%.2e  mean_diff=%.2e  (atol=%.2e)\n" status header max_d mean_d atol
    return pass
end

# ---------------------------------------------------------------------------
# NPZ / NPY I/O (via NPZ.jl if available, else plain binary)
# ---------------------------------------------------------------------------

"""
    save_npy(arr, path)

Save a Julia array to a NumPy `.npy` file using NPZ.jl if available,
otherwise falls back to writing raw Float32 binary with a minimal header.
"""
function save_npy(arr, path::String)
    try
        @eval using NPZ
        NPZ.npzwrite(path, Float32.(arr))
    catch
        # Fallback: write raw float32 little-endian binary
        open(path, "w") do io
            write(io, Float32.(arr[:]))
        end
        @warn "NPZ.jl not available. Wrote raw Float32 binary to $path (no numpy header)."
    end
end

"""
    load_npy(path) -> Array{Float32}

Load a `.npy` file as a Julia Float32 array using NPZ.jl.
"""
function load_npy(path::String)
    @eval using NPZ
    return Float32.(NPZ.npzread(path))
end

# ---------------------------------------------------------------------------
# Reference test case system
# ---------------------------------------------------------------------------

"""
    ReferenceCase

Parameterised validation case for a transformer model.

# Fields
- `name::String`: Human-readable label
- `model_id::String`: HuggingFace model ID *or* local directory path
- `test_texts::Vector{String}`: Input sentences
- `max_length::Int`: Tokenizer max length
- `atol::Float32`: Absolute tolerance for hidden-state comparison
"""
struct ReferenceCase
    name::String
    model_id::String
    test_texts::Vector{String}
    max_length::Int
    atol::Float32
end

ReferenceCase(name, model_id, texts; max_length=128, atol=1f-3) =
    ReferenceCase(name, model_id, texts, max_length, atol)

"""
    run_reference_cases(cases, validate_fn) -> Bool

Run `validate_fn(case)` for each `ReferenceCase` and print a summary table.
Returns `true` if all cases pass.
"""
function run_reference_cases(cases::Vector{ReferenceCase}, validate_fn)
    results = Pair{String,Bool}[]
    for c in cases
        @info "Running: $(c.name)"
        passed = try
            validate_fn(c)
        catch e
            @error "$(c.name) threw: $e"
            false
        end
        push!(results, c.name => passed)
    end

    println("\n", "="^60)
    println(" VALIDATION SUMMARY")
    println("="^60)
    all_pass = true
    for (name, ok) in results
        status = ok ? "✅" : "❌"
        println("  $status  $name")
        all_pass &= ok
    end
    println("="^60)
    return all_pass
end

# ---------------------------------------------------------------------------
# @validate macro
# ---------------------------------------------------------------------------

"""
    @validate label expr atol

Sugar over `compare_tensors` that works inside @testset blocks.

```julia
@validate "hidden states" julia_output python_output 1f-4
```
"""
macro validate(label, a, b, atol=:(1f-4))
    quote
        @test compare_tensors($(esc(a)), $(esc(b)); atol=$(esc(atol)), label=$(esc(label)))
    end
end

end # module ValidationUtils
