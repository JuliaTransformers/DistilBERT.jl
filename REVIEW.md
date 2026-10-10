# DistilBERT.jl — Code Review & Architectural Audit

**Date:** 2026-10-09 · **Reviewed against:** Julia 1.13.1, Flux 0.14, NNlib 0.9
**Verdict:** Solid numerical core with verified HF parity, but **not yet production-ready**:
one precompilation-breaking bug in the CUDA extension, one silent device-drop bug in
precision utilities, an untrainable MLM bias, and pervasive type instability in the hot path.

---

## 1. Executive Architectural Summary

The package is a faithful, numerically-validated port of DistilBERT built on the
right backbone (`Flux.jl` + `NNlib.dot_product_attention`, SafeTensors for weights,
weakdep-based CUDA support). The architecture of a trainable transformer is sound:
immutable model structs, `Flux.@layer` for `Functors`/`Optimisers` integration, and a
clean split between configuration, layers, models, tokenizer, loading, and inference.
Parity validation infrastructure (CPU/GPU vs. PyTorch) is unusually thorough for a
0.1.0 package and is a genuine asset.

However, the implementation has not yet crossed from "correct prototype" to
"production Julia library". The central findings:

| Severity | Finding | Location |
|---|---|---|
| 🔴 **Critical** | CUDA extension **overwrites** core `predict`/`embed` methods (identical positional signatures → "Method overwriting is not permitted during Module precompilation"). The extension **fails to precompile**; every CUDA user hits an error at load time. | `ext/DistilBERTCUDAExt.jl:104,115,133,156` vs `src/inference.jl:31,61,154,186` |
| 🔴 **High** | `to_bfloat16` silently **moves GPU arrays to CPU** (`convert(Array{BFloat16}, x)`), destroying device locality and silently breaking GPU pipelines. | `src/performance.jl:105` |
| 🔴 **High** | `vocab_projector_bias` (30,522 params) is **excluded from `trainable`** → never updated by Zygote and invisible to `Flux.params`/`num_parameters`. | `src/models.jl:167` |
| 🟠 **Medium** | Hot path is type-unstable: `for block in m.transformer` iterates a `Chain`, whose `eltype` is `Any` → `Body::Any` for the entire forward pass (verified with `@code_warntype`). Every block call is a dynamic dispatch; compiler cannot inline across blocks. | `src/models.jl:34` |
| 🟠 **Medium** | `_td_available()` reaches into `Main` and hardcodes a UUID to find `TransformersDownloads` — anti-pattern (breaks precompilation/`Base.require` semantics). Should be a package extension like the CUDA one. | `src/loading.jl:126`, `src/tokenizer.jl:74` |
| 🟠 **Medium** | Tokenizer deviates from HF `BertTokenizer` on accented input (no NFKD accent stripping) and reimplements hot paths with per-character allocations. | `src/tokenizer.jl` |
| 🟡 **Low** | `unmask` silently ignores all but the **first** `[MASK]`; generic `export`s (`encode`, `tokenize`, `predict`, `embed`, `unmask`) will collide with other NLP packages; several stale docs/scripts. | `src/inference.jl:278`, `src/tokenizer.jl:4` |

What is **good** and should be preserved:

- Immutable `struct`s everywhere; `@kwdef` config; `Base.show` overrides.
- The single-pass `_make_nnlib_mask` conversion (4-D Bool, `true=keep`) computed **once per forward** and shared across all blocks — this is a real, correct optimization.
- SafeTensors-first loading with local fallback, `used_keys` tracking, and per-missing-key warnings.
- Package-extension pattern already used for CUDA — the right mechanism, just misused for `predict`.
- Numerical parity machinery (`<1e-4` FP32 vs. HF) gives the team a safety net for every refactor proposed below.

---

## 2. Key Design Pattern Enhancements

### 2.1 🔴 HIGH IMPACT — Stop overwriting `predict`/`embed` in the CUDA extension

**Current approach** (`ext/DistilBERTCUDAExt.jl`):

```julia
# src/inference.jl
function predict(model::DistilBertModel, tokenizer::WordPieceTokenizer, text::String)
    input_ids = encode(tokenizer, text)
    return model(reshape(input_ids, :, 1))
end

# ext/DistilBERTCUDAExt.jl — IDENTICAL positional signature → overwrite → precompile ERROR
function DistilBERT.predict(model::DistilBERT.DistilBertModel,
    tokenizer::DistilBERT.WordPieceTokenizer, text::String; device=Flux.gpu)
    ...
end
```

Julia kwargs do **not** participate in dispatch. The extension method overwrites the
core method, which is a hard error during extension precompilation (confirmed by
running `Pkg.test()` with CUDA loaded). CI does not catch it because the tests never
`using CUDA`.

**Idiomatic production Julia** — make the *core* functions device-agnostic by
adapting inputs to wherever the model already lives; the extension then only supplies
the device machinery (`to_gpu`, `cuda_inputs`), never overloads inference entry points:

```julia
# src/inference.jl — one method, works for CPU and CUDA alike
function predict(model::DistilBertModel, tokenizer::WordPieceTokenizer, text::String)
    input_ids = encode(tokenizer, text)
    input_matrix = reshape(input_ids, :, 1)
    return model(_transfer_like(input_matrix, model))
end

# Device transfer resolved by where the *model's parameters* live
_to_device(x, model) = x                      # CPU fallback (no CUDA loaded)
# ext/DistilBERTCUDAExt.jl
DistilBERT._to_device(x::AbstractArray, model) =
    DistilBERT.on_gpu(model) ? CuArray{Int32}(x) : x
```

If you must keep an explicit `device=` knob, make it part of the **positional**
signature so it cannot collide:

```julia
# core:  predict(model, tok, text)                    # CPU
# ext:   predict(model, tok, text, ::Flux.CUDADevice)  # different arity, no overwrite
```

Apply the same fix to `embed`. This also removes the duplicated pooling logic in the
extension (the `pooling == :cls/:mean/:max` chain exists twice — DRY violation).

### 2.2 🟠 HIGH IMPACT — Replace `Chain` with a homogeneous typed container

**Current approach** (`src/models.jl`):

```julia
struct DistilBertModel
    config::DistilBertConfig
    embeddings::Embeddings
    transformer::Chain   # eltype(Chain) === Any
end
...
for block in m.transformer      # block :: Any → dynamic dispatch × n_layers
    x = block(x; mask=nnlib_mask)
end
```

Verified: `code_warntype` reports `Body::ANY` for the entire `DistilBertModel`
call. Measured cost on CPU is modest (GEMM-bound), but it also prevents cross-layer
inlining and adds per-layer dispatch latency that dominates on GPU/small-model
workloads.

**Idiomatic production Julia** — an `NTuple` is both heterogeneous-safe (one
concrete type per `N`) and fully inferable, and `Functors`/`Zygote`/`Flux.@layer`
handle tuple fields natively:

```julia
struct DistilBertModel{N}
    config::DistilBertConfig
    embeddings::Embeddings
    transformer::NTuple{N,TransformerBlock}
end

DistilBertModel(config::DistilBertConfig) = DistilBertModel(
    config, Embeddings(config), ntuple(_ -> TransformerBlock(config), config.n_layers))
```

Because `TransformerBlock` is immutable and non-parametric in `N`, the tuple type is
`NTuple{N,TransformerBlock}` and the loop fully unrolls/inlines. Alternative (if you
prefer to stay on Flux containers): keep `Chain` but expose
`Blocks = Tuple{TransformerBlock...}` — functionally identical, more code. The tuple
is the idiomatic choice for fixed-architecture transformers.

### 2.3 🟠 HIGH IMPACT — TransformersDownloads as a real package extension

**Current approach** (`src/loading.jl`, `src/tokenizer.jl`):

```julia
_td_available() = isdefined(Main, :TransformersDownloads) ||
    haskey(Base.loaded_modules, Base.PkgId(Base.UUID("83ccf3dd-..."), "TransformersDownloads"))
```

Reaching into `Main` breaks `--compiled-modules` semantics, is worlds-unsafe, and the
UUID-lookup-by-string pattern is fragile. The mechanism you already use for CUDA is
the correct one.

**Idiomatic production Julia**:

```toml
# Project.toml
[weakdeps]
TransformersDownloads = "83ccf3dd-bfab-49d6-97f1-f49acc47b240"

[extensions]
DistilBERTCUDAExt = "CUDA"
DistilBERTTDExt = "TransformersDownloads"

[compat]
TransformersDownloads = "0.1"    # add this — currently missing entirely
```

```julia
# src/loading.jl
function _load_config_and_weights(source::String; init_random=false)
    if (ext = Base.get_extension(@__MODULE__, :DistilBERTTDExt)) !== nothing
        return ext.load_config_and_weights(source; init_random)
    end
    # existing local-only fallback path
end
```

The extension then owns every `TD.` call. Fallback for local directories stays
inline. This removes the last `Main`-coupling from the package.

### 2.4 🟠 MEDIUM IMPACT — `predict`/`embed` return-type contract (NamedTuple vs. matrix, GPU vs. CPU)

`predict(model, tok, text)` returns a CPU `Matrix`, but with a GPU model (via the
extension) it returns a `CuArray`. `embed` in the extension force-copies to CPU
(`Array(output)`) while core `embed` doesn't. Callers cannot reason about where their
data lives. Two options, in order of preference:

1. **Device follows model** (recommended, synergizes with §2.1): outputs live on the
   model's device. Document it; provide `to_cpu`/`to_gpu` for explicit movement.
2. Explicit `device` keyword in the **positional-free** signature style above.

Either way, the docstrings must state the device contract (currently they say
`Matrix{Float32}`, which is false on GPU).

### 2.5 🟠 MEDIUM IMPACT — Enumerated options instead of raw `Symbol`s

**Current approach:**

```julia
embed(..., texts; pooling::Symbol=:cls, max_length=512)
    if pooling == :cls ... elseif pooling == :mean ... elseif pooling == :max
    else error("Unknown pooling strategy: $pooling ...") end
```

A typo'd symbol (`:cls ` → `:cl s`) throws at runtime, deep in a user's loop, and
each branch duplicates the error message. The same pattern exists for
`truncation::Symbol` and `padding::Symbol` in the tokenizer.

**Idiomatic production Julia** — dispatch on a singleton-typed enum:

```julia
abstract type AbstractPooling end
struct CLSPooling  <: AbstractPooling end
struct MeanPooling <: AbstractPooling end
struct MaxPooling  <: AbstractPooling end

embed(model, tokenizer, text; pooling::Type{<:AbstractPooling}=CLSPooling) = ...

_pool(output, mask, ::Type{CLSPooling})  = cls_pooling(output)
_pool(output, mask, ::Type{MeanPooling}) = mean_pooling(output, mask)
_pool(output, mask, ::Type{MaxPooling})  = max_pooling(output, mask)
```

Misspelled options become `UndefVarError`s at parse time; adding a strategy is a new
method, not a new `elseif`. Keep `Symbol` inputs only if HF-API compatibility demands it.

### 2.6 🟡 MEDIUM-LOW IMPACT — Namespace the generic function names

`tokenize`, `encode`, `encode_batch`, `embed`, `predict`, `unmask` are exported
unqualified. Every other NLP/ML package in a user's environment (`Transformers.jl`,
`TextAnalysis.jl`, `Embeddings.jl`) exports at least one of these. Standard practice
for model-specific packages: export **types and loaders**, keep inference verbs
namespaced, e.g. `DistilBERT.encode(...)`, or adopt a package-unique prefix. At
minimum, document the collision risk; better, drop the most contested names
(`encode`, `tokenize`, `embed`, `predict`) from the export list.

### 2.7 🟡 LOW IMPACT — Deduplicate loading logic with an interface

Six near-identical `load_model(::Type{T}, ...)` methods and four near-identical
`load_weights!` methods differ only in head wiring. A `@kwdef` + a small
`has_task_head(::Type)` / `build_head!(...)` function-barrier pair would collapse
~120 lines into ~40 without losing dispatch clarity. This is a refactor for after
§2.1–2.3 land, since the extension fix touches the same files.

---

## 3. Performance Optimization Blueprint

All findings verified with `@code_warntype` / `@time` on Julia 1.13, Flux 0.14.

### 3.1 `src/models.jl` — forward passes

| Location | Issue | Fix |
|---|---|---|
| `:34` `for block in m.transformer` | `Chain` iteration: `block::Any`, `Body::ANY` for whole model | Store `NTuple{N,TransformerBlock}` (§2.2); loop becomes fully inferred/inlinable. |
| `:79` `hidden_states[:, 1, :]` | Copies a `(dim × batch)` slab | `@view hidden_states[:, 1, :]` or `selectdim(hidden_states, 2, 1)` — zero-copy; the subsequent `Dense` accepts views. |
| `:150-151` `logits[1, :, :]`, `logits[2, :, :]` | Two `(seq × batch)` copies per QA call | `(start_logits=@view(logits[1, :, :]), end_logits=@view(logits[2, :, :]))` — document that the NamedTuple aliases the output buffer. |
| `:192-198` MLM projection | `reshape` + `embed_weight' * flat` + `reshape` — all necessary for batched matmul | **Keep** — this is already the correct fused formulation. Only nit: hoist `m.distilbert.embeddings.word_embeddings.weight` lookup and the bias out of the hot closure if you later wrap in an inference loop. |
| `:167` `Flux.@layer ... trainable = (distilbert, vocab_transform, vocab_layer_norm)` | `vocab_projector_bias` untrainable and missing from `num_parameters` | Include it: `trainable = (distilbert, vocab_transform, vocab_layer_norm, vocab_projector_bias)`. |

### 3.2 `src/layers.jl` — the hot loop

| Location | Issue | Fix |
|---|---|---|
| `:85-93` `MultiHeadSelfAttention` | `q, k, v = ...` three `Dense` calls + `dot_product_attention` — already fused on CUDA via cuDNN; nothing to fix. | Annotate `@inline` on the 3-D `Dense` path only if you switch to the NTuple model (§2.2) and profiling shows benefit. |
| `:141,144` residuals | `sa_output .+ x` allocates a full tensor twice per block | Unavoidable in-place without unsafe ops; acceptable. If you need more, `NNlib.softmax!`-style fused kernels are out of scope for v1. |
| `:75` `_make_nnlib_mask` | `mask .!= 0` allocates a `BitMatrix` once per forward — **already optimal**. Keep the "convert once, share across blocks" design. | Only add a fast path: `mask isa Matrix{Bool} && all(mask)` → return `nothing`-equivalent skip (minor). |
| `:19-27` `Embeddings` | `position_embeddings(1:seq_length)` re-gathers the same `(dim, seq)` matrix every call — could be cached per `seq_length` | Optional: memoize in a `Dict{Int,Matrix{Float32}}` keyed by seq length (write-once, immutable model — safe). Low priority; skip unless profiling shows it. |

### 3.3 `src/inference.jl`

| Location | Issue | Fix |
|---|---|---|
| `:102-105` `mean_pooling` | `output .* mask_expanded` materializes a full `(dim, seq, batch)` temporary before summing | Fuse: `sum_embeddings = sum(output .* reshape(attention_mask, 1, size(attention_mask)...); dims=2) |> dropdims`. Julia's broadcast fusion already avoids the worst; writing it as a single expression guarantees one fused pass. |
| `:129-130` `max_pooling` | Three broadcast temporaries: `output .* mask`, `(1 .- mask) .* -1f9`, then `+` | One fused broadcast: `masked_output = ifelse.(mask_expanded .!= 0, output, -1f9)` or `output .* mask_expanded .+ (mask_expanded .== 0) .* -1f9` as a single expression. |
| `:282` `unmask` `NNlib.softmax(mask_logits)` | Fine (one vector). `logits[:, mask_pos, 1]` copies a `(vocab,)` vector — acceptable | `partialsortperm` already avoids full sort — good. |
| `:246` `r"\[MASK\]"i` compiled per call | Regex object re-created on every `unmask` call | Hoist to `const MASK_PATTERN = r"\[MASK\]"i` at module level. |
| `:290` `copy(input_ids)` + `map(...) do` closure | Minor; inference-time only | Acceptable. If you later batch `unmask`, switch `map` to a comprehension with concrete eltype. |

### 3.4 `src/tokenizer.jl`

| Location | Issue | Fix |
|---|---|---|
| `:110,118,127` `basic_tokenize` | Uses undocumented `buffer.size` (IOBuffer internals), `string(char)` per punct char, `take!` re-allocs | Use `position(buffer)` for emptiness; `write(io, text)` style chunk writes; or simply `split`-based with `eachmatch` over a punct regex. |
| `:134-180` `wordpiece_tokenize` | `"##" * substr_view` allocates per candidate probe — the classic WordPiece O(n²) worst case with allocation in the inner loop | Prepend once: `key = start_char > 1 ? "##" : ""` and build candidates via `SubString(token, start_byte, end_byte)` hashing. Better: since `Dict{String,Int}` hashing a `SubString` still allocates (it hashes content, no alloc actually — `hash(::SubString)` is allocation-free, so `haskey(vocab, SubString)` is fine if you avoid the `"##"*` concatenation by checking two dicts or interning). Pragmatic fix: keep a second `Dict{String,Int}` of `"##"`-prefixed lookups? No — simplest: `vocab_cont = vocab` and probe `SubString` directly, and separately probe `"##" * s` only when needed. At minimum: avoid `"##" * substr_view` for the first-token path (already avoided), and note `thisind/nextind` calls are per-probe overhead. |
| `:91-102` `load_vocab` | `strip(line)` per line OK; but vocab → `ids_to_tokens` builds a second full Dict — 2× memory for 30k entries | Acceptable (30k entries ≈ few MB). Add the `VOCAB_CACHE` from `FUTURE_IMPROVEMENTS.md` if you ever hot-load tokenizers. |
| `:248-253` `encode` padding | `while length(ids) < max_length; push!(ids, pad_id); end` — reallocation-prone | `append!(ids, fill(pad_id, max_length - length(ids)))` — single amortized growth. |
| `:320` `encode_pair` | `[fill(0, a); fill(1, b)]` — fine | Keep. |
| `:358-377` `encode_batch` | `maximum(length.(all_ids))` allocates; `padding::Symbol` silently ignores invalid values (`:foo` → `max_length`) | `target_len = maximum(length, all_ids)`; validate `padding ∈ (:longest, :max_length)` or use enums (§2.5). |

### 3.5 `src/performance.jl` & extension

| Location | Issue | Fix |
|---|---|---|
| `:105` `to_bfloat16` | `convert(Array{BFloat16}, x)` — **device-dropping bug**: `CuArray{Float32}` → CPU `Array{BFloat16}` | Use `Flux.fmap` + element-type-preserving conversion: `x isa AbstractArray ? (x isa CuArray ? CUDA.BFloat16.(x) : BFloat16.(x)) : x`, or better `oftype`-based: `map(y -> oftype(eltype-target, y), ...)`. Match `Flux.f16`'s device-preserving semantics. |
| `:163-170` `prepare_inputs` | `Int32.(input_ids)` re-allocates even when input is already `Int32` | `eltype(input_ids) === Int32 ? input_ids : Int32.(input_ids)` (same for mask). |
| `:39-44` `on_gpu` | `Flux.params(model)` traverses the whole model just to inspect one array; heuristic `!(first(ps) isa Array)` | Fine as a heuristic, but consider `Flux.fmap`-free alternative: check `model.embeddings.word_embeddings.weight`. Document that it inspects the first parameter only. |
| `:131-134` `num_parameters` | Under-counts MLM bias (see §3.1). `sum(length, Flux.params(m))` is `Any`-iterating | Use `sum(length, values(params))`; still O(1) per parameter, acceptable. |

---

## 4. Refactored Module Structure

```
DistilBERT.jl/
├── Project.toml            # + compat for TransformersDownloads, + DistilBERTTDExt
├── src/
│   ├── DistilBERT.jl       # module docstring, using's, include order, exports
│   ├── config.jl           # DistilBertConfig, show           (unchanged)
│   ├── layers.jl           # Embeddings, MultiHeadSelfAttention, FeedForward,
│   │                       #   TransformerBlock, _make_nnlib_mask (unchanged)
│   ├── models.jl           # DistilBertModel{N} + 4 task heads  (NTuple fix)
│   ├── tokenizer/
│   │   ├── tokenizer.jl    # WordPieceTokenizer struct + constructors
│   │   ├── wordpiece.jl    # basic_tokenize, wordpiece_tokenize, _detokenize
│   │   └── encoding.jl     # encode, encode_pair, encode_batch, load_vocab
│   ├── loading.jl          # load_model, load_weights!, local fallback
│   ├── inference.jl        # predict, embed, unmask, pooling fns (device-agnostic)
│   └── performance.jl      # to_gpu/to_cpu/to_half/to_bfloat16/num_parameters
├── ext/
│   ├── DistilBERTCUDAExt.jl    # to_gpu, cuda_inputs, gpu_memory_info ONLY
│   └── DistilBERTTDExt.jl      # NEW: TransformersDownloads hooks
├── test/
│   ├── runtests.jl
│   ├── test_tokenizer.jl       # incl. accent/NFC parity fixtures
│   ├── test_models.jl          # incl. gradients for ALL heads
│   ├── test_loading.jl         # local safetensors/pickle paths
│   └── test_inference.jl       # predict/embed/unmask contract tests
├── benchmark/                  # renamed from benchmarks/, + Project.toml
│   ├── benchmark.jl            # BenchmarkTools @benchmarkable suites
│   └── validate_parity.jl
├── validation/                 # keep as dev-only (CPU/GPU parity vs PyTorch)
├── docs/                       # + api.md, quickstart.md, gpu.md pages
└── REVIEW.md
```

Changes vs. today:

1. `src/tokenizer.jl` (382 lines) split into three files — tokenizer state, raw
   WordPiece algorithm, and encoding/padding policies are three separable concerns.
2. New `ext/DistilBERTTDExt.jl` removes all `Main`/UUID lookups (§2.3).
3. `benchmarks/` → `benchmark/` with its own `Project.toml` so `BenchmarkTools`
   stops being an undeclared, ad-hoc dependency of scripts.
4. Tests split by domain; GPU tests gated behind a `CUDA_AVAILABLE` flag so CI
   actually loads the extension (this would have caught §2.1 immediately).
5. `validation/` deliberately kept **outside** the test suite (depends on Python);
   add a `Makefile`/`justfile` target to run it manually.

---

## 5. Actionable Roadmap

### Phase 0 — Emergency fixes (ship before anything else)

- [x] **Fix the extension overwrite** (§2.1). Without this, the package cannot be
      used by anyone with CUDA.jl in their environment. Remove `predict`/`embed`
      methods from `DistilBERTCUDAExt`; make core methods device-agnostic.
- [x] **Fix `to_bfloat16` device drop** (§3.5) and add a regression test that
      round-trips `to_bfloat16` on a `CuArray`.
- [x] **Add `vocab_projector_bias` to `trainable`** (§3.1) and assert
      `num_parameters(mlm)` ≈ HF's published parameter count.
- [x] **CI: load the CUDA extension** (`using CUDA; using DistilBERT`) in a
      no-GPU-safe way (wrap in `CUDA.functional()` guard) so precompile failures
      become CI failures.

> Progress: Phase 0 landed in `5185998`. GPU tests ran on a CUDA-capable
> machine; extension precompiles cleanly.

### Phase 1 — Type stability & API hygiene (target: v0.2.0)

- [x] `DistilBertModel{N}` with `NTuple{N,TransformerBlock}` (§2.2); verify with
      `@inferred` tests on the forward pass (CPU and, where available, GPU).
      Extended to parameterize every layer/head struct — abstract field types
      were the remaining source of `Body::ANY`.
- [x] TransformersDownloads extension (§2.3) + `compat` entry; delete
      `_td_available` and the UUID lookups from both `loading.jl` and `tokenizer.jl`.
      `_td_available` is now `Base.get_extension`-based; `ext/DistilBERTTDExt.jl` added.
- [x] Device contract for `predict`/`embed` documented in docstrings (§2.4).
- [x] Enum-based `pooling` (§2.5) with `Base.depwarn` shim for `Symbol` inputs.
      (Tokenizer `truncation`/`padding` enums deferred to Phase 2.)
- [ ] Re-run full parity suite (CPU + GPU) after each change above — the existing
      validation scripts make these refactors safe.
      *Blocked locally: `models/` and the Python venv are not present in this
      checkout; run `validation/cpu|gpu` on a machine with weights before release.*


### Phase 2 — Tokenizer fidelity & performance

- [x] Add NFKD + accent-stripping when `do_lower_case=true` to match HF
      `BertTokenizer` exactly; parity fixtures added for `"café"`/`"naïve"`.
- [x] Apply tokenizer allocation fixes (§3.4): `position(buffer)`, single
      `append!` for padding, hoisted mask regex (Phase 0), `SubString`-based
      first-piece vocab probing.
- [~] Interning `"##"`-prefixed probes: attempted and reverted — the prefixed
      key is not contiguous in a `"##"*token` buffer, so a `SubString` cannot
      represent it. First-piece probes remain allocation-free; continuation
      probes materialise the key. A second `Dict{String,String}` index of bare
      suffixes would make continuation probes allocation-free if needed.
- [x] Multi-`[MASK]` support documented as first-mask-only in the `unmask`
      docstring.
- [x] **Bonus bug fixes found while adding multi-byte fixtures:**
      `wordpiece_tokenize` misused `thisind`/`lastindex` (byte APIs) with
      character indices, corrupting non-ASCII tokens; `encode_pair` did not
      enforce `max_length` for `only_first`/`only_second`/longest-first edge
      cases. Both fixed with tests.
- [x] Tokenizer `truncation` enums (`LongestFirst`/`OnlyFirst`/`OnlySecond`)
      with `Base.depwarn` shim; `encode_batch` validates its padding strategy.

### Phase 3 — Test & packaging hardening (production release)

- [x] Test coverage: gradients for **all four** task heads; `show` round-trips;
      truncation/padding edge cases; empty-batch rejection; **weight loading via
      synthetic HF-layout state dicts** (base + all four heads) and `load_model`
      dispatch against a real local `config.json`.
- [x] Property-based tests: config validation matrix (`dim ÷ n_heads`, ranges);
      `@inferred` forward/head checks. (Random-mask analytic pooling test still
      open.)
- [x] **Aqua.jl quality gates** (`test_all`, persistent_tasks disabled due to an
      upstream env bug): no ambiguities, no piracies, no stale deps, compat
      entries for all deps/extras, no undefined exports.
- [x] **Bonus bug fixes found while adding loading tests:**
      `_parse_distilbert_config(::Dict)` rejected `JSON.Object` returned by
      JSON.jl 1.4+, so **all local-directory loading was broken**; signature now
      accepts `AbstractDict`. Duplicated `load_model` methods collapsed into one
      generic method + `_build_model` barriers.
- [x] Benchmark suite: a dedicated `benchmarks/Project.toml` now declares
      BenchmarkTools/Flux/JSON/etc., and the README documents the
      `--project=benchmarks` workflow. (CI regression thresholds still open.)
- [x] Documentation: README GPU/device-contract section, corrected pooling API
      and benchmark instructions; docs pages cover pooling/MLM/device/precision
      APIs. Export-collision note added to README.
- [x] Decide export policy (§2.6): generic names kept for 0.1.0 compatibility;
      collision risk documented in the README (namespacing is a 1.0 breaking change).
- [ ] Git hygiene: `Manifest.toml` remains tracked (remove before release if the
      project adopts the library convention). Duplicated `using Unicode` removed.

### Definition of done (production)

- [ ] `Pkg.test()` green on 3 OS × (1.10, latest, pre) — including extension precompile.
- [ ] `@inferred model(x)` and `@inferred model(x; mask)` pass.
- [ ] Parity suite green (`<1e-4` FP32) **after** the NTuple refactor.
- [ ] `@allocated model(x)` reduced or equal vs. current baseline, with zero
      dynamic dispatches in the block loop (visible as concrete `Tuple` inference).
- [ ] All public functions have docstrings with return-type and device contracts;
      all four heads gradient-tested.
