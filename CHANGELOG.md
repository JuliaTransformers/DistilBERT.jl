# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-09

Production-readiness pass. **Breaking changes** are marked with ⚠️.

### Fixed

- CUDA extension overwrote core `predict`/`embed` methods, breaking extension
  precompilation for every CUDA user. The extension now only provides
  CUDA-specific helpers; core inference is device-agnostic.
- `to_bfloat16` silently moved GPU arrays to the CPU and referenced an
  undefined `BFloat16` binding at runtime.
- GPU position embeddings used a CPU index range against GPU weights,
  triggering scalar-indexing errors.
- `wordpiece_tokenize` mixed byte and character indices, corrupting multi-byte
  (non-ASCII) tokens.
- `encode_pair` did not enforce `max_length` for `only_first`/`only_second` and
  edge cases of longest-first truncation.
- Local model directories failed to load with JSON.jl ≥ 1.4
  (`JSON.Object` was not accepted by the config parser).
- `vocab_projector_bias` was excluded from the MLM trainables, so it received
  no gradients and was missing from `num_parameters`.
- `DistilBertConfig(dim=10, n_heads=3)` silently produced a wrong head size;
  the config constructor now validates divisibility and value ranges.

### Changed

- ⚠️ Requires **Julia ≥ 1.12**; `BFloat16s` is no longer a dependency
  (`Core.BFloat16` is used instead, which is the same type CUDA.jl uses).
- ⚠️ `DistilBertModel` stores transformer blocks as a concrete
  `NTuple{N,TransformerBlock}` with fully parameterized layer structs; the
  forward pass is now type-stable (`@inferred`) end-to-end.
- ⚠️ Sequence-classification and QA heads return views
  (`selectdim`/`@view`) instead of copies.
- ⚠️ `do_lower_case=true` now strips accents (NFKD, category `Mn`) to match
  HuggingFace `BertTokenizer`.
- ⚠️ `embed(...; pooling=...)` accepts `CLSPooling`/`MeanPooling`/`MaxPooling`
  types; `Symbol`s (`:cls`, `:mean`, `:max`) are deprecated but still work.
- ⚠️ `encode_pair(...; truncation=...)` accepts `LongestFirst`/`OnlyFirst`/
  `OnlySecond`; `Symbol`s are deprecated but still work.
- Device handling is backend-agnostic via `GPUArraysCore.AbstractGPUArray`;
  `to_gpu` uses `Flux.gpu`, so AMDGPU and Metal work through Flux's backend
  selection (only CUDA is covered by the test suite).
- `WordPieceTokenizer(source)` accepts a vocab file, a model directory, or an
  HF Hub ID; the previously documented directory/Hub forms now exist.
- TransformersDownloads integration is a real package extension
  (`DistilBERTTDExt`) instead of `Main`/UUID lookups.
- `load_model` methods collapsed into one generic method with `_build_model`
  barrier functions.
- `Manifest.toml` is no longer tracked.

### Added

- Tests: CUDA extension load, GPU predict/embed, bf16 device preservation,
  gradients for all four task heads, config validation, type-stability
  (`@inferred`), tokenizer fidelity (accents, multi-byte), weight loading from
  synthetic HF state dicts, `load_model` dispatch, edge cases.
- [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) quality gates in the test
  suite.
- Dedicated `benchmarks/Project.toml` environment.
- README GPU/device contract section and export-collision note.

## [0.1.0] - 2025-03

- Initial release: DistilBERT inference, WordPiece tokenizer, SafeTensors/Pickle
  loading, task heads (sequence/token classification, QA, masked LM), CUDA
  extension, and Python parity validation scripts.

[0.2.0]: https://github.com/JuliaTransformers/DistilBERT.jl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/JuliaTransformers/DistilBERT.jl/releases/tag/v0.1.0
