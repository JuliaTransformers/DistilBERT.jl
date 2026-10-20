# Parity fixtures

Hermetic golden data for `test/test_parity.jl`: a tiny deterministic DistilBERT
(2 layers, dim 32, vocab 31) plus HuggingFace reference outputs, committed to
`test/fixtures/tiny-distilbert/`.

The Julia test suite compares tokenization, hidden states, and MLM logits
against these files with `atol = 1e-5` — no Python and no network required.

## Regenerating

Use the pinned environment (`requirements.lock`):

```bash
uv venv --python 3.12 /tmp/distilbert-parity
uv pip install --python /tmp/distilbert-parity/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python /tmp/distilbert-parity/bin/python -r validation/fixtures/requirements.lock
/tmp/distilbert-parity/bin/python validation/fixtures/generate.py
```

Then review and commit the diff. A nightly CI job (`validation.yml`)
regenerates the fixtures with the locked versions and fails if the committed
files drift.

## Live full-size validation

The committed fixture covers the math path hermetically. For full-size models
(`distilbert-base-uncased`, GPU, Float16), use the scripts in
`validation/cpu/` and `validation/gpu/` (they require a Python environment with
`torch`/`transformers` and a local model directory).
