#!/usr/bin/env python3
"""Benchmark and output fill-mask results for parity comparison."""
import torch, time, json, sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/big"

print(f"Loading model from {model_path}...")
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForMaskedLM.from_pretrained(model_path)
model.eval()

text = "Hello I'm a [MASK] model."
print(f'Input: "{text}"')

inputs = tokenizer(text, return_tensors="pt")
mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()

# Warmup
with torch.no_grad():
    _ = model(**inputs)

# Benchmark
times = []
for _ in range(20):
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    times.append(time.perf_counter() - t0)

mean_ms = sum(times) / len(times) * 1000
min_ms = min(times) * 1000
max_ms = max(times) * 1000

print(f"\nPython DistilBertForMaskedLM benchmark (20 runs):")
print(f"  Min:  {min_ms:.1f} ms")
print(f"  Mean: {mean_ms:.1f} ms")
print(f"  Max:  {max_ms:.1f} ms")

# Get top-5 predictions
with torch.no_grad():
    logits = model(**inputs).logits

mask_logits = logits[0, mask_idx]
probs = torch.softmax(mask_logits, dim=0)
top5 = torch.topk(probs, 5)

print(f"\nPython top-5 predictions for [MASK]:")
results = []
for score, token_id in zip(top5.values, top5.indices):
    token = tokenizer.decode([token_id.item()]).strip()
    filled = text.replace("[MASK]", token)
    print(f"  {score.item():.4f}  {token}  =>  {filled}")
    results.append({
        "score": float(score.item()),
        "token": token,
        "token_id": int(token_id.item())
    })

# Output JSON for parity comparison
print("\n__PARITY_JSON__" + json.dumps(results))
