import torch
from safetensors.torch import save_file
import os

files_dir = "files"
bin_path = os.path.join(files_dir, "pytorch_model.bin")
safetensors_path = os.path.join(files_dir, "model.safetensors")

if os.path.exists(safetensors_path):
    print(f"{safetensors_path} already exists.")
else:
    print(f"Loading {bin_path}...")
    state_dict = torch.load(bin_path, map_location="cpu")
    print(f"Saving to {safetensors_path}...")
    save_file(state_dict, safetensors_path)
    print("Conversion complete.")
