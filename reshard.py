#!/usr/bin/env python3
"""
Reshard a single safetensors model file into multiple shards.

Usage:
    python reshard_single.py <input_dir> <output_dir> [--max-shard-size SIZE_IN_GB]

Example:
    python reshard_single.py ./my_model ./my_model_resharded --max-shard-size 20
"""

import json
import os
import shutil
import argparse
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import torch

def get_tensor_size(tensor):
    """Return estimated memory size of tensor in bytes (assuming float16)."""
    return tensor.numel() * 2

def main():
    parser = argparse.ArgumentParser(description="Reshard a single safetensors model file.")
    parser.add_argument("input_dir", help="Directory containing the original model files (including model.safetensors).")
    parser.add_argument("output_dir", help="Directory where resharded files will be saved.")
    parser.add_argument("--max-shard-size", type=float, default=20, help="Maximum shard size in GB (default: 20).")
    args = parser.parse_args()

    max_shard_bytes = int(args.max_shard_size * 1024 * 1024 * 1024)
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Paths
    model_file = os.path.join(input_dir, "model.safetensors")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Expected {model_file} not found.")

    print(f"Loading metadata from {model_file} ...")
    with safe_open(model_file, framework="pt") as f:
        keys = list(f.keys())
        # We'll load tensors one by one later to save memory, but first get metadata.
        # For memory efficiency, we'll load tensors sequentially while writing shards.
        # We need to know sizes to plan sharding. We can get shape/dtype from metadata without loading data.
        # safetensors doesn't provide size without loading? Actually, we can compute size from shape and dtype.
        # Let's first read all tensor metadata (shape, dtype) without loading data.
        # Unfortunately, safe_open doesn't give a way to get shape without loading. We'll have to load all tensors,
        # but that might be heavy. Alternative: use the underlying file to read metadata only.
        # For simplicity, we'll assume the user has enough memory to load all tensors (common in cloud instances).
        # We'll load all tensors into a dict.
        print("Loading all tensors into memory (this may take a while and use a lot of RAM)...")
        tensors = {}
        for key in tqdm(keys, desc="Loading tensors"):
            tensors[key] = f.get_tensor(key)

    # Now we have all tensors in memory. Let's plan sharding.
    print("Planning sharding...")
    # Sort tensors by name to keep deterministic order (optional)
    sorted_items = sorted(tensors.items(), key=lambda x: x[0])

    new_weight_map = {}
    current_shard = {}
    current_size = 0
    shard_idx = 1
    shards_data = []  # list of (shard_num, tensor_dict) for later saving

    for name, tensor in tqdm(sorted_items, desc="Partitioning"):
        tensor_size = get_tensor_size(tensor)
        if current_size + tensor_size > max_shard_bytes and current_shard:
            # Save current shard
            shards_data.append((shard_idx, current_shard))
            shard_idx += 1
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_size

    # Last shard
    if current_shard:
        shards_data.append((shard_idx, current_shard))

    total_shards = shard_idx
    print(f"Will create {total_shards} shards.")

    # Now save each shard and build weight map
    print("Saving shards...")
    for shard_num, shard_dict in tqdm(shards_data, desc="Writing shards"):
        shard_filename = f"model-{shard_num:05d}-of-{total_shards:05d}.safetensors"
        shard_path = os.path.join(output_dir, shard_filename)
        save_file(shard_dict, shard_path)
        for name in shard_dict:
            new_weight_map[name] = shard_filename

    # Generate index file
    index = {
        "metadata": {
            "total_size": sum(get_tensor_size(t) for t in tensors.values())
        },
        "weight_map": new_weight_map
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Copy other essential files (config, tokenizer, etc.) from input_dir to output_dir
    print("Copying other model files...")
    for filename in os.listdir(input_dir):
        if filename == "model.safetensors" or filename.endswith(".safetensors"):
            continue  # skip original model file and any other .safetensors (should be none)
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    print(f"Resharding complete. Output directory: {output_dir}")
    print(f"Model index file: {index_path}")

if __name__ == "__main__":
    main()
