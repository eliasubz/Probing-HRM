import argparse
import csv
import glob
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, expansion_factor=4):
        super().__init__()
        self.latent_dim = input_dim * expansion_factor
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        x_centered = x - self.decoder_bias
        latent = self.relu(self.encoder(x_centered))
        reconstructed = self.decoder(latent) + self.decoder_bias
        return reconstructed, latent


def parse_cycle(path):
    match = re.search(r"cycle_(\d+)", os.path.basename(path))
    if match is None:
        raise ValueError(f"Could not parse cycle from {path}")
    return int(match.group(1))


def load_sae(weights_path, input_dim, expansion_factor, device):
    sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor)
    sae.load_state_dict(torch.load(weights_path, map_location="cpu"))
    sae.to(device)
    sae.eval()
    return sae


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="h_sae_cycle_analysis")
    parser.add_argument("--layer_filter", type=str, default="H_level")
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_tokens_per_file", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = sorted(glob.glob(os.path.join(args.data_dir, f"states_layer_*{args.layer_filter}*.pt")))
    if not files:
        raise FileNotFoundError(f"No state files found in {args.data_dir} for {args.layer_filter}")
    if args.max_files > 0 and len(files) > args.max_files:
        files = random.sample(files, args.max_files)

    sample = torch.load(files[0], map_location="cpu")
    input_dim = sample.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = load_sae(args.weights, input_dim, args.expansion_factor, device)

    num_features = sae.latent_dim
    max_cycle = max(parse_cycle(path) for path in files)
    num_cycles = max_cycle + 1

    token_counts = np.zeros(num_cycles, dtype=np.int64)
    active_counts = np.zeros(num_cycles, dtype=np.float64)
    firing_counts = np.zeros((num_cycles, num_features), dtype=np.float64)
    activation_sums = np.zeros((num_cycles, num_features), dtype=np.float64)

    print(f"Analyzing {len(files)} files")
    print(f"Input dim: {input_dim}, SAE features: {num_features}, cycles: {num_cycles}")

    with torch.no_grad():
        for file_idx, path in enumerate(files):
            cycle = parse_cycle(path)
            states = torch.load(path, map_location="cpu").to(torch.float32)
            states = states.view(-1, states.shape[-1])
            if args.max_tokens_per_file > 0 and states.shape[0] > args.max_tokens_per_file:
                idx = torch.randperm(states.shape[0])[:args.max_tokens_per_file]
                states = states[idx]

            token_counts[cycle] += states.shape[0]
            for start in range(0, states.shape[0], args.batch_size):
                batch = states[start:start + args.batch_size].to(device)
                _, latent = sae(batch)
                active = latent > 0
                active_counts[cycle] += active.sum(dim=1).float().sum().item()
                firing_counts[cycle] += active.sum(dim=0).cpu().numpy()
                activation_sums[cycle] += latent.sum(dim=0).cpu().numpy()

            if (file_idx + 1) % 25 == 0:
                print(f"Processed {file_idx + 1}/{len(files)} files")

    safe_counts = np.maximum(token_counts[:, None], 1)
    firing_freq = firing_counts / safe_counts
    mean_activation = activation_sums / safe_counts
    avg_active_per_token = active_counts / np.maximum(token_counts, 1)

    rows = [
        {
            "cycle": cycle,
            "tokens": int(token_counts[cycle]),
            "avg_active_features_per_token": float(avg_active_per_token[cycle]),
        }
        for cycle in range(num_cycles)
    ]
    write_csv(
        os.path.join(args.output_dir, "avg_active_features_by_cycle.csv"),
        rows,
        ["cycle", "tokens", "avg_active_features_per_token"],
    )

    plt.figure(figsize=(8, 4))
    plt.plot(range(num_cycles), avg_active_per_token, marker="o")
    plt.xlabel("ACT cycle")
    plt.ylabel("Average active SAE features per token")
    plt.title("SAE sparsity across HRM reasoning cycles")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "avg_active_features_by_cycle.png"), dpi=180)
    plt.close()

    global_freq = firing_counts.sum(axis=0) / max(token_counts.sum(), 1)
    specificity = firing_freq - global_freq[None, :]
    top_rows = []
    for cycle in range(num_cycles):
        top_features = np.argsort(specificity[cycle])[::-1][:args.top_k]
        for rank, feature in enumerate(top_features, start=1):
            top_rows.append(
                {
                    "cycle": cycle,
                    "rank": rank,
                    "feature": int(feature),
                    "cycle_firing_frequency": float(firing_freq[cycle, feature]),
                    "global_firing_frequency": float(global_freq[feature]),
                    "specificity": float(specificity[cycle, feature]),
                    "cycle_mean_activation": float(mean_activation[cycle, feature]),
                }
            )
    write_csv(
        os.path.join(args.output_dir, "top_cycle_specific_features.csv"),
        top_rows,
        [
            "cycle",
            "rank",
            "feature",
            "cycle_firing_frequency",
            "global_firing_frequency",
            "specificity",
            "cycle_mean_activation",
        ],
    )

    feature_score = specificity.max(axis=0) - specificity.min(axis=0)
    heatmap_features = np.argsort(feature_score)[::-1][:args.top_k]
    heatmap = firing_freq[:, heatmap_features].T

    plt.figure(figsize=(10, max(6, args.top_k * 0.14)))
    plt.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Firing frequency")
    plt.xlabel("ACT cycle")
    plt.ylabel("SAE feature")
    plt.xticks(range(num_cycles), range(num_cycles))
    plt.yticks(range(len(heatmap_features)), heatmap_features)
    plt.title(f"Top {len(heatmap_features)} cycle-varying SAE features")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "feature_cycle_heatmap.png"), dpi=180)
    plt.close()

    np.save(os.path.join(args.output_dir, "firing_frequency_by_cycle.npy"), firing_freq)
    np.save(os.path.join(args.output_dir, "mean_activation_by_cycle.npy"), mean_activation)

    print(f"Saved analysis to {args.output_dir}")
    print("Top cycle-specific CSV:", os.path.join(args.output_dir, "top_cycle_specific_features.csv"))
    print("Heatmap:", os.path.join(args.output_dir, "feature_cycle_heatmap.png"))


if __name__ == "__main__":
    main()
