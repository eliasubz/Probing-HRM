"""Identify SAE features that are specific to individual Sudoku digits.

This is the cell-level twin of analyze_sae_cycles.py. Instead of asking
"which SAE features fire on which ACT cycle?", we ask:
"which SAE features fire specifically when a Sudoku cell contains digit N?"

Input: structured_batch_*.pt files from probing_sudoku_structured.py
       (each contains activations + per-cell labels paired together)
       sae_weights.pt from train_sae_cluster.py

Output:
  top_digit_specific_features.csv  -- ranked (digit, feature, specificity) table
  digit_feature_heatmap.png         -- 9 digits x top-K features firing frequency
  avg_active_features_by_digit.png  -- how many features fire per digit on average

Label encoding (from probing_sudoku_structured.py manifest):
  blank = 1, digit 1..9 = labels 2..10
  token 0 is the puzzle embedding token; Sudoku cells are tokens 1..81.
  We ignore blank cells so the SAE only sees cells with a known solution digit.
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory of structured_batch_*.pt files from "
                        "probing_sudoku_structured.py.")
    p.add_argument("--sae_weights", type=str, required=True,
                   help="Path to sae_weights.pt from train_sae_cluster.py.")
    p.add_argument("--output_dir", type=str, default="results/cell_analysis")
    p.add_argument("--expansion_factor", type=int, default=4)
    p.add_argument("--layer_key", type=str,
                   default="model_inner_H_level_layers_0",
                   help="Key inside payload['layers'] to use (dots replaced with _).")
    p.add_argument("--act_cycle", type=int, default=-1,
                   help="Which ACT cycle to read from the structured tensor. "
                        "-1 means the last available cycle.")
    p.add_argument("--inner_call", type=int, default=-1,
                   help="Which inner hook call to use. -1 = last.")
    p.add_argument("--max_files", type=int, default=0,
                   help="Cap on number of batch files to read (0 = all).")
    p.add_argument("--batch_size", type=int, default=4096,
                   help="Token batch size for SAE forward passes.")
    p.add_argument("--top_k", type=int, default=20,
                   help="Top features to report per digit.")
    p.add_argument("--blanks_only", action="store_true",
                   help="If set, restrict to originally-blank cells only "
                        "(given cells are trivially identifiable from the input).")
    return p.parse_args()


def load_sae(weights_path, input_dim, expansion_factor, device):
    sae = SparseAutoencoder(input_dim=input_dim, expansion_factor=expansion_factor)
    sae.load_state_dict(torch.load(weights_path, map_location="cpu"))
    sae.to(device)
    sae.eval()
    return sae


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    files = sorted(glob.glob(os.path.join(args.data_dir, "structured_batch_*.pt")))
    if not files:
        raise FileNotFoundError(f"No structured_batch_*.pt files in {args.data_dir}")
    if args.max_files > 0:
        files = files[:args.max_files]
    print(f"Found {len(files)} batch files")

    # Probe one file to get input_dim and resolve -1 indices.
    sample = torch.load(files[0], map_location="cpu")
    layer_keys = list(sample["layers"].keys())
    if args.layer_key not in sample["layers"]:
        raise KeyError(f"layer_key '{args.layer_key}' not in payload. "
                       f"Available: {layer_keys}")
    sample_tensor = sample["layers"][args.layer_key]
    # shape: [act_cycle, inner_call, batch, seq, hidden]
    n_cycles, n_inner, _, seq_len, hidden_size = sample_tensor.shape
    act_cycle_idx = args.act_cycle if args.act_cycle >= 0 else n_cycles - 1
    inner_call_idx = args.inner_call if args.inner_call >= 0 else n_inner - 1
    print(f"Tensor shape: {list(sample_tensor.shape)}")
    print(f"Using act_cycle={act_cycle_idx}, inner_call={inner_call_idx}, "
          f"hidden_size={hidden_size}")

    sae = load_sae(args.sae_weights, hidden_size, args.expansion_factor, device)
    num_features = sae.latent_dim
    num_digits = 9  # digits 1..9; label encoding: digit d -> label d+1

    # Accumulators: shape [num_digits, num_features]
    token_counts = np.zeros(num_digits, dtype=np.int64)
    firing_counts = np.zeros((num_digits, num_features), dtype=np.float64)
    activation_sums = np.zeros((num_digits, num_features), dtype=np.float64)
    active_counts = np.zeros(num_digits, dtype=np.float64)

    for file_idx, path in enumerate(files):
        payload = torch.load(path, map_location="cpu")
        labels = payload["labels"]          # [batch, seq_len] -- values 1..10
        given_mask = payload["given_mask"]  # [batch, seq_len] bool
        states = payload["layers"][args.layer_key]
        # [act_cycle, inner_call, batch, seq, hidden] -> [batch, seq, hidden]
        states = states[act_cycle_idx, inner_call_idx].to(torch.float32)

        # Sudoku cells are tokens 1..81 (token 0 is the puzzle embedding).
        # labels[:, 0] is the puzzle token; skip it.
        cell_states = states[:, 1:, :]   # [batch, 81, hidden]
        cell_labels = labels[:, 1:]      # [batch, 81]
        cell_given  = given_mask[:, 1:]  # [batch, 81]

        # Flatten to (N_cells, hidden)
        B, S, H = cell_states.shape
        cell_states_flat = cell_states.reshape(-1, H)   # [B*81, H]
        cell_labels_flat = cell_labels.reshape(-1)       # [B*81]
        cell_given_flat  = cell_given.reshape(-1)        # [B*81]

        # Keep only filled cells (label in 2..10 => digit 1..9)
        filled_mask = (cell_labels_flat >= 2) & (cell_labels_flat <= 10)
        if args.blanks_only:
            filled_mask = filled_mask & ~cell_given_flat

        cell_states_flat = cell_states_flat[filled_mask]
        cell_labels_flat = cell_labels_flat[filled_mask]

        if cell_states_flat.shape[0] == 0:
            continue

        # Run through SAE in mini-batches
        all_latents = []
        with torch.no_grad():
            for start in range(0, cell_states_flat.shape[0], args.batch_size):
                chunk = cell_states_flat[start:start + args.batch_size].to(device)
                _, latent = sae(chunk)
                all_latents.append(latent.cpu())
        all_latents = torch.cat(all_latents, dim=0)  # [N, num_features]

        # Accumulate per-digit statistics
        for digit_idx in range(num_digits):
            digit_label = digit_idx + 2  # label encoding: digit d -> d+1, so d=1 -> 2
            mask = (cell_labels_flat == digit_label)
            if mask.sum() == 0:
                continue
            latents_d = all_latents[mask]
            active = latents_d > 0
            token_counts[digit_idx] += mask.sum().item()
            active_counts[digit_idx] += active.sum(dim=1).float().sum().item()
            firing_counts[digit_idx] += active.sum(dim=0).numpy()
            activation_sums[digit_idx] += latents_d.sum(dim=0).numpy()

        if (file_idx + 1) % 10 == 0:
            print(f"Processed {file_idx + 1}/{len(files)} files | "
                  f"tokens so far: {token_counts.sum():,}")

    print(f"\nTotal cell-tokens per digit: {token_counts}")
    safe_counts = np.maximum(token_counts[:, None], 1)
    firing_freq = firing_counts / safe_counts          # [9, num_features]
    mean_activation = activation_sums / safe_counts   # [9, num_features]
    avg_active = active_counts / np.maximum(token_counts, 1)

    # ---- specificity = how much MORE a feature fires for this digit vs overall ----
    global_freq = firing_counts.sum(axis=0) / max(token_counts.sum(), 1)
    specificity = firing_freq - global_freq[None, :]

    # ---- CSV: top-K features per digit ----------------------------------------
    top_rows = []
    for digit_idx in range(num_digits):
        digit = digit_idx + 1
        top_features = np.argsort(specificity[digit_idx])[::-1][:args.top_k]
        for rank, feat in enumerate(top_features, start=1):
            top_rows.append({
                "digit": digit,
                "rank": rank,
                "feature": int(feat),
                "digit_firing_freq": float(firing_freq[digit_idx, feat]),
                "global_firing_freq": float(global_freq[feat]),
                "specificity": float(specificity[digit_idx, feat]),
                "digit_mean_activation": float(mean_activation[digit_idx, feat]),
            })
    csv_path = os.path.join(args.output_dir, "top_digit_specific_features.csv")
    write_csv(csv_path, top_rows,
              ["digit", "rank", "feature", "digit_firing_freq",
               "global_firing_freq", "specificity", "digit_mean_activation"])
    print(f"Saved {csv_path}")

    # ---- Plot 1: heatmap (top-K most digit-varying features) -------------------
    feature_score = specificity.max(axis=0) - specificity.min(axis=0)
    heatmap_features = np.argsort(feature_score)[::-1][:args.top_k]
    heatmap = firing_freq[:, heatmap_features].T   # [top_k, 9]
    digit_labels = [str(d) for d in range(1, 10)]

    fig, ax = plt.subplots(figsize=(10, max(6, args.top_k * 0.2)))
    im = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Firing frequency")
    ax.set_xlabel("Sudoku digit")
    ax.set_ylabel("SAE feature")
    ax.set_xticks(range(9))
    ax.set_xticklabels(digit_labels)
    ax.set_yticks(range(len(heatmap_features)))
    ax.set_yticklabels(heatmap_features)
    ax.set_title(f"Top {len(heatmap_features)} digit-varying SAE features")
    plt.tight_layout()
    heatmap_path = os.path.join(args.output_dir, "digit_feature_heatmap.png")
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)
    print(f"Saved {heatmap_path}")

    # ---- Plot 2: average active features per digit -----------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 10), avg_active, color="steelblue")
    ax.set_xlabel("Sudoku digit")
    ax.set_ylabel("Avg active SAE features per cell")
    ax.set_xticks(range(1, 10))
    ax.set_title("SAE sparsity per Sudoku digit")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    sparsity_path = os.path.join(args.output_dir, "avg_active_features_by_digit.png")
    fig.savefig(sparsity_path, dpi=180)
    plt.close(fig)
    print(f"Saved {sparsity_path}")

    # ---- Summary: best feature per digit ---------------------------------------
    print("\nBest SAE feature per digit (highest specificity):")
    print(f"{'Digit':>6} {'Feature':>8} {'Specificity':>12} {'Digit freq':>11} {'Global freq':>12}")
    for digit_idx in range(num_digits):
        best = int(np.argmax(specificity[digit_idx]))
        print(f"{digit_idx+1:>6} {best:>8} "
              f"{specificity[digit_idx, best]:>12.4f} "
              f"{firing_freq[digit_idx, best]:>11.4f} "
              f"{global_freq[best]:>12.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
