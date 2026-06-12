"""Activation steering for HRM.

This is the causal counterpart to the read-only probing in `probing_analysis.py`.
Instead of copying a hidden state out, we add a steering vector `v` to the
H-level residual stream during the forward pass and measure how the model's
behaviour changes:

    h  <-  h  +  alpha * v

The steering direction is, by default, a single SAE feature direction (a column
of the trained SAE decoder, which is unit-normalised at train time). Because the
H-level state is the recurrent carry that HRM reuses across ACT steps, a nudge on
one step propagates into the model's later reasoning.

One direction goes in; several effects come out. For each value of `alpha` we
record a *broad* readout so a single sweep tells us whether the direction moves:
  - overall behaviour       -> solve_accuracy
  - the ACT / "cycle" axis   -> mean_act_steps (steps taken before halting)
  - specific Sudoku cells    -> frac_cells_changed / mean_logit_shift vs alpha=0

A random unit vector of the same norm is the mandatory control: if the real
direction moves the readouts and the random one does not, the effect is causal
and not just "any large perturbation breaks the model".
"""

import argparse
import csv
import os

os.environ["DISABLE_COMPILE"] = "1"

import torch
import torch.nn as nn
import yaml

from pretrain import PretrainConfig, create_dataloader, init_train_state


# --------------------------------------------------------------------------- #
# SAE definition (kept in sync with probing/train_sae_cluster.py so we can load
# the saved state_dict without importing the training module).
# --------------------------------------------------------------------------- #
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


def get_args():
    p = argparse.ArgumentParser(description="Activation steering for HRM")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained checkpoint (same as probing_analysis.py).")
    p.add_argument("--output_dir", type=str, default="results/steering")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_batches", type=int, default=10,
                   help="Number of eval batches per alpha (-1 for all).")
    p.add_argument("--target_layer", type=str,
                   default="model.inner.H_level.layers.0",
                   help="Module whose output gets steered (the H-level residual stream).")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # Steering direction
    p.add_argument("--vector_source", choices=["sae", "random"], default="sae")
    p.add_argument("--sae_weights", type=str, default=None,
                   help="Path to sae_weights.pt (required for --vector_source sae).")
    p.add_argument("--expansion_factor", type=int, default=4)
    p.add_argument("--feature_id", type=int, default=None,
                   help="SAE feature (decoder column) to steer with. If omitted, "
                        "auto-picked from --cycle_features_csv or --digit_features_csv.")
    p.add_argument("--cycle_features_csv", type=str, default=None,
                   help="top_cycle_specific_features.csv from analyze_sae_cycles.py.")
    p.add_argument("--digit_features_csv", type=str, default=None,
                   help="top_digit_specific_features.csv from analyze_sae_cells.py. "
                        "Use with --steer_digit to pick the top feature for that digit.")
    p.add_argument("--steer_digit", type=int, default=None,
                   help="Which Sudoku digit (1-9) to pick the top feature for when "
                        "using --digit_features_csv.")

    # Sweep
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-8, -4, -2, 0, 2, 4, 8])
    p.add_argument("--inject_steps", choices=["all", "early", "late"], default="all",
                   help="Which ACT steps to steer (early/late tests recurrent amplification).")
    p.add_argument("--early_cutoff", type=int, default=2,
                   help="ACT steps < cutoff are 'early', >= cutoff are 'late'.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Steering direction
# --------------------------------------------------------------------------- #
def pick_feature_id(args):
    if args.feature_id is not None:
        return args.feature_id

    if args.digit_features_csv is not None:
        with open(args.digit_features_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if args.steer_digit is not None:
            rows = [r for r in rows if int(r["digit"]) == args.steer_digit]
            if not rows:
                raise ValueError(f"No rows for digit {args.steer_digit} in {args.digit_features_csv}")
        best = max(rows, key=lambda r: float(r["specificity"]))
        fid = int(best["feature"])
        print(f"Auto-selected SAE feature {fid} "
              f"(digit {best['digit']}, specificity {float(best['specificity']):.4f}).")
        return fid

    if args.cycle_features_csv is not None:
        with open(args.cycle_features_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        best = max(rows, key=lambda r: float(r["specificity"]))
        fid = int(best["feature"])
        print(f"Auto-selected SAE feature {fid} "
              f"(cycle {best['cycle']}, specificity {float(best['specificity']):.4f}).")
        return fid

    raise ValueError("Provide --feature_id, --digit_features_csv, or --cycle_features_csv.")


def build_vector(args, hidden_size, device):
    """Return a unit-norm steering direction of shape [hidden_size]."""
    if args.vector_source == "random":
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        v = torch.randn(hidden_size, generator=g)
        meta = {"vector_source": "random", "feature_id": -1}
    else:
        if args.sae_weights is None:
            raise ValueError("--sae_weights is required for --vector_source sae")
        sae = SparseAutoencoder(input_dim=hidden_size, expansion_factor=args.expansion_factor)
        sae.load_state_dict(torch.load(args.sae_weights, map_location="cpu"))
        fid = pick_feature_id(args)
        # decoder.weight is [input_dim, latent_dim]; column fid is feature fid's
        # direction in activation space (unit-norm by SAE training convention).
        v = sae.decoder.weight.data[:, fid].clone()
        meta = {"vector_source": "sae", "feature_id": fid}

    v = v / v.norm()  # enforce unit norm so alpha is the only magnitude knob
    return v.to(device), meta


# --------------------------------------------------------------------------- #
# Steered forward pass
# --------------------------------------------------------------------------- #
class Steerer:
    """Holds the live (alpha, step) state and the forward hook."""
    def __init__(self, module, vector, inject_steps, early_cutoff):
        self.vector = vector
        self.inject_steps = inject_steps
        self.early_cutoff = early_cutoff
        self.alpha = 0.0
        self.act_step = 0
        self.handle = module.register_forward_hook(self._hook)

    def _active(self):
        if self.alpha == 0.0:
            return False
        if self.inject_steps == "all":
            return True
        if self.inject_steps == "early":
            return self.act_step < self.early_cutoff
        return self.act_step >= self.early_cutoff  # "late"

    def _hook(self, module, inp, out):
        if not self._active():
            return None  # leave output unchanged
        h = out[0] if isinstance(out, tuple) else out
        steered = h + self.alpha * self.vector.to(h.dtype)
        if isinstance(out, tuple):
            return (steered,) + tuple(out[1:])
        return steered

    def remove(self):
        self.handle.remove()


def run_alpha(model, loader, steerer, alpha, args):
    """Run the eval loop at a given alpha; return aggregate readouts and raw logits."""
    steerer.alpha = alpha
    total_correct = total_puzzles = 0
    total_steps = total_seqs = 0
    logits_store = []
    labels_store = []

    with torch.inference_mode():
        for batch_idx, (_set, batch, gbs) in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.device(args.device):
                carry = model.initial_carry(batch)

            steerer.act_step = 0
            final_logits = None
            while True:
                carry, _, _, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=["logits"]
                )
                final_logits = preds["logits"] if "logits" in preds else final_logits
                steerer.act_step += 1
                if all_finish:
                    break

            steps_taken = carry.steps.float().sum().item() if hasattr(carry, "steps") else steerer.act_step
            total_steps += steps_taken
            total_seqs += batch["inputs"].shape[0]

            if final_logits is not None and "labels" in batch:
                pred_tokens = final_logits.argmax(dim=-1)
                labels = batch["labels"][:gbs]
                pred_tokens = pred_tokens[:gbs]
                # Exact-match accuracy over non-ignored positions.
                mask = labels != -100
                correct = ((pred_tokens == labels) | ~mask).all(dim=-1)
                total_correct += correct.sum().item()
                total_puzzles += correct.shape[0]
                logits_store.append(final_logits[:gbs].float().cpu())
                labels_store.append(labels.cpu())

    accuracy = total_correct / max(total_puzzles, 1)
    mean_steps = total_steps / max(total_seqs, 1)
    logits = torch.cat(logits_store, dim=0) if logits_store else None
    return {"accuracy": accuracy, "mean_act_steps": mean_steps}, logits


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    torch.manual_seed(args.seed)
    print(f"Device: {args.device}")

    # ----- config + data + model (mirrors probing_analysis.py) -----
    config_path = os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config at {config_path}.")
    with open(config_path) as f:
        config = PretrainConfig(**yaml.safe_load(f))
    config.global_batch_size = args.batch_size

    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1,
    )

    train_state = init_train_state(config, eval_metadata, world_size=1)
    try:
        train_state.model.load_state_dict(
            torch.load(args.checkpoint, map_location=args.device), assign=True)
    except Exception:
        train_state.model.load_state_dict(
            {k.removeprefix("_orig_mod."): v
             for k, v in torch.load(args.checkpoint, map_location=args.device).items()},
            assign=True)
    model = train_state.model.to(args.device).eval()

    hidden_size = model.config.hidden_size
    vector, meta = build_vector(args, hidden_size, args.device)
    print(f"Steering vector: {meta} | hidden_size={hidden_size} | "
          f"inject_steps={args.inject_steps}")

    # Find and hook the target module.
    target = None
    for name, mod in model.named_modules():
        if name == args.target_layer or name.endswith(args.target_layer):
            target = mod
            break
    if target is None:
        raise ValueError(f"Module {args.target_layer} not found.")
    steerer = Steerer(target, vector, args.inject_steps, args.early_cutoff)

    # ----- sweep over alpha -----
    rows = []
    baseline_logits = None
    for alpha in args.alphas:
        readout, logits = run_alpha(model, eval_loader, steerer, alpha, args)
        if alpha == 0.0:
            baseline_logits = logits

        cell_shift = frac_changed = float("nan")
        if logits is not None and baseline_logits is not None and logits.shape == baseline_logits.shape:
            cell_shift = (logits - baseline_logits).abs().mean().item()
            frac_changed = (logits.argmax(-1) != baseline_logits.argmax(-1)).float().mean().item()

        row = {
            "alpha": alpha,
            "vector_source": meta["vector_source"],
            "feature_id": meta["feature_id"],
            "inject_steps": args.inject_steps,
            "accuracy": readout["accuracy"],
            "mean_act_steps": readout["mean_act_steps"],
            "mean_logit_shift_vs_base": cell_shift,
            "frac_cells_changed_vs_base": frac_changed,
        }
        rows.append(row)
        print(f"alpha={alpha:+.1f} | acc={row['accuracy']:.3f} | "
              f"act_steps={row['mean_act_steps']:.2f} | "
              f"cells_changed={frac_changed:.3f}")

    steerer.remove()

    # ----- save -----
    csv_path = os.path.join(args.output_dir, "steer_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved sweep to {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        alphas = [r["alpha"] for r in rows]
        tag = f"{meta['vector_source']}_f{meta['feature_id']}_{args.inject_steps}"

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()
        ax1.plot(alphas, [r["accuracy"] for r in rows], "g-o", label="accuracy")
        ax2.plot(alphas, [r["mean_act_steps"] for r in rows], "b-s", label="ACT steps")
        ax1.set_xlabel("steering coefficient alpha")
        ax1.set_ylabel("solve accuracy", color="g")
        ax2.set_ylabel("mean ACT steps", color="b")
        plt.title(f"Steering dose-response ({tag})")
        fig.tight_layout()
        fig.savefig(f"results/figures/steer_doseresponse_{tag}.png", dpi=180)
        plt.close(fig)
        print(f"Saved plot to results/figures/steer_doseresponse_{tag}.png")
    except Exception as e:  # plotting is best-effort
        print(f"Plotting skipped: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
