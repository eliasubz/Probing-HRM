import argparse
import json
import os
from collections import defaultdict

os.environ["DISABLE_COMPILE"] = "1"

import torch
import yaml

from pretrain import PretrainConfig, create_dataloader, init_train_state


def get_args():
    parser = argparse.ArgumentParser(description="Extract structured HRM Sudoku probe data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sudoku_probe_data_h_layer3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--target_layers", nargs="+", default=["model.inner.H_level.layers.3"])
    parser.add_argument("--return_logits", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class StructuredHiddenStateExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.current_cycle = 0
        self.hidden_states = defaultdict(lambda: defaultdict(list))
        self.hooks = []
        self._register_hooks()

    def _get_module_by_name(self, name):
        for module_name, module in self.model.named_modules():
            if module_name == name or module_name.endswith(name):
                return module
        available = [n for n, _ in self.model.named_modules() if "layer" in n.lower() or "level" in n.lower()]
        raise ValueError(f"Module '{name}' not found. Available layer-like modules: {available}")

    def _hook_fn(self, layer_name):
        def hook(_module, _input, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            self.hidden_states[layer_name][self.current_cycle].append(out_tensor.detach().cpu())
        return hook

    def _register_hooks(self):
        for name in self.target_layers:
            module = self._get_module_by_name(name)
            self.hooks.append(module.register_forward_hook(self._hook_fn(name)))

    def clear_states(self):
        self.hidden_states.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def as_structured_tensors(self):
        structured = {}
        for layer_name, cycle_dict in self.hidden_states.items():
            cycle_tensors = []
            for cycle in sorted(cycle_dict):
                # Shape: [inner_hook_call, batch, seq_with_puzzle_token, hidden]
                cycle_tensors.append(torch.stack(cycle_dict[cycle], dim=0))
            safe_name = layer_name.replace(".", "_")
            # Shape: [act_cycle, inner_hook_call, batch, seq_with_puzzle_token, hidden]
            structured[safe_name] = torch.stack(cycle_tensors, dim=0).to(torch.bfloat16)
        return structured


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {args.device}")
    config_path = os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config file at {config_path}")

    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
    config.global_batch_size = args.batch_size

    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )

    train_state = init_train_state(config, eval_metadata, world_size=1)
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    try:
        train_state.model.load_state_dict(checkpoint, assign=True)
    except Exception:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in checkpoint.items()}, assign=True)

    model = train_state.model
    model.to(args.device)
    model.eval()

    extractor = StructuredHiddenStateExtractor(model, args.target_layers)
    manifest = {
        "checkpoint": args.checkpoint,
        "target_layers": args.target_layers,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "return_logits": args.return_logits,
        "notes": {
            "states_shape": "[act_cycle, inner_hook_call, batch, seq_with_puzzle_token, hidden]",
            "cell_states": "drop token 0 if puzzle_emb_len is 1; Sudoku cells are then tokens 1..81",
            "token_encoding": "Sudoku dataset stores blank as 1 and digits 1..9 as labels 2..10",
        },
    }

    print("Extracting structured probe batches...")
    with torch.inference_mode():
        for batch_idx, (set_name, batch, global_batch_size) in enumerate(eval_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                print(f"Stopping early at {args.max_batches} batches.")
                break

            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.device(args.device):
                carry = model.initial_carry(batch)

            logits_by_cycle = []
            extractor.current_cycle = 0
            while True:
                return_keys = ["logits"] if args.return_logits else []
                carry, _, _metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=return_keys)
                if args.return_logits and preds is not None and "logits" in preds:
                    logits_by_cycle.append(preds["logits"].detach().cpu().to(torch.float32))

                extractor.current_cycle += 1
                if all_finish:
                    break

            inputs = batch["inputs"].detach().cpu()[:global_batch_size]
            labels = batch["labels"].detach().cpu()[:global_batch_size]
            puzzle_identifiers = batch["puzzle_identifiers"].detach().cpu()[:global_batch_size]
            given_mask = inputs != 1

            payload = {
                "set_name": set_name,
                "batch_idx": batch_idx,
                "global_batch_size": int(global_batch_size),
                "inputs": inputs,
                "labels": labels,
                "given_mask": given_mask,
                "puzzle_identifiers": puzzle_identifiers,
                "layers": extractor.as_structured_tensors(),
            }
            if logits_by_cycle:
                payload["logits"] = torch.stack(logits_by_cycle, dim=0)

            save_path = os.path.join(args.output_dir, f"structured_batch_{batch_idx:05d}.pt")
            torch.save(payload, save_path)
            extractor.clear_states()
            print(f"Saved {save_path}")

    extractor.remove_hooks()
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
