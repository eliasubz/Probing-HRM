import argparse
import os

os.environ["DISABLE_COMPILE"] = "1"

import torch
import torch.nn as nn
from collections import defaultdict
import yaml

from pretrain import PretrainConfig, create_dataloader, init_train_state

def get_args():
    parser = argparse.ArgumentParser(description="Extract hidden states for probing")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (e.g. checkpoints/project/run/step_100)")
    parser.add_argument("--output_dir", type=str, default="probing_data", help="Directory to save extracted hidden states")
    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size for inference")
    parser.add_argument("--max_batches", type=int, default=5, help="Maximum number of batches to process (-1 for all)")
    parser.add_argument("--target_layers", nargs='+', default=["model.inner.H_level.layers.0", "model.inner.L_level.layers.0"], help="Layers to hook")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

class HiddenStateExtractor:
    """
    Utility class to extract hidden states from a PyTorch model using forward hooks.
    """
    def __init__(self, model, target_layers):
        """
        Args:
            model (nn.Module): The PyTorch model to hook.
            target_layers (list[str]): List of module names to extract states from.
                                       (e.g., ['encoder.layers.0', 'decoder.layers.2'])
        """
        self.model = model
        self.target_layers = target_layers
        self.hidden_states = defaultdict(lambda: defaultdict(list))
        self.current_cycle = 0
        self.hooks = []
        self._register_hooks()

    def _get_module_by_name(self, name):
        for n, m in self.model.named_modules():
            if n == name or n.endswith(name):
                return m
        
        
        available = [n for n, m in self.model.named_modules() if 'layer' in n.lower() or 'level' in n.lower()]
        raise ValueError(f"Module '{name}' not found in the model.\nHint: Here are some available layers you can hook: {available}")

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            out_tensor = output[0] if isinstance(output, tuple) else output
            
            self.hidden_states[layer_name][self.current_cycle].append(out_tensor.detach().cpu())
        return hook

    def _register_hooks(self):
        for name in self.target_layers:
            module = self._get_module_by_name(name)
            hook = module.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Removes all attached hooks. Call this when you're done extracting."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_states(self):
        """Clears accumulated states."""
        self.hidden_states.clear()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {args.device}")

    # 1. Load Config and DataLoader
    print("Loading config and dataloader...")
    
    config_path = os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config file at {config_path}. Checkpoint path must be the full path including the file itself.")

    with open(config_path, "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
        
    config.global_batch_size = args.batch_size
    
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1, 
        global_batch_size=config.global_batch_size, rank=0, world_size=1
    )


    # 2. Load Model
    print("Initializing model...")
    train_state = init_train_state(config, eval_metadata, world_size=1)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    try:
        train_state.model.load_state_dict(torch.load(args.checkpoint, map_location=args.device), assign=True)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(args.checkpoint, map_location=args.device).items()}, assign=True)
        
    model = train_state.model
    model.to(args.device)
    model.eval()

    # 3. Define target layers for probing
    target_layers = args.target_layers
    extractor = HiddenStateExtractor(model, target_layers)

    # 4. Extract Hidden States
    print("Extracting hidden states...")
    all_labels = []
    
    with torch.inference_mode():
        for batch_idx, (set_name, batch, global_batch_size) in enumerate(eval_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                print(f"Stopping early at {args.max_batches} batches.")
                break
                
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            with torch.device(args.device):
                carry = model.initial_carry(batch)

            # ACT Forward Loop
            extractor.current_cycle = 0
            while True:
                carry, _, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=[])
                
                extractor.current_cycle += 1
                
                if all_finish:
                    break
            
            if "labels" in batch:
                all_labels.append(batch["labels"].cpu()[:global_batch_size])

            
            for layer_name, cycles_dict in extractor.hidden_states.items():
                for cycle_idx, states_list in cycles_dict.items():
                    concatenated_states = torch.cat(states_list, dim=0)
                    
                    safe_layer_name = layer_name.replace(".", "_")
                    save_path = os.path.join(args.output_dir, f"states_layer_{safe_layer_name}_cycle_{cycle_idx}_batch_{batch_idx}.pt")
                    
                    # Cast to bfloat16 to save disk space
                    torch.save(
                        concatenated_states.to(torch.bfloat16),
                        save_path,
                        _use_new_zipfile_serialization=False,
                    )
            
            # CLEAR states from memory
            extractor.clear_states()
            print(f"Processed and saved batch {batch_idx}")

    print("Finished processing all batches. Saving labels...")
    if all_labels:
        all_labels_tensor = torch.cat(all_labels, dim=0)
        labels_save_path = os.path.join(args.output_dir, "labels.pt")
        torch.save(all_labels_tensor, labels_save_path, _use_new_zipfile_serialization=False)
        print(f"  Saved {labels_save_path} | Shape: {all_labels_tensor.shape}")

    extractor.remove_hooks()
    print("Done!")

if __name__ == "__main__":
    main()
