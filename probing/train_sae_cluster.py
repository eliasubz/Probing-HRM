import argparse
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import json

class InMemoryProbingDataset(Dataset):
    def __init__(self, data_dir, layer_filter="L_level", max_files=1000):
        files = sorted(glob.glob(os.path.join(data_dir, f"states_layer_*{layer_filter}*.pt")))
        if len(files) > max_files:
            import random
            random.seed(42)
            files = random.sample(files, max_files)
        print(f"Loading {len(files)} files...")
        tensors = []
        for f in files:
            try:
                t = torch.load(f, map_location='cpu')
                tensors.append(t.view(-1, t.shape[-1])) 
            except Exception:
                pass
        self.data = torch.cat(tensors, dim=0).to(torch.float32)
        print(f"Loaded dataset shape: {self.data.shape}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, expansion_factor=4):
        super().__init__()
        self.latent_dim = input_dim * expansion_factor
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        
        with torch.no_grad():
            self.decoder.weight.data = self.decoder.weight.data / self.decoder.weight.data.norm(dim=0, keepdim=True)
            
    def forward(self, x):
        # Center with pre-bias (decoder bias)
        x_centered = x - self.decoder_bias
        
        latent = self.relu(self.encoder(x_centered))
        
        reconstructed = self.decoder(latent) + self.decoder_bias
        return reconstructed, latent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="probing_data_layer3", help="Directory containing probing data")
    parser.add_argument("--output_dir", type=str, default="sae_results", help="Directory to save weights and plots")
    parser.add_argument("--layer_filter", type=str, default="L_level", help="Filter for which layer's data to load")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max_files", type=int, default=1000)
    parser.add_argument("--l1_coeff", type=float, default=1e-3, help="L1 regularization coefficient for sparsity")
    parser.add_argument("--expansion_factor", type=int, default=4, help="Overcomplete expansion factor")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = InMemoryProbingDataset(args.data_dir, layer_filter=args.layer_filter, max_files=args.max_files)
    actual_input_dim = dataset.data.shape[-1]
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    sae = SparseAutoencoder(input_dim=actual_input_dim, expansion_factor=args.expansion_factor).to(device)
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    
    metrics = {
        "mse_loss": [],
        "l1_loss": [],
        "l0_norm": [],
        "dead_features": []
    }
    
    print(f"Starting SAE training (Input Dim: {actual_input_dim}, Latent Dim: {sae.latent_dim})...")
    for epoch in range(args.epochs):
        epoch_mse = 0.0
        epoch_l1 = 0.0
        epoch_l0 = 0.0
        
        feature_activations = torch.zeros(sae.latent_dim, device=device)
        
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed, latent = sae(batch)
            
            mse_loss = mse_loss_fn(reconstructed, batch)
            l1_loss = latent.abs().sum(dim=-1).mean()
            loss = mse_loss + args.l1_coeff * l1_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # SAE best practice: normalize decoder weights continuously to prevent unbounded growth
            with torch.no_grad():
                sae.decoder.weight.data = sae.decoder.weight.data / sae.decoder.weight.data.norm(dim=0, keepdim=True)
            
            optimizer.step()
            
            epoch_mse += mse_loss.item()
            epoch_l1 += l1_loss.item()
            
            # Tracking metrics
            l0 = (latent > 0).float().sum(dim=-1).mean().item() # Avg active features per token
            epoch_l0 += l0
            feature_activations += (latent > 0).float().sum(dim=0)
            
        metrics["mse_loss"].append(epoch_mse / len(dataloader))
        metrics["l1_loss"].append(epoch_l1 / len(dataloader))
        metrics["l0_norm"].append(epoch_l0 / len(dataloader))
        
        dead_feature_count = (feature_activations == 0).sum().item()
        metrics["dead_features"].append(dead_feature_count)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] | MSE: {metrics['mse_loss'][-1]:.4f} | L1: {metrics['l1_loss'][-1]:.4f} | L0: {metrics['l0_norm'][-1]:.1f} | Dead: {dead_feature_count}")

    # Save weights
    weights_path = os.path.join(args.output_dir, "sae_weights.pt")
    torch.save(sae.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}")
    
    with open(os.path.join(args.output_dir, "sae_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot losses
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(metrics["mse_loss"], 'g-', label='MSE Loss')
    ax2.plot(metrics["l1_loss"], 'b-', label='L1 Loss (Sparsity)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss', color='g')
    ax2.set_ylabel('L1 Loss', color='b')
    plt.title('SAE Training Losses')
    plt.savefig(os.path.join(args.output_dir, "sae_losses.png"))
    plt.close()
    
    # Plot L0 Norm
    plt.figure(figsize=(8, 4))
    plt.plot(metrics["l0_norm"], color='orange')
    plt.title('L0 Norm (Average Active Features per Token)')
    plt.xlabel('Epoch')
    plt.ylabel('L0 Norm')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "sae_l0_norm.png"))
    plt.close()
    
    # Plot Dead Features
    plt.figure(figsize=(8, 4))
    plt.plot(metrics["dead_features"], color='red')
    plt.title('Dead Features Count')
    plt.xlabel('Epoch')
    plt.ylabel('# Dead Features')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "sae_dead_features.png"))
    plt.close()

    print("Done!")

if __name__ == "__main__":
    main()
