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
            
        print(f"Loading {len(files)} files into memory (limited to {max_files} to prevent RAM OOM)...")
        
        tensors = []
        for f in files:
            try:
                t = torch.load(f, map_location='cpu')
                tensors.append(t.view(-1, t.shape[-1])) 
            except Exception as e:
                print(f"Skipping and deleting corrupted file {f}: {e}")
                try:
                    os.remove(f)
                except OSError:
                    pass
            
        self.data = torch.cat(tensors, dim=0).to(torch.float32)
        print(f"Loaded dataset shape: {self.data.shape} (Total Tokens: {self.data.shape[0]})")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class UndercompleteAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="probing_data_layer3", help="Directory containing probing data")
    parser.add_argument("--output_dir", type=str, default="uae_results", help="Directory to save weights and plots")
    parser.add_argument("--layer_filter", type=str, default="L_level", help="Filter for which layer's data to load")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max_files", type=int, default=1000)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = InMemoryProbingDataset(args.data_dir, layer_filter=args.layer_filter, max_files=args.max_files)
    actual_input_dim = dataset.data.shape[-1]
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    autoencoder = UndercompleteAutoencoder(input_dim=actual_input_dim, latent_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    loss_history = []
    print("Starting UAE training...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed, _ = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")
        
    # Save model
    weights_path = os.path.join(args.output_dir, "uae_weights.pt")
    torch.save(autoencoder.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}")
    
    # Save metrics
    metrics = {"final_loss": loss_history[-1], "loss_history": loss_history}
    with open(os.path.join(args.output_dir, "uae_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label='Train Loss', color='purple')
    plt.title('UAE Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "uae_loss.png"))
    plt.close()
    
    # Viz
    print("Extracting latents for visualization...")
    autoencoder.eval()
    subset_size = min(5000, len(dataset))
    indices = torch.randperm(len(dataset))[:subset_size]
    subset_data = dataset.data[indices].to(device)
    with torch.no_grad():
        _, latents = autoencoder(subset_data)
    latents_np = latents.cpu().numpy()
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(latents_np)
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_pca[:, 0], latents_pca[:, 1], alpha=0.6, s=15, c='dodgerblue')
    plt.title('PCA of UAE Latent Space')
    plt.savefig(os.path.join(args.output_dir, "uae_pca.png"))
    plt.close()
    
    print("Running t-SNE...")
    tsne_kwargs = {"n_components": 2, "perplexity": 30}
    if "max_iter" in TSNE.__init__.__code__.co_varnames:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    tsne = TSNE(**tsne_kwargs)
    latents_tsne = tsne.fit_transform(latents_np)
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], alpha=0.6, s=15, c='mediumseagreen')
    plt.title('t-SNE of UAE Latent Space')
    plt.savefig(os.path.join(args.output_dir, "uae_tsne.png"))
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()
