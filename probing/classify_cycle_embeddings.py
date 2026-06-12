import argparse
import glob
import os
import random
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class UndercompleteAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


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


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes=16, hidden_dim=0):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)


def parse_name(path):
    name = os.path.basename(path)
    cycle_match = re.search(r"cycle_(\d+)", name)
    batch_match = re.search(r"batch_(\d+)", name)
    if cycle_match is None or batch_match is None:
        raise ValueError(f"Could not parse cycle/batch from {name}")
    return int(cycle_match.group(1)), int(batch_match.group(1))


def load_states(data_dir, layer_filter, max_files, max_tokens_per_file, seed):
    files = sorted(glob.glob(os.path.join(data_dir, f"states_layer_*{layer_filter}*.pt")))
    if not files:
        raise FileNotFoundError(f"No state files found for filter {layer_filter} in {data_dir}")

    rng = random.Random(seed)
    if max_files > 0 and len(files) > max_files:
        files = rng.sample(files, max_files)

    xs = []
    ys = []
    batch_ids = []
    for path in files:
        cycle, batch_id = parse_name(path)
        state = torch.load(path, map_location="cpu").to(torch.float32)
        state = state.view(-1, state.shape[-1])
        if max_tokens_per_file > 0 and state.shape[0] > max_tokens_per_file:
            idx = torch.randperm(state.shape[0])[:max_tokens_per_file]
            state = state[idx]
        xs.append(state)
        ys.append(torch.full((state.shape[0],), cycle, dtype=torch.long))
        batch_ids.extend([batch_id] * state.shape[0])

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.tensor(batch_ids, dtype=torch.long)


def batch_split(batch_ids, test_frac, seed):
    unique_batches = sorted(batch_ids.unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(unique_batches)
    test_count = max(1, int(len(unique_batches) * test_frac))
    test_batches = set(unique_batches[:test_count])
    test_mask = torch.tensor([int(b.item()) in test_batches for b in batch_ids], dtype=torch.bool)
    train_mask = ~test_mask
    return train_mask, test_mask


def encode_features(args, x, input_dim, device):
    if args.representation == "raw":
        return x

    if args.representation == "uae":
        if not args.weights:
            raise ValueError("--weights is required for UAE representation")
        model = UndercompleteAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim)
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.to(device).eval()
        encoder = lambda batch: model.encoder(batch)
    elif args.representation == "sae":
        if not args.weights:
            raise ValueError("--weights is required for SAE representation")
        model = SparseAutoencoder(input_dim=input_dim, expansion_factor=args.expansion_factor)
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.to(device).eval()
        encoder = lambda batch: model.relu(model.encoder(batch - model.decoder_bias))
    else:
        raise ValueError(f"Unknown representation: {args.representation}")

    encoded = []
    with torch.no_grad():
        for start in range(0, x.shape[0], args.batch_size):
            batch = x[start:start + args.batch_size].to(device)
            encoded.append(encoder(batch).cpu())
    return torch.cat(encoded, dim=0)


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--layer_filter", type=str, default="H_level")
    parser.add_argument("--representation", choices=["raw", "uae", "sae"], default="raw")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--classifier_hidden_dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_tokens_per_file", type=int, default=512)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.layer_filter} states from {args.data_dir}")
    x, y, batch_ids = load_states(
        args.data_dir,
        args.layer_filter,
        args.max_files,
        args.max_tokens_per_file,
        args.seed,
    )
    input_dim = x.shape[-1]
    print(f"Raw dataset: x={tuple(x.shape)}, y={tuple(y.shape)}, input_dim={input_dim}")

    z = encode_features(args, x, input_dim, device)
    train_mask, test_mask = batch_split(batch_ids, args.test_frac, args.seed)

    train_z = z[train_mask]
    test_z = z[test_mask]
    train_y = y[train_mask]
    test_y = y[test_mask]

    mean = train_z.mean(dim=0, keepdim=True)
    std = train_z.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_z = (train_z - mean) / std
    test_z = (test_z - mean) / std

    print(f"Representation: {args.representation}, dim={train_z.shape[-1]}")
    print(f"Train tokens: {train_z.shape[0]}, test tokens: {test_z.shape[0]}")

    train_loader = DataLoader(TensorDataset(train_z, train_y), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_z, test_y), batch_size=args.batch_size)

    classifier = Classifier(
        input_dim=train_z.shape[-1],
        num_classes=int(y.max().item()) + 1,
        hidden_dim=args.classifier_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        classifier.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = loss_fn(classifier(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_acc = accuracy(classifier, train_loader, device)
        test_acc = accuracy(classifier, test_loader, device)
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} "
            f"loss={total_loss / len(train_loader):.4f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
