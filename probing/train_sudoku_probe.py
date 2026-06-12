import argparse
import csv
import glob
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Probe(nn.Module):
    def __init__(self, input_dim, hidden_dim=0, num_classes=9):
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


def load_examples(data_dir, layer_name, cycle, inner_call, max_batches, include_givens, include_blanks):
    files = sorted(glob.glob(os.path.join(data_dir, "structured_batch_*.pt")))
    if max_batches > 0:
        files = files[:max_batches]
    if not files:
        raise FileNotFoundError(f"No structured batches found in {data_dir}")

    xs = []
    ys = []
    masks = []
    batch_ids = []
    for path in files:
        payload = torch.load(path, map_location="cpu")
        layers = payload["layers"]
        if layer_name not in layers:
            raise KeyError(f"Layer '{layer_name}' not found. Available: {list(layers)}")

        states = layers[layer_name].to(torch.float32)
        if cycle >= states.shape[0]:
            continue
        if inner_call >= states.shape[1]:
            continue

        # Shape after selection: [batch, seq_with_puzzle_token, hidden].
        selected = states[cycle, inner_call]
        labels = payload["labels"]
        given_mask = payload["given_mask"]

        # Token 0 is the puzzle embedding token for this HRM config; cells are 1..81.
        if selected.shape[1] == labels.shape[1] + 1:
            selected = selected[:, 1:, :]
        elif selected.shape[1] != labels.shape[1]:
            raise ValueError(
                f"Unexpected seq length {selected.shape[1]} for labels length {labels.shape[1]} in {path}"
            )

        label_digits = labels - 2
        valid = (label_digits >= 0) & (label_digits < 9)
        if not include_givens:
            valid = valid & ~given_mask
        if not include_blanks:
            valid = valid & given_mask

        flat_x = selected.reshape(-1, selected.shape[-1])
        flat_y = label_digits.reshape(-1)
        flat_given = given_mask.reshape(-1)
        flat_valid = valid.reshape(-1)

        xs.append(flat_x[flat_valid])
        ys.append(flat_y[flat_valid].to(torch.long))
        masks.append(flat_given[flat_valid])
        batch_ids.extend([payload["batch_idx"]] * int(flat_valid.sum().item()))

    if not xs:
        raise ValueError("No examples loaded for the requested cycle/inner_call/filter")

    return torch.cat(xs), torch.cat(ys), torch.cat(masks), torch.tensor(batch_ids, dtype=torch.long)


def split_by_batch(batch_ids, test_frac, seed):
    unique = sorted(batch_ids.unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(unique)
    test_n = max(1, int(len(unique) * test_frac))
    test_batches = set(unique[:test_n])
    test_mask = torch.tensor([int(x.item()) in test_batches for x in batch_ids], dtype=torch.bool)
    return ~test_mask, test_mask


def eval_accuracy(model, x, y, given_mask, device, batch_size):
    model.eval()
    pred_chunks = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            logits = model(x[start:start + batch_size].to(device))
            pred_chunks.append(logits.argmax(dim=-1).cpu())
    pred = torch.cat(pred_chunks)
    correct = pred == y

    def masked_acc(mask):
        count = int(mask.sum().item())
        if count == 0:
            return float("nan")
        return float(correct[mask].float().mean().item())

    return {
        "accuracy": float(correct.float().mean().item()),
        "given_accuracy": masked_acc(given_mask),
        "blank_accuracy": masked_acc(~given_mask),
    }


def train_one(args, cycle, inner_call):
    x, y, given_mask, batch_ids = load_examples(
        args.data_dir,
        args.layer_name,
        cycle,
        inner_call,
        args.max_batches,
        include_givens=True,
        include_blanks=True,
    )
    train_mask, test_mask = split_by_batch(batch_ids, args.test_frac, args.seed)
    train_x, test_x = x[train_mask], x[test_mask]
    train_y, test_y = y[train_mask], y[test_mask]
    train_given, test_given = given_mask[train_mask], given_mask[test_mask]

    if args.train_on == "givens":
        train_filter = train_given
    elif args.train_on == "blanks":
        train_filter = ~train_given
    else:
        train_filter = torch.ones_like(train_given, dtype=torch.bool)

    train_x = train_x[train_filter]
    train_y = train_y[train_filter]

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Probe(train_x.shape[-1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for _epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = loss_fn(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    metrics = eval_accuracy(model, test_x, test_y, test_given, device, args.batch_size)
    return {
        "cycle": cycle,
        "inner_call": inner_call,
        "train_examples": int(train_x.shape[0]),
        "test_examples": int(test_x.shape[0]),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="sudoku_probe_results.csv")
    parser.add_argument("--layer_name", type=str, default="model_inner_H_level_layers_3")
    parser.add_argument("--cycles", type=int, nargs="*", default=None)
    parser.add_argument("--inner_calls", type=int, nargs="*", default=None)
    parser.add_argument("--train_on", choices=["all", "givens", "blanks"], default="all")
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    sample_file = sorted(glob.glob(os.path.join(args.data_dir, "structured_batch_*.pt")))[0]
    sample = torch.load(sample_file, map_location="cpu")
    sample_states = sample["layers"][args.layer_name]
    cycles = args.cycles if args.cycles is not None else list(range(sample_states.shape[0]))
    inner_calls = args.inner_calls if args.inner_calls is not None else list(range(sample_states.shape[1]))

    rows = []
    for cycle in cycles:
        for inner_call in inner_calls:
            print(f"Training probe for cycle={cycle}, inner_call={inner_call}")
            row = train_one(args, cycle, inner_call)
            rows.append(row)
            print(row)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        fieldnames = [
            "cycle",
            "inner_call",
            "train_examples",
            "test_examples",
            "accuracy",
            "given_accuracy",
            "blank_accuracy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
