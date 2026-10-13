"""
train_hierarchical_dqn.py

Trains a Two-Level Hierarchical Deep Q-Network (HDQN) for CPU Cache Replacement.

Architecture:
  - High-Level Meta Agent: Extracts phase features (rolling_miss_rate, rolling_read_ratio)
    and encodes workload execution context into a high-level phase embedding.
  - Low-Level Action Agent: Evaluates per-way features (recency, access frequency, age)
    concatenated with High-Level phase context to choose optimal eviction way.

Chronological 80/20 train/val split ensures zero time-series data leakage.
Exports network weights to 'hierarchical_dqn_weights.json' for C++ ChampSim integration.
"""

import argparse
import csv
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------
# Hierarchical Dataset Reader
# --------------------------------------------------------------------------

class HierarchicalCacheDataset(Dataset):
    def __init__(self, csv_path, num_ways=16, max_type=8):
        self.num_ways = num_ways
        self.max_type = max_type
        high_level_states = []
        low_level_states = []
        labels = []

        skipped_hits = 0
        skipped_bad = 0

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if int(row["hit"]) == 1:
                        skipped_hits += 1
                        continue

                    # High-Level Phase Features
                    miss_rate = float(row.get("rolling_miss_rate", 0.5))
                    read_ratio = float(row.get("rolling_read_ratio", 0.5))
                    high_feat = [miss_rate, read_ratio]

                    # Per-Way Low-Level Features
                    recency_raw = [float(x) for x in row["recency_ranks"].split(";")]
                    counts_raw = [float(x) for x in row.get("access_counts", "0;"*15+"0").split(";")[:num_ways]]
                    ages_raw = [float(x) for x in row.get("insertion_ages", "0;"*15+"0").split(";")[:num_ways]]

                    if len(recency_raw) != num_ways:
                        skipped_bad += 1
                        continue

                    # Feature Normalization
                    recency_norm = [r / (num_ways - 1) for r in recency_raw]
                    counts_norm = [min(c / 10.0, 1.0) for c in counts_raw]
                    ages_norm = [min(np.log1p(a) / 15.0, 1.0) for a in ages_raw]

                    # Access Type One-Hot
                    acc_type = int(row["type"])
                    type_onehot = [0.0] * max_type
                    if 0 <= acc_type < max_type:
                        type_onehot[acc_type] = 1.0

                    label = int(row["belady_victim_way"])
                    if not (0 <= label < num_ways):
                        skipped_bad += 1
                        continue

                    low_feat = recency_norm + counts_norm + ages_norm + type_onehot

                    high_level_states.append(high_feat)
                    low_level_states.append(low_feat)
                    labels.append(label)

                except (KeyError, ValueError):
                    skipped_bad += 1
                    continue

        self.high_states = torch.tensor(np.array(high_level_states, dtype=np.float32))
        self.low_states = torch.tensor(np.array(low_level_states, dtype=np.float32))
        self.labels = torch.tensor(np.array(labels, dtype=np.int64))

        print(f"[dataset] Loaded {len(self.labels)} miss samples (skipped {skipped_hits} hits, {skipped_bad} bad rows)")
        print(f"[dataset] High-Level Feature Dim: {self.high_states.shape[1]}, Low-Level Feature Dim: {self.low_states.shape[1]}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.high_states[idx], self.low_states[idx], self.labels[idx]


# --------------------------------------------------------------------------
# Two-Level Hierarchical DQN Model Architecture
# --------------------------------------------------------------------------

class HighLevelPhaseEncoder(nn.Module):
    """Encodes phase statistics into a 16-dimensional workload context vector."""
    def __init__(self, in_dim=2, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class LowLevelActionPolicy(nn.Module):
    """Combines way features + high-level phase embedding to pick victim way."""
    def __init__(self, low_dim, phase_dim=16, num_actions=16):
        super().__init__()
        total_in = low_dim + phase_dim
        self.net = nn.Sequential(
            nn.Linear(total_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, low_x, phase_embed):
        combined = torch.cat([low_x, phase_embed], dim=1)
        return self.net(combined)


class HierarchicalDQN(nn.Module):
    def __init__(self, high_dim=2, low_dim=56, phase_dim=16, num_actions=16):
        super().__init__()
        self.high_agent = HighLevelPhaseEncoder(high_dim, phase_dim)
        self.low_agent = LowLevelActionPolicy(low_dim, phase_dim, num_actions)

    def forward(self, high_x, low_x):
        phase_embed = self.high_agent(high_x)
        q_values = self.low_agent(low_x, phase_embed)
        return q_values


# --------------------------------------------------------------------------
# Training Engine
# --------------------------------------------------------------------------

def train(model, train_loader, val_loader, num_actions, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for high_x, low_x, labels in train_loader:
            high_x, low_x, labels = high_x.to(device), low_x.to(device), labels.to(device)

            targets = torch.zeros(high_x.size(0), num_actions, device=device)
            targets.scatter_(1, labels.unsqueeze(1), 1.0)

            q_values = model(high_x, low_x)
            loss = F.mse_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * high_x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch:2d}/{epochs}] train_loss={avg_loss:.5f}  val_match_acc={val_acc:.4f}")

    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for high_x, low_x, labels in loader:
        high_x, low_x, labels = high_x.to(device), low_x.to(device), labels.to(device)
        q_values = model(high_x, low_x)
        preds = q_values.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


# --------------------------------------------------------------------------
# C++ Export Function
# --------------------------------------------------------------------------

def export_hierarchical_weights(model, path):
    """Exports weights of both High-Level Phase Encoder and Low-Level Action Policy."""
    high_layers = []
    for m in model.high_agent.net:
        if isinstance(m, nn.Linear):
            high_layers.append({
                "weight": m.weight.detach().cpu().numpy().tolist(),
                "bias": m.bias.detach().cpu().numpy().tolist()
            })

    low_layers = []
    for m in model.low_agent.net:
        if isinstance(m, nn.Linear):
            low_layers.append({
                "weight": m.weight.detach().cpu().numpy().tolist(),
                "bias": m.bias.detach().cpu().numpy().tolist()
            })

    export_data = {
        "high_level_agent": high_layers,
        "low_level_agent": low_layers
    }

    with open(path, "w") as f:
        json.dump(export_data, f)
    print(f"[export] Successfully exported Hierarchical RL weights to '{path}'.")


# --------------------------------------------------------------------------
# Main Program
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="datasets/hierarchical_rl_log_belady.csv", help="Input CSV path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--out", default="hierarchical_dqn_model.pth")
    parser.add_argument("--export-json", default="hierarchical_dqn_weights.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Using device: {device}")

    dataset = HierarchicalCacheDataset(args.csv)

    # Chronological Contiguous Time-Series Split (80% Train, 20% Validation)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_set = torch.utils.data.Subset(dataset, range(0, train_size))
    val_set = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    print(f"[split] Chronological split: {train_size} train samples (first 80%), {val_size} val samples (last 20%)")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = HierarchicalDQN(
        high_dim=dataset.high_states.shape[1],
        low_dim=dataset.low_states.shape[1],
        phase_dim=16,
        num_actions=16
    ).to(device)

    print(model)

    model = train(model, train_loader, val_loader, 16, args.epochs, args.lr, device)

    final_acc = evaluate(model, val_loader, device)
    print(f"\n[result] Final Validation Match-Accuracy vs Belady: {final_acc:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"[save] Model saved to '{args.out}'.")

    export_hierarchical_weights(model, args.export_json)

if __name__ == "__main__":
    main()
