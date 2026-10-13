"""
train_dqn.py

Trains a DQN-style cache replacement agent from ChampSim replacement-policy
logs (champsim_rl_log.csv), using Belady's optimal eviction choice as the
learning signal.

CSV format expected (header row required):
    cycle,cpu,ip,full_addr,set,way,hit,type,recency_ranks,belady_victim_way

  - recency_ranks: semicolon-separated list of N values (one per cache way),
    e.g. "3;2;1;0;15;14;...". Lower value = more recently used.
  - belady_victim_way: the way index (0..N-1) that Belady's optimal policy
    would have evicted for this access.

Design note (for viva / report):
    Because we have Belady's optimal action as a ground-truth label for
    every logged access, this is trained as an *offline, single-step*
    Q-learning problem: for each state we push the Q-value of the optimal
    action toward +1 and all other actions toward 0 (a masked MSE loss).
    This is equivalent to training a policy from a fixed dataset with a
    dense reward signal, and is a standard simplification used in cache
    replacement RL literature since a full "trial and error" simulation
    loop is far more expensive to train than a labeled dataset like this.

Usage:
    python train_dqn.py \
        --csv champsim_rl_log.csv \
        --epochs 20 \
        --batch-size 256 \
        --out dqn_model.pth \
        --export-json dqn_weights.json
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
from torch.utils.data import Dataset, DataLoader, random_split


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class CacheReplacementDataset(Dataset):
    """
    Loads champsim_rl_log.csv and builds (state, action_label) pairs.

    Only rows where `hit == 0` (i.e. a miss occurred, so an eviction
    decision was actually needed) are used for training, since on a hit
    there is nothing to evict.

    State features per row:
        - recency_ranks (N values, normalized to [0,1])
        - type (one-hot encoded access type, small cardinality assumed)

    Label:
        - belady_victim_way (int, the way that should be evicted)
    """

    def __init__(self, csv_path, num_ways=16, max_type=8):
        self.num_ways = num_ways
        self.max_type = max_type
        states = []
        labels = []

        skipped_bad_rows = 0
        skipped_hits = 0

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    hit = int(row["hit"])
                    if hit == 1:
                        skipped_hits += 1
                        continue

                    recency_raw = row["recency_ranks"].split(";")
                    recency = [float(x) for x in recency_raw]
                    if len(recency) != num_ways:
                        skipped_bad_rows += 1
                        continue

                    # normalize recency ranks to [0, 1]
                    recency_norm = [r / (num_ways - 1) for r in recency]

                    access_type = int(row["type"])
                    type_onehot = [0.0] * max_type
                    if 0 <= access_type < max_type:
                        type_onehot[access_type] = 1.0

                    label = int(row["belady_victim_way"])
                    if not (0 <= label < num_ways):
                        skipped_bad_rows += 1
                        continue

                    state = recency_norm + type_onehot
                    states.append(state)
                    labels.append(label)

                except (KeyError, ValueError):
                    skipped_bad_rows += 1
                    continue

        if not states:
            raise RuntimeError(
                "No usable rows found. Check that the CSV has a 'hit' "
                "column with miss rows (hit==0), and that recency_ranks "
                "has the expected number of ways."
            )

        self.states = torch.tensor(np.array(states, dtype=np.float32))
        self.labels = torch.tensor(np.array(labels, dtype=np.int64))

        print(f"[dataset] loaded {len(self.states)} miss rows "
              f"(skipped {skipped_hits} hits, {skipped_bad_rows} bad rows)")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx]

    @property
    def state_dim(self):
        return self.states.shape[1]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(model, train_loader, val_loader, num_actions, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for states, labels in train_loader:
            states, labels = states.to(device), labels.to(device)

            # Build target Q-values: 1.0 for the optimal (Belady) action,
            # 0.0 for all others. This treats each logged access as a
            # single-step episode with a dense reward signal.
            targets = torch.zeros(states.size(0), num_actions, device=device)
            targets.scatter_(1, labels.unsqueeze(1), 1.0)

            q_values = model(states)
            loss = F.mse_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * states.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch:3d}/{epochs}] "
              f"train_loss={avg_loss:.5f}  val_match_acc={val_acc:.4f}")

    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Accuracy = fraction of rows where the agent's argmax action matches
    Belady's optimal eviction choice. This is directly comparable to how
    well an LRU baseline would match Belady (LRU usually matches far less
    often), which is a good number to report in your evaluation section.
    """
    model.eval()
    correct = 0
    total = 0
    for states, labels in loader:
        states, labels = states.to(device), labels.to(device)
        q_values = model(states)
        preds = q_values.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


# --------------------------------------------------------------------------
# Export for C++ inference (Step 4 of the pipeline)
# --------------------------------------------------------------------------

def export_weights_json(model, path):
    """
    Dumps weights/biases of each Linear layer as plain JSON arrays so they
    can be loaded by a small hand-written C++ inference function inside
    ChampSim's find_victim(), without needing a C++ ML framework.
    """
    layers = []
    for module in model.net:
        if isinstance(module, nn.Linear):
            layers.append({
                "weight": module.weight.detach().cpu().numpy().tolist(),  # [out, in]
                "bias": module.bias.detach().cpu().numpy().tolist(),      # [out]
            })
    with open(path, "w") as f:
        json.dump({"layers": layers}, f)
    print(f"[export] wrote {len(layers)} layers to {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to champsim_rl_log.csv")
    parser.add_argument("--num-ways", type=int, default=16, help="Cache set associativity")
    parser.add_argument("--max-type", type=int, default=8, help="Max distinct access 'type' values")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--out", default="dqn_model.pth", help="Where to save trained PyTorch weights")
    parser.add_argument("--export-json", default="dqn_weights.json", help="Where to export JSON weights for C++ port")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] using device: {device}")

    t0 = time.time()
    dataset = CacheReplacementDataset(args.csv, num_ways=args.num_ways, max_type=args.max_type)
    print(f"[setup] dataset loaded in {time.time() - t0:.1f}s, state_dim={dataset.state_dim}")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    # Chronological contiguous time-series split (first 80% train, last 20% validation)
    # Prevents data leakage between near-duplicate contiguous cycle accesses
    train_set = torch.utils.data.Subset(dataset, range(0, train_size))
    val_set = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    print(f"[split] Chronological time-series split: {train_size} train miss samples (first 80%), {val_size} val miss samples (last 20%)")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = DQN(state_dim=dataset.state_dim, num_actions=args.num_ways).to(device)
    print(model)

    model = train(model, train_loader, val_loader, args.num_ways, args.epochs, args.lr, device)

    final_acc = evaluate(model, val_loader, device)
    print(f"\n[result] final validation match-accuracy vs Belady: {final_acc:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"[save] model weights saved to {args.out}")

    export_weights_json(model, args.export_json)


if __name__ == "__main__":
    sys.exit(main())
