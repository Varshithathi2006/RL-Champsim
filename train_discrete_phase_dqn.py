"""
train_discrete_phase_dqn.py

Trains Low-Level Action Policy with 28-dim input:
  - 16 Recency Ranks (normalized)
  - 8 Access Type One-Hot
  - 4 Discrete Phase Cluster One-Hot

Performs 80/20 chronological time-series split within each trace.
Evaluates accuracy overall, broken down by trace (mcf vs gcc), and by phase cluster (0..3).
Runs a CONTROL experiment with randomly scrambled phase labels to verify phase utility.
"""

import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class DiscretePhaseCacheDataset(Dataset):
    def __init__(self, csv_path, num_ways=16, max_type=8, num_clusters=4, scramble_phase=False):
        self.num_ways = num_ways
        self.num_clusters = num_clusters
        
        mcf_samples = []
        gcc_samples = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                recency_raw = [float(x) for x in row["recency_ranks"].split(";")]
                if len(recency_raw) != num_ways:
                    continue
                recency_norm = [r / (num_ways - 1) for r in recency_raw]

                acc_type = int(row["type"])
                type_onehot = [0.0] * max_type
                if 0 <= acc_type < max_type:
                    type_onehot[acc_type] = 1.0

                cluster_id = int(row["phase_cluster"])
                cluster_onehot = [0.0] * num_clusters
                if 0 <= cluster_id < num_clusters:
                    cluster_onehot[cluster_id] = 1.0

                label = int(row["belady_victim_way"])
                trace = row["trace_source"]

                sample = {
                    "recency": recency_norm,
                    "type_onehot": type_onehot,
                    "cluster_onehot": list(cluster_onehot),
                    "cluster_id": cluster_id,
                    "label": label,
                    "trace": trace
                }

                if trace == "429.mcf":
                    mcf_samples.append(sample)
                else:
                    gcc_samples.append(sample)

        all_samples = mcf_samples + gcc_samples

        if scramble_phase:
            # Generate truly independent random phase clusters 0..3
            rng = np.random.RandomState(42)
            random_cluster_ids = rng.randint(0, num_clusters, size=len(all_samples))
            for i, s in enumerate(all_samples):
                c_id = int(random_cluster_ids[i])
                c_onehot = [0.0] * num_clusters
                c_onehot[c_id] = 1.0
                s["cluster_onehot"] = c_onehot
                s["cluster_id"] = c_id

        # 80/20 Chronological Split per trace
        mcf_val_idx = int(len(mcf_samples) * 0.8)
        gcc_val_idx = int(len(gcc_samples) * 0.8)

        self.train_samples = mcf_samples[:mcf_val_idx] + gcc_samples[:gcc_val_idx]
        self.val_samples = mcf_samples[mcf_val_idx:] + gcc_samples[gcc_val_idx:]

    def get_train_val(self):
        return SampleSet(self.train_samples), SampleSet(self.val_samples)


class SampleSet(Dataset):
    def __init__(self, samples):
        self.samples = samples
        inputs = []
        labels = []
        for s in samples:
            feat = s["recency"] + s["type_onehot"] + s["cluster_onehot"] # 16 + 8 + 4 = 28 dims
            inputs.append(feat)
            labels.append(s["label"])
        self.x = torch.tensor(np.array(inputs, dtype=np.float32))
        self.y = torch.tensor(np.array(labels, dtype=np.int64))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx


class LowLevelDQN(nn.Module):
    def __init__(self, in_dim=28, num_actions=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.net(x)


def train_model(train_loader, val_loader, val_set_obj, epochs=20, lr=1e-3, seed=42, device="cpu", title="Experiment"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LowLevelDQN(in_dim=28, num_actions=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n================ {title} ================")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            targets = torch.zeros(x.size(0), 16, device=device)
            targets.scatter_(1, y.unsqueeze(1), 1.0)

            q_values = model(x)
            loss = F.mse_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_acc, _, _ = evaluate_detailed(model, val_loader, val_set_obj, device)
        print(f"[epoch {epoch:2d}/{epochs}] train_loss={avg_loss:.5f}  val_match_acc={val_acc:.4f}")

    return model


@torch.no_grad()
def evaluate_detailed(model, val_loader, val_set_obj, device):
    model.eval()
    correct = 0
    total = 0

    trace_correct = {}
    trace_total = {}

    cluster_correct = {}
    cluster_total = {}

    for x, y, indices in val_loader:
        x, y = x.to(device), y.to(device)
        q_values = model(x)
        preds = q_values.argmax(dim=1).cpu().numpy()
        targets = y.cpu().numpy()

        for i, idx in enumerate(indices.numpy()):
            sample = val_set_obj.samples[idx]
            p = preds[i]
            t = targets[i]
            is_correct = (p == t)

            if is_correct:
                correct += 1
            total += 1

            tr = sample["trace"]
            trace_correct[tr] = trace_correct.get(tr, 0) + (1 if is_correct else 0)
            trace_total[tr] = trace_total.get(tr, 0) + 1

            cl = sample["cluster_id"]
            cluster_correct[cl] = cluster_correct.get(cl, 0) + (1 if is_correct else 0)
            cluster_total[cl] = cluster_total.get(cl, 0) + 1

    overall_acc = correct / total if total > 0 else 0.0

    trace_acc = {tr: trace_correct[tr] / trace_total[tr] for tr in trace_total}
    cluster_acc = {cl: cluster_correct[cl] / cluster_total[cl] for cl in cluster_total}

    return overall_acc, trace_acc, cluster_acc


def get_predictions(model, loader, device="cpu"):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    return all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/combined_phase_belady_dataset.csv")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"[setup] Using device: {device}")

    # 1. Main Experiment: Real Phase Signal
    ds_real = DiscretePhaseCacheDataset(args.csv, scramble_phase=False)
    train_real, val_real = ds_real.get_train_val()

    train_loader_real = DataLoader(train_real, batch_size=256, shuffle=True)
    val_loader_real = DataLoader(val_real, batch_size=256, shuffle=False)

    print(f"[dataset] Total Train: {len(train_real)}, Total Val: {len(val_real)}")

    model_real = train_model(train_loader_real, val_loader_real, val_real, epochs=args.epochs, seed=42, device=device, title="REAL PHASE SIGNAL TRAINING")
    acc_real, trace_acc_real, cluster_acc_real = evaluate_detailed(model_real, val_loader_real, val_real, device)
    preds_real = get_predictions(model_real, val_loader_real, device)

    # 2. Control Experiment: Scrambled Phase Signal
    ds_scrambled = DiscretePhaseCacheDataset(args.csv, scramble_phase=True)
    train_scrambled, val_scrambled = ds_scrambled.get_train_val()

    train_loader_scrambled = DataLoader(train_scrambled, batch_size=256, shuffle=True)
    val_loader_scrambled = DataLoader(val_scrambled, batch_size=256, shuffle=False)

    model_scrambled = train_model(train_loader_scrambled, val_loader_scrambled, val_scrambled, epochs=args.epochs, seed=100, device=device, title="CONTROL (SCRAMBLED PHASE SIGNAL) TRAINING")
    acc_scrambled, trace_acc_scrambled, cluster_acc_scrambled = evaluate_detailed(model_scrambled, val_loader_scrambled, val_scrambled, device)
    preds_scrambled = get_predictions(model_scrambled, val_loader_scrambled, device)

    print("\n" + "="*70)
    print("                      SAMPLE PREDICTIONS COMPARISON (FIRST 15 VAL ROWS)                      ")
    print("="*70)
    print(f"{'Idx':<4} | {'Target (Belady)':<15} | {'Real Phase Pred':<16} | {'Scrambled Pred':<15} | {'Real One-Hot':<18} | {'Scrambled One-Hot':<18}")
    print("-" * 105)
    for i in range(15):
        t_label = val_real.samples[i]["label"]
        r_pred = preds_real[i]
        s_pred = preds_scrambled[i]
        r_oh = val_real.samples[i]["cluster_onehot"]
        s_oh = val_scrambled.samples[i]["cluster_onehot"]
        print(f"{i:<4} | {t_label:<15} | {r_pred:<16} | {s_pred:<15} | {str(r_oh):<18} | {str(s_oh):<18}")

    print("\n" + "="*70)
    print("                      FINAL COMPARISON REPORT                      ")
    print("="*70)
    print(f"Overall Validation Match-Accuracy (Real Phase Signal)      : {acc_real*100:.2f}%")
    print(f"Overall Validation Match-Accuracy (Scrambled Control Phase) : {acc_scrambled*100:.2f}%")
    print(f"Delta Accuracy (Real vs Scrambled)                          : {+(acc_real - acc_scrambled)*100:+.2f}%")

    print("\n--- Accuracy Breakdown by Trace ---")
    for tr in sorted(trace_acc_real.keys()):
        print(f"  {tr:<10} | Real Phase: {trace_acc_real[tr]*100:.2f}%  |  Scrambled: {trace_acc_scrambled.get(tr, 0)*100:.2f}%")

    print("\n--- Accuracy Breakdown by Phase Cluster ---")
    for cl in sorted(cluster_acc_real.keys()):
        print(f"  Cluster {cl}  | Real Phase: {cluster_acc_real[cl]*100:.2f}%  |  Scrambled: {cluster_acc_scrambled.get(cl, 0)*100:.2f}%")

if __name__ == "__main__":
    main()
