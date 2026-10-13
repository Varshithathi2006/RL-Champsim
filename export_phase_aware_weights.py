import json
import torch
from train_discrete_phase_dqn import DiscretePhaseCacheDataset, DataLoader, train_model, evaluate_detailed

def main():
    ds_real = DiscretePhaseCacheDataset("datasets/combined_phase_belady_dataset.csv", scramble_phase=False)
    train_real, val_real = ds_real.get_train_val()

    train_loader = DataLoader(train_real, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_real, batch_size=256, shuffle=False)

    model = train_model(train_loader, val_loader, val_real, epochs=20, seed=42, device="cpu", title="EXPORTING 28-DIM PHASE AWARE MODEL")

    layers = []
    for m in model.net:
        if isinstance(m, torch.nn.Linear):
            layers.append({
                "weight": m.weight.detach().cpu().numpy().tolist(),
                "bias": m.bias.detach().cpu().numpy().tolist()
            })

    with open("phase_aware_dqn_weights.json", "w") as f:
        json.dump(layers, f, indent=2)
    print("Successfully exported 'phase_aware_dqn_weights.json'.")

if __name__ == "__main__":
    main()
