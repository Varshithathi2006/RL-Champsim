"""
convert_weights_to_js.py

Converts KMeans phase model and Phase-Aware DQN weights JSON files
into a single JavaScript file (weights.js) for browser-based neural network inference.
"""

import json
import os

def main():
    out_dir = "interactive_simulator"
    os.makedirs(out_dir, exist_ok=True)
    
    dqn_weights_path = "phase_aware_dqn_weights.json"
    kmeans_model_path = "kmeans_phase_model.json"
    js_output_path = os.path.join(out_dir, "weights.js")
    
    # Read DQN weights
    dqn_weights = []
    if os.path.exists(dqn_weights_path):
        with open(dqn_weights_path, "r") as f:
            dqn_weights = json.load(f)
        print(f"[convert] Read DQN weights with {len(dqn_weights)} layers.")
    else:
        print(f"[warning] {dqn_weights_path} not found. Simulator will use random/placeholder weights.")
        
    # Read KMeans model
    kmeans_model = {}
    if os.path.exists(kmeans_model_path):
        with open(kmeans_model_path, "r") as f:
            kmeans_model = json.load(f)
        print(f"[convert] Read KMeans model with {len(kmeans_model.get('cluster_centers', []))} clusters.")
    else:
        print(f"[warning] {kmeans_model_path} not found. Simulator will use default phase boundaries.")

    # Write to weights.js
    with open(js_output_path, "w") as f:
        f.write("// Auto-generated weights file for Phase-Aware Cache RL Simulator\n\n")
        f.write(f"const KMEANS_MODEL = {json.dumps(kmeans_model, indent=2)};\n\n")
        f.write(f"const DQN_WEIGHTS = {json.dumps(dqn_weights, indent=2)};\n")
        
    print(f"[convert] Successfully generated {js_output_path}.")

if __name__ == "__main__":
    main()
