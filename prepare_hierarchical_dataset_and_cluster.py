"""
prepare_hierarchical_dataset_and_cluster.py

1. Fits K=4 KMeans phase clustering model on combined mcf + gcc phase features.
2. Exports scaler and cluster centroids to 'kmeans_phase_model.json'.
3. Calculates True Belady MIN lookahead victim RECENCY RANK (0..15, where 15=LRU).
4. Assigns discrete phase cluster labels (0..3) to each miss sample.
5. Saves the combined labeled dataset to 'combined_phase_belady_dataset.csv'.
"""

import csv
import json
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def calculate_true_belady_recency_ranks(rows):
    """Calculates True Belady MIN lookahead victim RECENCY RANK (0..15) for miss accesses."""
    set_accesses = defaultdict(list)
    for idx, row in enumerate(rows):
        set_idx = int(row["set"])
        addr = row["full_addr"]
        hit = int(row["hit"])
        set_accesses[set_idx].append((idx, addr, hit))

    NUM_WAYS = 16
    cache_state = defaultdict(lambda: [None] * NUM_WAYS)
    belady_rank_labels = []

    for row_idx, row in enumerate(rows):
        set_idx = int(row["set"])
        addr = row["full_addr"]
        hit = int(row["hit"])
        way = int(row["way"])
        recency_raw = [float(x) for x in row["recency_ranks"].split(";")]

        if hit == 0:
            current_ways = cache_state[set_idx]
            belady_victim_way = 0
            max_future_use = -1

            for w in range(NUM_WAYS):
                cached_addr = current_ways[w]
                if cached_addr is None:
                    belady_victim_way = w
                    break
                else:
                    next_use = float('inf')
                    for r_idx, f_addr, _ in set_accesses[set_idx]:
                        if r_idx > row_idx and f_addr == cached_addr:
                            next_use = r_idx
                            break
                    if next_use > max_future_use:
                        max_future_use = next_use
                        belady_victim_way = w

            # Convert belady_victim_way to its recency rank (0..15, where 15 is LRU)
            victim_rank = int(recency_raw[belady_victim_way]) if belady_victim_way < len(recency_raw) else 15
            belady_rank_labels.append(victim_rank)

            # Fill cache way with new address
            fill_way = way if way < NUM_WAYS else belady_victim_way
            current_ways[fill_way] = addr
        else:
            if way < NUM_WAYS:
                cache_state[set_idx][way] = addr
            belady_rank_labels.append(-1)

    return belady_rank_labels


def main():
    print("[1/4] Reading phase logs for 429.mcf and 403.gcc...")
    with open('datasets/champsim_phase_log_mcf.csv') as f:
        mcf_rows = list(csv.DictReader(f))
    with open('datasets/champsim_phase_log_gcc.csv') as f:
        gcc_rows = list(csv.DictReader(f))

    print(f"Loaded {len(mcf_rows)} mcf rows and {len(gcc_rows)} gcc rows.")

    def get_phase_feats(rows):
        return np.array([[
            float(r['rolling_miss_rate']),
            float(r['write_fraction']),
            float(r['reuse_fraction']),
            float(r['llc_access_rate'])
        ] for r in rows], dtype=np.float32)

    mcf_phase_x = get_phase_feats(mcf_rows)
    gcc_phase_x = get_phase_feats(gcc_rows)

    combined_phase_x = np.vstack([mcf_phase_x, gcc_phase_x])

    print("[2/4] Fitting K=4 StandardScaler + KMeans model...")
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(combined_phase_x)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaled_x)

    mcf_clusters = kmeans.predict(scaler.transform(mcf_phase_x))
    gcc_clusters = kmeans.predict(scaler.transform(gcc_phase_x))

    export_json = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "cluster_centers": kmeans.cluster_centers_.tolist()
    }
    with open("kmeans_phase_model.json", "w") as f:
        json.dump(export_json, f, indent=2)
    print("Saved 'kmeans_phase_model.json'.")

    print("[3/4] Computing True Belady Recency Ranks for mcf and gcc...")
    mcf_belady = calculate_true_belady_recency_ranks(mcf_rows)
    gcc_belady = calculate_true_belady_recency_ranks(gcc_rows)

    print("[4/4] Writing combined_phase_belady_dataset.csv...")
    out_fieldnames = list(mcf_rows[0].keys()) + ["trace_source", "phase_cluster", "belady_victim_way"]

    combined_rows = []
    
    for idx, r in enumerate(mcf_rows):
        if int(r["hit"]) == 0:
            r["trace_source"] = "429.mcf"
            r["phase_cluster"] = str(mcf_clusters[idx])
            r["belady_victim_way"] = str(mcf_belady[idx])
            combined_rows.append(r)

    for idx, r in enumerate(gcc_rows):
        if int(r["hit"]) == 0:
            r["trace_source"] = "403.gcc"
            r["phase_cluster"] = str(gcc_clusters[idx])
            r["belady_victim_way"] = str(gcc_belady[idx])
            combined_rows.append(r)

    with open('datasets/combined_phase_belady_dataset.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"Successfully generated 'datasets/combined_phase_belady_dataset.csv' with {len(combined_rows)} miss samples.")

if __name__ == "__main__":
    main()
