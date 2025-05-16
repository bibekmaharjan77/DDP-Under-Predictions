# good version of run110.py

# multibend_protocol.py
# Updated to ensure stretch is measured correctly through full spiral paths
# Implements MultiBend paper logic: cluster construction (Type-1, Type-2, and root), spiral paths, and routing evaluation

import networkx as nx                    # NetworkX handles graph data structures and algorithms
import math                              # Math provides logarithmic and square root functions
import os                                # OS for file path manipulations
import random                            # Random sampling for simulation experiments
import matplotlib.pyplot as plt          # Matplotlib for visualizing error and stretch plots
from collections import defaultdict      # Defaultdict for grouping clusters and leader maps
import pandas as pd                      # Pandas for aggregating and analyzing simulation results

# ---------------------------- CONFIGURATION ----------------------------

# Fractions of nodes used as predicted requesters
PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]

# Error rates (inaccuracy of predictions): 0.0 = perfect prediction, 0.5 = total mismatch
ERROR_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Number of trials to average for each error rate and prediction fraction
NUM_TRIALS = 2

# Enable or disable detailed debug output
DEBUG = True

# ---------------------------- SUBMESH HIERARCHY ----------------------------

def generate_type1_submeshes(size):
    """
    Creates Type-1 submeshes (aligned blocks) for each level.
    At level l, blocks are of size 2^l x 2^l.
    """
    levels = int(math.log2(size)) + 1
    hierarchy = defaultdict(list)
    for level in range(levels):
        block_size = 2 ** level
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                nodes = set((x, y) for x in range(i, min(i+block_size, size))
                                     for y in range(j, min(j+block_size, size)))
                hierarchy[(level, 2)].append(nodes)
    return hierarchy

def generate_type2_submeshes(size):
    """
    Creates Type-2 submeshes (offset blocks) for each level > 0.
    These are shifted by half the block size to create overlap.
    """
    levels = int(math.log2(size))
    hierarchy = defaultdict(list)
    for level in range(1, levels):
        block_size = 2 ** level
        offset = block_size // 2
        for i in range(-offset, size, block_size):
            for j in range(-offset, size, block_size):
                nodes = set((x, y) for x in range(i, i + block_size)
                                     for y in range(j, j + block_size)
                            if 0 <= x < size and 0 <= y < size)
                if nodes:
                    hierarchy[(level, 1)].append(nodes)
    return hierarchy

def build_mesh_hierarchy(size):
    """
    Combines Type-1 and Type-2 submeshes to form a full hierarchy.
    Adds a top-level root cluster containing all nodes.
    """
    H = generate_type1_submeshes(size)
    H.update(generate_type2_submeshes(size))
    all_nodes = set((x, y) for x in range(size) for y in range(size))
    max_level = int(math.log2(size)) + 1
    H[(max_level, 2)].append(all_nodes)
    return H

def assign_cluster_leaders(hierarchy):
    """
    Assigns a leader node (lowest ID) for each cluster at every level.
    Returns a map from level_type to list of (leader, cluster).
    """
    leader_map = defaultdict(list)
    for level_type, clusters in hierarchy.items():
        for cluster in clusters:
            leader = min(cluster)
            leader_map[level_type].append((leader, cluster))
    return leader_map

# ---------------------------- PATH EXTRACTION ----------------------------

def get_spiral_path(node, leader_map):
    """
    Constructs the spiral path for a node by walking up through its cluster leaders.
    One leader per level is added to the path, including the root.
    """
    path = [node]
    visited = set(path)

    for level_type in sorted(leader_map):
        for leader, cluster in leader_map[level_type]:
            if node in cluster:
                if leader not in visited:
                    path.append(leader)
                    visited.add(leader)
                break  # Stop at first match in that level

    # Ensure the top-level leader is included
    max_level = max(leader_map.keys(), key=lambda x: x[0])
    for leader, cluster in leader_map[max_level]:
        if node in cluster and leader not in visited:
            path.append(leader)
            break

    return path

# ---------------------------- GRAPH LOADING ----------------------------

def load_graph_from_directory(filename):
    """
    Loads a .graphml file from the given path inside 'graphs/grid'.
    Relabels nodes with integer IDs for consistent indexing.
    """
    filepath = os.path.join("graphs", "grid", filename)
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, lambda x: int(x))
    return G

# ---------------------------- METRICS ----------------------------

def calculate_error(pred_nodes, actual_nodes, G):
    """
    Calculates prediction error as average normalized distance between
    predicted and actual requester nodes, normalized by graph diameter.
    """
    diameter = nx.diameter(G)
    total_error = 0
    for u, v in zip(pred_nodes, actual_nodes):
        d = nx.shortest_path_length(G, source=u, target=v)
        total_error += d / diameter
    return total_error / len(pred_nodes)

def calculate_stretch_spiral(pred_nodes, actual_nodes, leader_map, G):
    """
    Calculates stretch for each requester pair using spiral paths
    (paths up through cluster leaders to their lowest common ancestor).
    Stretch = spiral distance / shortest distance in G.
    """
    sum_spiral = 0
    sum_shortest = 0
    count = 0

    for u, v in zip(pred_nodes, actual_nodes):
        if u == v:
            if DEBUG:
                print(f"[WARN] Skipping identical node pair (u={u}, v={v})")
            continue

        path_u = get_spiral_path(u, leader_map)
        path_v = get_spiral_path(v, leader_map)

        common = set(path_u) & set(path_v)
        if not common:
            if DEBUG:
                print(f"[WARN] No common leader in spiral paths for (u={u}, v={v})")
            continue

        lca = next(n for n in reversed(path_u) if n in common)

        if lca is None:
            if DEBUG:
                print(f"[WARN] No LCA found for pair (u={u}, v={v})")
            continue


        try:
            idx_u = path_u.index(lca)
            idx_v = path_v.index(lca)
            spiral_path = path_u[:idx_u+1] + list(reversed(path_v[:idx_v]))
        except ValueError:
            if DEBUG:
                print(f"[ERROR] LCA index not found for u={u}, v={v}, lca={lca}")
            continue

        if len(spiral_path) <= 2:
            if DEBUG:
                print(f"[WARN] Spiral path is trivial (length 2) for (u={u}, v={v}), skipping.")
            continue

        try:
            dist_spiral = sum(nx.shortest_path_length(G, spiral_path[i], spiral_path[i+1])
                              for i in range(len(spiral_path)-1))
            dist_shortest = nx.shortest_path_length(G, source=u, target=v)
        except nx.NetworkXNoPath:
            if DEBUG:
                print(f"[ERROR] No path in G between nodes in spiral path or from u={u} to v={v}")
            continue

        if dist_shortest == 0:
            if DEBUG:
                print(f"[WARN] Shortest path from {u} to {v} is zero, skipping.")
            continue

        stretch_value = dist_spiral / dist_shortest

        if DEBUG:
            print(f"\n[DEBUG] u={u}, v={v}")
            print(f"Spiral path: {spiral_path}")
            print(f"Spiral path length: {len(spiral_path)}")
            print(f"Spiral path distance: {dist_spiral}")
            print(f"Shortest path distance: {dist_shortest}")
            print(f"Stretch: {stretch_value:.3f}")

        sum_spiral += dist_spiral
        sum_shortest += dist_shortest
        count += 1

    if DEBUG:
        print(f"\n[INFO] Total valid pairs used: {count}")
        print(f"[INFO] Total spiral distance: {sum_spiral}")
        print(f"[INFO] Total shortest distance: {sum_shortest}")

    return sum_spiral / sum_shortest if sum_shortest > 0 else 1.0

# ---------------------------- SIMULATION ----------------------------

def simulate_multibend_eval(graph_file):
    """
    Main experiment loop.
    For each error rate and prediction fraction:
      - Generates predicted and actual requester node lists
      - Computes error and spiral stretch
    """
    G = load_graph_from_directory(graph_file)
    size = int(math.sqrt(len(G)))
    hierarchy = build_mesh_hierarchy(size)
    leader_map = assign_cluster_leaders(hierarchy)
    node_list = list(G.nodes())
    results = []

    for error_rate in ERROR_VALUES:
        for frac in PREDICTION_FRACTIONS:
            num_nodes = int(frac * len(G))
            for trial in range(NUM_TRIALS):
                overlap = int((1 - error_rate) * num_nodes)
                shared = random.sample(node_list, overlap)
                remaining_pool = [n for n in node_list if n not in shared]
                remaining_pred = random.sample(remaining_pool, num_nodes - overlap)
                remaining_actual = random.sample(remaining_pool, num_nodes - overlap)

                pred_nodes = shared + remaining_pred
                actual_nodes = shared + remaining_actual
                random.shuffle(actual_nodes)

                if DEBUG:
                    print(f"\n[INFO] Trial {trial+1}, Fraction {frac}, Error Rate {error_rate}")
                    print(f"Predicted nodes: {pred_nodes}")
                    print(f"Actual nodes:    {actual_nodes}")

                err = calculate_error(pred_nodes, actual_nodes, G)
                stretch = calculate_stretch_spiral(pred_nodes, actual_nodes, leader_map, G)

                results.append((frac, error_rate, trial, err, stretch))

    return results

# ---------------------------- PLOTTING ----------------------------

def plot_results(results):
    """
    Aggregates and plots average error and stretch metrics.
    Produces two subplots:
      - Error vs. Prediction Fraction (one curve per error rate)
      - Stretch vs. Prediction Fraction (one curve per error rate)
    """
    df = pd.DataFrame(results, columns=["Fraction", "ErrorRate", "Trial", "Error", "Stretch"])
    avg_df = df.groupby(["Fraction", "ErrorRate"]).mean().reset_index()

    plt.figure(figsize=(12, 6))

    # Error Plot: how far off predicted requesters are from actual ones
    plt.subplot(1, 2, 1)
    for error_rate in ERROR_VALUES:
        sub = avg_df[avg_df["ErrorRate"] == error_rate]
        plt.plot(sub["Fraction"], sub["Error"], marker='o', label=f"{error_rate:.1f} Error")
    plt.title("Error vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Error (Normalized)")
    plt.xticks(PREDICTION_FRACTIONS)
    plt.yticks(ERROR_VALUES)
    plt.grid(True)
    plt.legend()

    # Stretch Plot: how efficient spiral paths are compared to shortest paths
    plt.subplot(1, 2, 2)
    for error_rate in ERROR_VALUES:
        sub = avg_df[avg_df["ErrorRate"] == error_rate]
        plt.plot(sub["Fraction"], sub["Stretch"], marker='o', label=f"{error_rate:.1f} Stretch")
    plt.title("Stretch vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Stretch")
    plt.xticks(PREDICTION_FRACTIONS)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# ---------------------------- MAIN ----------------------------

def main():
    """
    Entry point for executing the full evaluation pipeline:
      - Loads graph
      - Runs simulation
      - Plots results
    """
    results = simulate_multibend_eval("64grid_diameter14test.edgelist")  # Adjust to your file name
    plot_results(results)

if __name__ == "__main__":
    main()
