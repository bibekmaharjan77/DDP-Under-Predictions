# multibend_protocol.py
# Research-faithful implementation of MultiBend protocol using cluster-based hierarchy
# Evaluates move operation effectiveness via prediction error and stretch
# Plots aligned with tree-based experiment format: multiple prediction accuracies

import networkx as nx
import math
import os
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# ---------------------------- CONFIGURATION ----------------------------

PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]
NUM_TRIALS = 5

# ---------------------------- SUBMESH HIERARCHY ----------------------------

def generate_type1_submeshes(size):
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
    H = generate_type1_submeshes(size)
    H.update(generate_type2_submeshes(size))
    return H

# ---------------------------- GRAPH LOADING ----------------------------

def load_graph_from_directory(filename):
    filepath = os.path.join("graphs", "grid", filename)
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, lambda x: int(x))
    return G

# ---------------------------- METRICS ----------------------------

def calculate_error(pred_nodes, actual_nodes, G):
    diameter = nx.diameter(G)
    total_error = 0
    for u, v in zip(pred_nodes, actual_nodes):
        d = nx.shortest_path_length(G, source=u, target=v)
        total_error += d / diameter
    return total_error / len(pred_nodes)

def calculate_stretch(pred_nodes, actual_nodes, hierarchy, G):
    sum_G = 0
    sum_cluster = 0

    for u, v in zip(pred_nodes, actual_nodes):
        dist_G = nx.shortest_path_length(G, source=u, target=v)
        sum_G += dist_G
        found = False

        for level in sorted(hierarchy):
            for submesh in hierarchy[level]:
                if u in submesh and v in submesh:
                    H = G.subgraph(submesh)
                    if nx.is_connected(H):
                        dist_cluster = nx.shortest_path_length(H, source=u, target=v)
                        sum_cluster += dist_cluster
                        found = True
                    break
            if found:
                break

        if not found:
            sum_cluster += dist_G

    return sum_cluster / sum_G if sum_G > 0 else 1.0

# ---------------------------- SIMULATION ----------------------------

def simulate_multibend_eval(graph_file):
    G = load_graph_from_directory(graph_file)
    size = int(math.sqrt(len(G)))
    hierarchy = build_mesh_hierarchy(size)
    node_list = list(G.nodes())

    results = []

    for frac in PREDICTION_FRACTIONS:
        num_nodes = int(frac * len(G))
        for trial in range(NUM_TRIALS):
            # Randomly select predicted and actual nodes
            pred_nodes = random.sample(node_list, num_nodes)
            actual_nodes = random.sample(node_list, num_nodes)

            err = calculate_error(pred_nodes, actual_nodes, G)
            stretch = calculate_stretch(pred_nodes, actual_nodes, hierarchy, G)

            results.append((frac, trial, err, stretch))

    return results

# ---------------------------- PLOTTING ----------------------------

def plot_results(results):
    df = pd.DataFrame(results, columns=["Fraction", "Trial", "Error", "Stretch"])
    avg_df = df.groupby("Fraction").mean().reset_index()

    # Plot Error
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(avg_df["Fraction"], avg_df["Error"], marker='o', label="Error")
    plt.title("Error vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Error (Normalized)")
    plt.grid(True)
    plt.legend()

    # Plot Stretch
    plt.subplot(1, 2, 2)
    plt.plot(avg_df["Fraction"], avg_df["Stretch"], marker='o', color='orange', label="Stretch")
    plt.title("Stretch vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Stretch")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# ---------------------------- MAIN ----------------------------

def main():
    results = simulate_multibend_eval("64grid_diameter14test.edgelist")
    plot_results(results)

if __name__ == "__main__":
    main()
