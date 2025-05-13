# multibend_protocol.py
# Research-faithful implementation of the MultiBend protocol for 2D grid graphs
# Modified to support only move operations, predictions, and compute error/stretch curves
# No load-balancing or directory pointers — purely structural

import networkx as nx
import math
import os
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------- CONFIGURATION ----------------------------

PREDICTION_FRACTIONS = [0.25, 0.5, 0.75]  # Fraction of nodes to predict
NUM_TRIALS = 10  # Number of trials per prediction setting

# ---------------------------- SUBMESH TYPES ----------------------------

def generate_type1_submeshes(size):
    """
    Create type-1 submeshes: regular partitioning into 2^i x 2^i blocks.
    Each level (i) creates non-overlapping square subgrids of size 2^i.
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
    Create type-2 submeshes: shifted versions of type-1 by half a block.
    These overlap with type-1 meshes to give a hierarchical cover.
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
    Combines type-1 and type-2 submeshes into a hierarchical mesh structure.
    Returns a dictionary of level keys -> list of sets of node tuples.
    """
    H = generate_type1_submeshes(size)
    H.update(generate_type2_submeshes(size))
    return H

# ---------------------------- GRAPH UTILITIES ----------------------------

def load_graph_from_directory(filename):
    """
    Load a GraphML-formatted grid graph from graphs/grid directory.
    Relabel node IDs from string to integer if needed.
    """
    filepath = os.path.join("graphs", "grid", filename)
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, lambda x: int(x))
    return G

# ---------------------------- EXPERIMENTAL METRICS ----------------------------

def calculate_error(pred_nodes, actual_nodes, G):
    """
    Error is defined as average normalized distance between pairs of predicted and actual nodes.
    Normalized by the graph's diameter.
    """
    diameter = nx.diameter(G)
    total_error = 0
    for u, v in zip(pred_nodes, actual_nodes):
        d = nx.shortest_path_length(G, source=u, target=v)
        total_error += d / diameter
    return total_error / len(pred_nodes)

def calculate_stretch(pred_nodes, actual_nodes, hierarchy, G):
    """
    Stretch is the sum of distances in the cluster hierarchy vs in G.
    Each node belongs to multiple submeshes — match them via lowest common level.
    """
    sum_G = sum(nx.shortest_path_length(G, source=u, target=v) for u, v in zip(pred_nodes, actual_nodes))
    sum_cluster = 0

    for u, v in zip(pred_nodes, actual_nodes):
        found = False
        for level in sorted(hierarchy):
            for submesh in hierarchy[level]:
                if u in submesh and v in submesh:
                    # Use intra-cluster distance as approximate proxy
                    H = G.subgraph(submesh)
                    if nx.is_connected(H):
                        sum_cluster += nx.shortest_path_length(H, source=u, target=v)
                        found = True
                    break
            if found:
                break
        else:
            # fallback to G if no shared submesh found
            sum_cluster += nx.shortest_path_length(G, source=u, target=v)

    return sum_cluster / sum_G if sum_G > 0 else 1.0

# ---------------------------- SIMULATION ----------------------------

def simulate_multibend_eval(graph_file):
    """
    Main loop to evaluate error and stretch for move operations using cluster hierarchy.
    """
    G = load_graph_from_directory(graph_file)
    size = int(math.sqrt(len(G)))
    hierarchy = build_mesh_hierarchy(size)
    node_list = list(G.nodes())

    results = []

    for frac in PREDICTION_FRACTIONS:
        num_nodes = int(frac * len(G))
        for trial in range(NUM_TRIALS):
            pred_nodes = random.sample(node_list, num_nodes)
            actual_nodes = random.sample(node_list, num_nodes)

            err = calculate_error(pred_nodes, actual_nodes, G)
            stretch = calculate_stretch(pred_nodes, actual_nodes, hierarchy, G)

            results.append((frac, trial, err, stretch))

    return results

# ---------------------------- PLOTTING ----------------------------

def plot_results(results):
    """
    Plot error and stretch as a function of prediction fraction.
    """
    import pandas as pd
    df = pd.DataFrame(results, columns=["Fraction", "Trial", "Error", "Stretch"])

    avg_df = df.groupby("Fraction").mean().reset_index()

    plt.figure()
    plt.plot(avg_df["Fraction"], avg_df["Error"], marker='o')
    plt.title("Prediction Error vs. Fraction")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Error")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(avg_df["Fraction"], avg_df["Stretch"], marker='x')
    plt.title("Stretch vs. Fraction")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Stretch")
    plt.grid(True)
    plt.show()

# ---------------------------- MAIN ----------------------------

def main():
    results = simulate_multibend_eval("64grid_diameter14test.edgelist")
    plot_results(results)

if __name__ == "__main__":
    main()
