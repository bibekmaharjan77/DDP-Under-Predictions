# multibend_protocol.py
# Fully functional version using a graphML grid from file (e.g., 256 nodes)
# Applies MultiBend approach: mesh-based clustering, spiral paths, and stretch/error analysis

import networkx as nx
import math
import os
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# ---------------------------- CONFIGURATION ----------------------------

PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_TRIALS = 2
DEBUG = True

# ---------------------------- GRAPH LOADING ----------------------------

def load_graph_from_directory(filename):
    filepath = os.path.join("graphs", "grid", filename)
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, lambda x: int(x))
    return G

# ---------------------------- CLUSTER CONSTRUCTION ----------------------------

def build_grid_mapping(G):
    side = int(math.sqrt(len(G)))
    id_to_coord = {}
    coord_to_id = {}
    for idx, node in enumerate(sorted(G.nodes())):
        x, y = divmod(idx, side)
        id_to_coord[node] = (x, y)
        coord_to_id[(x, y)] = node
    return id_to_coord, coord_to_id, side

def build_mesh_hierarchy(G):
    id_to_coord, coord_to_id, size = build_grid_mapping(G)
    hierarchy = defaultdict(list)
    levels = int(math.log2(size)) + 1

    for level in range(levels):
        block_size = 2 ** level
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                coords = [(x, y) for x in range(i, min(i+block_size, size))
                                  for y in range(j, min(j+block_size, size))]
                cluster_nodes = [coord_to_id[(x, y)] for (x, y) in coords if (x, y) in coord_to_id]
                if cluster_nodes:
                    hierarchy[(level, 2)].append(set(cluster_nodes))

    for level in range(1, levels):
        block_size = 2 ** level
        offset = block_size // 2
        for i in range(-offset, size, block_size):
            for j in range(-offset, size, block_size):
                coords = [(x, y) for x in range(i, i + block_size)
                                  for y in range(j, j + block_size)
                                  if 0 <= x < size and 0 <= y < size]
                cluster_nodes = [coord_to_id[(x, y)] for (x, y) in coords if (x, y) in coord_to_id]
                if cluster_nodes:
                    hierarchy[(level, 1)].append(set(cluster_nodes))

    all_nodes = set(G.nodes())
    hierarchy[(levels, 2)].append(all_nodes)

    return hierarchy

def assign_cluster_leaders(hierarchy):
    leader_map = defaultdict(list)
    for level_type, clusters in hierarchy.items():
        for cluster in clusters:
            leader = min(cluster)
            leader_map[level_type].append((leader, cluster))
    return leader_map

def get_spiral_path(node, leader_map):
    path = [node]
    visited = set(path)

    for level_type in sorted(leader_map.keys()):
        clusters_with_node = [(leader, cluster) for leader, cluster in leader_map[level_type] if node in cluster]
        if clusters_with_node:
            leader = min(clusters_with_node, key=lambda x: len(x[1]))[0]
            if leader not in visited:
                path.append(leader)
                visited.add(leader)

    max_level = max(leader_map.keys(), key=lambda x: x[0])
    root_candidates = [leader for leader, cluster in leader_map[max_level] if node in cluster]
    if root_candidates:
        root_leader = root_candidates[0]
        if root_leader not in visited:
            path.append(root_leader)

    return path

# ---------------------------- METRICS & SIMULATION ----------------------------

def calculate_error(pred_nodes, actual_nodes, G):
    diameter = nx.diameter(G)
    total_error = 0
    for u, v in zip(pred_nodes, actual_nodes):
        d = nx.shortest_path_length(G, source=u, target=v)
        total_error += d / diameter
    return total_error / len(pred_nodes)

def calculate_stretch_spiral(pred_nodes, actual_nodes, leader_map, G):
    sum_spiral = 0
    sum_shortest = 0
    count = 0
    for u, v in zip(pred_nodes, actual_nodes):
        if u == v:
            continue
        path_u = get_spiral_path(u, leader_map)
        path_v = get_spiral_path(v, leader_map)
        common = set(path_u) & set(path_v)
        if not common:
            continue
        lca = next(n for n in reversed(path_u) if n in common)
        try:
            idx_u = path_u.index(lca)
            idx_v = path_v.index(lca)
            spiral_path = path_u[:idx_u+1] + list(reversed(path_v[:idx_v]))
            dist_spiral = sum(nx.shortest_path_length(G, spiral_path[i], spiral_path[i+1]) for i in range(len(spiral_path)-1))
            dist_shortest = nx.shortest_path_length(G, source=u, target=v)
        except:
            continue
        if dist_shortest == 0:
            continue
        sum_spiral += dist_spiral
        sum_shortest += dist_shortest
        count += 1
    return sum_spiral / sum_shortest if sum_shortest > 0 else 1.0

def simulate_multibend_eval(graph_file):
    G = load_graph_from_directory(graph_file)
    hierarchy = build_mesh_hierarchy(G)
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
                err = calculate_error(pred_nodes, actual_nodes, G)
                stretch = calculate_stretch_spiral(pred_nodes, actual_nodes, leader_map, G)
                results.append((frac, error_rate, trial, err, stretch))
    return results

def plot_results(results):
    df = pd.DataFrame(results, columns=["Fraction", "ErrorRate", "Trial", "Error", "Stretch"])
    avg_df = df.groupby(["Fraction", "ErrorRate"]).mean().reset_index()
    plt.figure(figsize=(12, 6))
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

def main():
    results = simulate_multibend_eval("256grid_diameter30test.graphml")
    plot_results(results)

if __name__ == "__main__":
    main()
