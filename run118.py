# multibend_protocol.py
# Updated to ensure stretch is measured correctly through full spiral paths
# Implements MultiBend paper logic: cluster construction (Type-1, Type-2, and root), spiral paths, and routing evaluation

import networkx as nx
import math
import os
import random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd

# ---------------------------- CONFIGURATION ----------------------------

PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES = [999999999999999999999999999999999999, 10, 5, 3.3333333333333335, 2.5, 2]
ERROR_VALUES_2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_TRIALS = 2
DEBUG = True

# ---------------------------- CLUSTERING ----------------------------

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
    all_nodes = set((x, y) for x in range(size) for y in range(size))
    max_level = int(math.log2(size)) + 1
    H[(max_level, 2)].append(all_nodes)
    return H

def assign_cluster_leaders(hierarchy):
    leader_map = defaultdict(list)
    for level_type, clusters in hierarchy.items():
        for cluster in clusters:
            leader = min(cluster)
            leader_map[level_type].append((leader, cluster))
    return leader_map

# ---------------------------- SPIRAL PATH ----------------------------

def get_cluster_path(node, hierarchy, leader_map):
    path = [node]
    visited = set([node])
    for level in sorted(hierarchy):
        for cluster in hierarchy[level]:
            if node in cluster:
                leader = min(cluster)
                if leader not in visited:
                    path.append(leader)
                    visited.add(leader)
                break
    return path

# ---------------------------- GRAPH ----------------------------

def load_graph_from_directory(filename):
    filepath = os.path.join("graphs", "grid", filename)
    G = nx.read_graphml(filepath)
    G = nx.relabel_nodes(G, lambda x: int(x))
    return G

# ---------------------------- METRICS ----------------------------

def calculate_error_updated(Vp, Q, G_example):
    diameter_of_G = nx.diameter(G_example, weight='weight')
    errors = []
    for req, pred in zip(Q, Vp):
        dist = nx.shortest_path_length(G_example, source=req, target=pred, weight='weight')
        error = dist / diameter_of_G
        errors.append(error)
    total_max_error = max(errors) if errors else 0
    return total_max_error

def calculate_stretch_cluster(Vp, Q, hierarchy, leader_map, G):
    total_spiral = 0
    total_shortest = 0
    count = 0
    for owner, requester in zip(Vp, Q):
        path_up = get_cluster_path(requester, hierarchy, leader_map)
        path_down = get_cluster_path(owner, hierarchy, leader_map)
        lca_set = set(path_up) & set(path_down)
        if not lca_set:
            continue
        lca = next((n for n in reversed(path_up) if n in lca_set), None)
        if lca is None:
            continue
        idx_up = path_up.index(lca)
        idx_down = path_down.index(lca)
        spiral_path = path_up[:idx_up+1] + list(reversed(path_down[:idx_down]))
        spiral_dist = sum(nx.shortest_path_length(G, source=spiral_path[i], target=spiral_path[i+1])
                          for i in range(len(spiral_path)-1))
        try:
            shortest = nx.shortest_path_length(G, source=owner, target=requester)
        except:
            continue
        if shortest == 0:
            continue
        total_spiral += spiral_dist
        total_shortest += shortest
        count += 1
    return total_spiral / total_shortest if total_shortest > 0 else 1.0

# ---------------------------- SAMPLING ----------------------------

def choose_Vp(G, fraction):
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    random.shuffle(nodes)
    vp_size = int(len(nodes) * fraction)
    original_Vp = list(random.choices(nodes, k=vp_size))
    random.shuffle(original_Vp)
    reduced_Vp = set(original_Vp)
    reduced_Vp = list(reduced_Vp)
    random.shuffle(reduced_Vp)
    remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(remaining))
    reduced_Vp.insert(random.randint(0, len(reduced_Vp)), owner)
    return original_Vp

def count_duplicates(input_list):
    counts = Counter(input_list)
    return {element: count for element, count in counts.items() if count > 1}

def sample_Q_within_diameter(G, Vp, error_cutoff):
    diam = nx.diameter(G, weight='weight')
    for _ in range(100000):
        Q = []
        for v in Vp:
            dist_map = nx.single_source_dijkstra_path_length(G, v, cutoff=float(diam / error_cutoff), weight="weight")
            Q.append(random.choice(list(dist_map.keys())))
        dup_counts = count_duplicates(Q)
        extra_dups = sum(cnt for cnt in dup_counts.values())
        current_overlap = extra_dups / len(Q) * 100
        if current_overlap <= 100:
            return Q
    random.shuffle(Q)
    return Q

# ---------------------------- SIMULATION ----------------------------

def simulate_multibend_eval(graph_file):
    G = load_graph_from_directory(graph_file)
    size = int(math.sqrt(len(G)))
    hierarchy = build_mesh_hierarchy(size)
    leader_map = assign_cluster_leaders(hierarchy)
    results = []
    for error_rate in ERROR_VALUES:
        for frac in PREDICTION_FRACTIONS:
            pred_nodes = choose_Vp(G, frac)
            actual_nodes = sample_Q_within_diameter(G, pred_nodes, error_rate)
            err = calculate_error_updated(pred_nodes, actual_nodes, G)
            stretch = calculate_stretch_cluster(pred_nodes, actual_nodes, hierarchy, leader_map, G)
            eff_err = 0 if error_rate > 15 else 1 / error_rate
            results.append((frac, eff_err, err, stretch))
    return results

# ---------------------------- PLOTTING ----------------------------

def plot_results(results):
    df = pd.DataFrame(results, columns=["Fraction", "ErrorRate", "Error", "Stretch"])
    df.to_excel("excel/data.xlsx")
    avg_df = df.groupby(["Fraction", "ErrorRate"]).mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for error_rate in ERROR_VALUES_2:
        sub = avg_df[avg_df["ErrorRate"] == error_rate]
        plt.plot(sub["Fraction"], sub["Error"], marker='o', label=f"{error_rate:.1f} Error")
    plt.title("Error vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Error (Max)")
    plt.xticks(PREDICTION_FRACTIONS)
    plt.yticks(ERROR_VALUES_2)
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 2, 2)
    for error_rate in ERROR_VALUES_2:
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
    results = simulate_multibend_eval("256grid_diameter30test.edgelist")
    plot_results(results)

if __name__ == "__main__":
    main()
