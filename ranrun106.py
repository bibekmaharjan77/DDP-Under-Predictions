# run_spiral_random.py — Spiral Protocol implemented in run125.py structure

import networkx as nx
import os, random, math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# -------------------- Parameters --------------------
NUM_TRIALS = 50
PREDICTION_FRACTIONS = [0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES = [9999, 10, 5, 3.3, 2.5]

# -------------------- Helper Functions --------------------
def choose_Vp(G, fraction):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    k = max(1, int(len(nodes) * fraction))
    return nodes[:k]

def sample_Q_within_diameter(G, Vp, error_factor):
    diam = nx.diameter(G)
    Q = []
    for v in Vp:
        reach = nx.single_source_shortest_path_length(G, v, cutoff=max(1, diam // int(error_factor)))
        Q.append(random.choice(list(reach.keys())))
    return Q

def count_duplicates(lst):
    counts = defaultdict(int)
    for item in lst:
        counts[item] += 1
    return {k: v for k, v in counts.items() if v > 1}

# -------------------- Spiral Cluster Construction --------------------
def build_sparse_cover(G, beta=2):
    if not nx.is_connected(G):
        print("Warning: Graph is not connected. Taking the largest connected component.")
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    hierarchy = []
    max_radius = nx.diameter(G)
    r = 1
    covered_global = set()
    while r <= max_radius:
        remaining = set(G.nodes()) - covered_global
        level_clusters = []
        while remaining:
            center = random.choice(list(remaining))
            ball = set(nx.single_source_shortest_path_length(G, center, cutoff=r).keys())
            if not ball:
                remaining.remove(center)
                continue
            level_clusters.append((center, ball))
            remaining -= ball
        hierarchy.append(level_clusters)
        covered_global.update(set().union(*[c[1] for c in level_clusters]))
        r *= beta
    return hierarchy

# -------------------- Assign Leaders to Clusters --------------------
def assign_cluster_leaders(cluster_hierarchy):
    cluster_leaders = defaultdict(dict)  # level -> node -> leader
    for level, clusters in enumerate(cluster_hierarchy):
        for leader, cluster in clusters:
            for node in cluster:
                cluster_leaders[level][node] = leader
    return cluster_leaders

# -------------------- Print Clusters --------------------
def print_clusters(cluster_hierarchy):
    for level, clusters in enumerate(cluster_hierarchy):
        print(f"Level {level} has {len(clusters)} clusters:")
        for i, (leader, members) in enumerate(clusters):
            print(f"  Cluster {i}: Leader={leader}, Size={len(members)}")

# -------------------- Spiral Path Construction --------------------
def get_spiral_path(node, cluster_leaders):
    path = [node]
    visited = set(path)
    for level in sorted(cluster_leaders):
        if node in cluster_leaders[level]:
            leader = cluster_leaders[level][node]
            if leader not in visited:
                path.append(leader)
                visited.add(leader)
                node = leader
    return path

# -------------------- Publish and Lookup --------------------
down_links = {}

def publish(owner, cluster_leaders):
    global down_links
    down_links.clear()
    path = get_spiral_path(owner, cluster_leaders)
    for i in range(len(path)-1, 0, -1):
        down_links[path[i]] = path[i-1]

def lookup(requester, owner, cluster_leaders, G):
    path_up = get_spiral_path(requester, cluster_leaders)
    up_hops = 0
    intersection = None

    for i in range(len(path_up)-1):
        u, v = path_up[i], path_up[i+1]
        up_hops += nx.shortest_path_length(G, u, v)
        if v in down_links:
            intersection = v
            break
    if intersection is None:
        intersection = path_up[-1]

    down_hops = 0
    cur = intersection
    visited = {cur}
    while cur != owner:
        if cur in down_links and down_links[cur] not in visited:
            nxt = down_links[cur]
            down_hops += nx.shortest_path_length(G, cur, nxt)
            visited.add(nxt)
            cur = nxt
        else:
            down_hops += nx.shortest_path_length(G, cur, owner)
            break

    total_dist = up_hops + down_hops
    optimal = nx.shortest_path_length(G, requester, owner)
    return total_dist / optimal if optimal > 0 else 1.0

# -------------------- Simulation Wrapper --------------------
def simulate(G, cluster_leaders):
    df = measure_stretch(G, cluster_leaders)
    df.to_csv("spiral_results.csv", index=False)
    return df

# -------------------- Evaluation --------------------
def measure_stretch(G, cluster_leaders):
    results = []
    for error in ERROR_VALUES:
        for pf in PREDICTION_FRACTIONS:
            for _ in range(NUM_TRIALS):
                owner = random.choice(list(G.nodes()))
                publish(owner, cluster_leaders)
                Vp = choose_Vp(G, pf)
                Q = sample_Q_within_diameter(G, Vp, error)
                for q in Q:
                    if q == owner:
                        continue
                    stretch = lookup(q, owner, cluster_leaders, G)
                    err_rate = 0.0 if error > 1000 else round(1.0 / error, 2)
                    results.append((pf, err_rate, stretch))
                    owner = q
                    publish(owner, cluster_leaders)
    return pd.DataFrame(results, columns=["predicted_fraction", "error_rate", "stretch"])

# -------------------- Plotting --------------------
def plot_results(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Error vs Fraction
    grouped_error = df.groupby(['predicted_fraction', 'error_rate'])['stretch'].max().unstack()
    grouped_error.plot(ax=ax1, marker='o')
    ax1.set_title("Error vs Fraction of Predicted Nodes")
    ax1.set_xlabel("Fraction of Predicted Nodes")
    ax1.set_ylabel("Error (Max)")
    ax1.grid(True)
    ax1.legend(title="Error")

    # Stretch vs Fraction
    grouped_stretch = df.groupby(['predicted_fraction', 'error_rate'])['stretch'].mean().unstack()
    grouped_stretch.plot(ax=ax2, marker='x')
    ax2.set_title("Stretch vs Fraction of Predicted Nodes")
    ax2.set_xlabel("Fraction of Predicted Nodes")
    ax2.set_ylabel("Stretch")
    ax2.grid(True)
    ax2.legend(title="Stretch")

    plt.tight_layout()
    plt.show()

    # Error Rate vs Prediction Fraction
    grouped2 = df.groupby(['error_rate', 'predicted_fraction'])['stretch'].mean().unstack()
    grouped2.plot(marker='x')
    plt.title("Spiral Protocol: Avg. Stretch vs Error Rate")
    plt.xlabel("Error Rate")
    plt.ylabel("Average Stretch")
    plt.grid(True)
    plt.legend(title="Prediction Fraction")
    plt.tight_layout()
    plt.show()

# -------------------- Graph Loader --------------------
def load_graph(filename):
    path = os.path.join("graphs", "random", filename)
    try:
        return nx.read_edgelist(path, nodetype=int)
    except Exception as e:
        print("Warning: read_edgelist failed. Attempting to read as GraphML.")
        return nx.read_graphml(path)

# -------------------- Main --------------------
if __name__ == "__main__":
    fname = "64random_diameter38test.edgelist"
    G = load_graph(fname)
    clusters = build_sparse_cover(G)
    print_clusters(clusters)
    cluster_leaders = assign_cluster_leaders(clusters)
    df = simulate(G, cluster_leaders)
    plot_results(df)
