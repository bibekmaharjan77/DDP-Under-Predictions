# run_spiral_random.py — Spiral Protocol implemented in run125.py structure

import networkx as nx
import os, random, math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# -------------------- Parameters --------------------
NUM_TRIALS = 50
PREDICTION_FRACTIONS = [0.0312, 0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# -------------------- Helper Functions --------------------
def choose_Vp(G, fraction):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    k = max(1, int(len(nodes) * fraction))
    return nodes[:k]

def sample_Q_within_diameter(G, Vp, error_fraction):
    diam = nx.diameter(G)
    Q = []
    for v in Vp:
        cutoff = max(1, int(diam * (1 - error_fraction)))
        reach = nx.single_source_shortest_path_length(G, v, cutoff=cutoff)
        if reach:
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

# -------------------- Evaluation --------------------
def simulate(G, cluster_leaders):
    results = []
    for error in ERROR_VALUES:
        for pf in PREDICTION_FRACTIONS:
            for _ in range(NUM_TRIALS):
                Vp = choose_Vp(G, pf)
                Q = sample_Q_within_diameter(G, Vp, error)
                if not Q:
                    continue

                correct = 0
                stretches = []

                for q in Q:
                    owner = random.choice(list(G.nodes()))
                    publish(owner, cluster_leaders)
                    stretch = lookup(q, owner, cluster_leaders, G)
                    stretches.append(stretch)
                    if q in Vp:
                        correct += 1

                error_rate = round(1 - correct / len(Q), 2)
                avg_stretch = sum(stretches) / len(stretches) if stretches else 1.0
                results.append((pf, error, error_rate, avg_stretch))

    df = pd.DataFrame(results, columns=["predicted_fraction", "error_value", "error_rate", "stretch"])
    df = df[df["predicted_fraction"].isin(PREDICTION_FRACTIONS)]
    df = df[df["error_value"].isin(ERROR_VALUES)]
    df.to_csv("spiral_results.csv", index=False)
    return df

# -------------------- Plotting --------------------
def plot_results(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for error in ERROR_VALUES:
        subset = df[df["error_value"] == error]
        means = subset.groupby("predicted_fraction").agg({"error_rate": "mean", "stretch": "mean"})
        means = means.reindex(PREDICTION_FRACTIONS)
        ax1.plot(PREDICTION_FRACTIONS, means["error_rate"], marker='o', label=f"{error:.1f}")
        ax2.plot(PREDICTION_FRACTIONS, means["stretch"], marker='x', label=f"{error:.1f}")

    ax1.set_title("Error vs Fraction of Predicted Nodes")
    ax1.set_xlabel("Fraction of Predicted Nodes")
    ax1.set_ylabel("Error (Max)")
    ax1.grid(True)
    ax1.legend(title="Error")

    ax2.set_title("Stretch vs Fraction of Predicted Nodes")
    ax2.set_xlabel("Fraction of Predicted Nodes")
    ax2.set_ylabel("Stretch")
    ax2.grid(True)
    ax2.legend(title="Stretch")

    plt.tight_layout()
    plt.show()

# -------------------- Graph Loader --------------------
def load_graph(filename):
    path = os.path.join("graphs", "random", filename)
    try:
        return nx.read_edgelist(path, nodetype=int)
    except Exception:
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
