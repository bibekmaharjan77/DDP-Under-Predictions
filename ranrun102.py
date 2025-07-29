# spiral_protocol.py — Spiral implementation for general graphs, run125-style

import networkx as nx
import math, random
from collections import defaultdict, deque
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Parameters --------------------
NUM_TRIALS = 10
PREDICTION_FRACTIONS = [0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES = [9999, 10, 5, 3.3, 2.5]

# -------------------- Sparse Cover Hierarchy --------------------
def build_sparse_cover(G, k=2):
    levels = []
    covered = set()
    nodes = set(G.nodes())
    radius = 1
    while covered != nodes:
        level_clusters = []
        remaining = nodes - covered
        while remaining:
            center = random.choice(list(remaining))
            cluster = set(nx.single_source_shortest_path_length(G, center, cutoff=radius).keys())
            level_clusters.append((center, cluster))
            covered.update(cluster)
            remaining = nodes - covered
        levels.append(level_clusters)
        radius *= k
    return levels

# -------------------- Spiral Path Construction --------------------
def get_spiral_path(node, hierarchy):
    path = [node]
    visited = set([node])
    for level in hierarchy:
        for leader, cluster in level:
            if node in cluster and leader not in visited:
                path.append(leader)
                visited.add(leader)
                break
    return path

# -------------------- Publish and Lookup --------------------
down_links = {}


def publish(owner, hierarchy):
    global down_links
    down_links.clear()
    path = get_spiral_path(owner, hierarchy)
    for i in range(len(path)-1, 0, -1):
        down_links[path[i]] = path[i-1]


def lookup(requester, owner, hierarchy, G):
    path_up = get_spiral_path(requester, hierarchy)
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
    optimal_dist = nx.shortest_path_length(G, requester, owner)
    stretch = total_dist / optimal_dist if optimal_dist > 0 else 1.0
    return stretch

# -------------------- Predictions and Request Sampling --------------------
def choose_predicted_nodes(G, fraction):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    k = max(1, int(len(nodes) * fraction))
    return nodes[:k]


def sample_requests(G, Vp, error_factor):
    diam = nx.diameter(G)
    Q = []
    for v in Vp:
        reach = nx.single_source_shortest_path_length(G, v, cutoff=max(1, diam // int(error_factor)))
        Q.append(random.choice(list(reach.keys())))
    return Q

# -------------------- Simulation --------------------
def measure_stretch(G, hierarchy):
    records = []
    for error in ERROR_VALUES:
        for frac in PREDICTION_FRACTIONS:
            for _ in range(NUM_TRIALS):
                owner = random.choice(list(G.nodes()))
                publish(owner, hierarchy)
                Vp = choose_predicted_nodes(G, frac)
                Q = sample_requests(G, Vp, error)
                for req in Q:
                    if req == owner:
                        continue
                    stretch = lookup(req, owner, hierarchy, G)
                    err_rate = 0.0 if error > 1000 else round(1.0 / error, 2)
                    records.append((frac, err_rate, stretch))
                    owner = req
                    publish(owner, hierarchy)
    return pd.DataFrame(records, columns=["predicted_fraction", "error_rate", "stretch"])

# -------------------- Plotting --------------------
def plot_stretch(df):
    grouped = df.groupby(['predicted_fraction', 'error_rate'])['stretch'].mean().unstack()
    grouped.plot(marker='o')
    plt.title("Average Stretch vs. Prediction Fraction")
    plt.xlabel("Prediction Fraction")
    plt.ylabel("Average Stretch")
    plt.grid(True)
    plt.legend(title="Error Rate")
    plt.tight_layout()
    plt.show()

# -------------------- Graph Loader --------------------
def load_graph_from_edgelist(path):
    G = nx.read_edgelist(path, nodetype=int)
    return G

# -------------------- Run Main --------------------
if __name__ == "__main__":
    G = load_graph_from_edgelist("graphs/random/64random_diameter38test.edgelist")
    hierarchy = build_sparse_cover(G)
    df = measure_stretch(G, hierarchy)
    df.to_csv("spiral_results.csv", index=False)
    plot_stretch(df)
