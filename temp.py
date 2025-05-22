import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict

# ---- Parameters ----
PREDICTION_FRACTIONS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
ERROR_VALUES = [1, 2, 4, 8, 12, 16]  # Use the same for both simulation and plotting
ERROR_VALUES_2 = [1, 2, 4, 8, 12, 16]
NUM_TRIALS = 3

# ---- Helper Functions ----

def load_graph(graph_file):
    # Load as edgelist or graphml
    if graph_file.endswith('.edgelist'):
        G = nx.read_edgelist(graph_file, nodetype=int)
    else:
        G = nx.read_graphml(graph_file)
        G = nx.convert_node_labels_to_integers(G)
    return G

def build_mesh_hierarchy(size):
    # Build clusters for a 2D mesh of given size (size x size)
    H = {}
    max_level = int(math.log2(size))
    for lvl in range(max_level+1):
        for t in [1, 2]:
            clusters = []
            step = 2 ** lvl
            for i in range(0, size, step):
                for j in range(0, size, step):
                    nodes = []
                    for x in range(i, min(i+step, size)):
                        for y in range(j, min(j+step, size)):
                            if t == 1 or ((x-i)+(y-j)) % 2 == 0:
                                nodes.append(x*size + y)
                    if nodes:
                        clusters.append(set(nodes))
            H[(lvl, t)] = clusters
    return H

def print_clusters(H):
    print("=== Cluster Hierarchy ===")
    levels = sorted({lvl for (lvl, _) in H})
    for lvl in levels:
        for t in [1, 2]:
            key = (lvl, t)
            if key in H:
                print(f"Level {lvl} Type-{t}: {len(H[key])} clusters")
                for idx, cl in enumerate(H[key]):
                    print(f"  Cluster {idx}: {sorted(cl)}")
    print("="*40)

def assign_cluster_leaders(H):
    # Pick the lowest-numbered node as leader for each cluster
    leader_map = {}
    for key, clusters in H.items():
        leader_map[key] = []
        for cl in clusters:
            leader = min(cl)
            leader_map[key].append((leader, cl))
    return leader_map

def get_spiral(node, leader_map):
    path = [node]
    seen = {node}
    for lvl_type in sorted(leader_map):
        for leader, cl in leader_map[lvl_type]:
            if node in cl and leader not in seen:
                path.append(leader)
                seen.add(leader)
                break
    return path

# ---- Directory Logic ----
down_link = {}

def publish(owner, leader_map):
    global down_link
    down_link.clear()
    sp = get_spiral(owner, leader_map)
    for i in range(len(sp)-1, 0, -1):
        down_link[sp[i]] = sp[i-1]

def measure_stretch(requesters, owner, leader_map, G):
    global down_link
    stretches = []
    for r in requesters:
        if r == owner:
            continue

        sp = get_spiral(r, leader_map)
        print(f"\nRequester {r}:")
        print(f"  Upward path: {sp}")

        # Find intersection
        up_hops = 0
        intersection = None
        for i in range(len(sp)-1):
            up_hops += nx.shortest_path_length(G, sp[i], sp[i+1])
            if sp[i+1] in down_link:
                intersection = sp[i+1]
                break
        if intersection is None:
            intersection = sp[-1]
        print(f"  Intersection at leader: {intersection}")

        # Downward path
        down_hops = 0
        cur = intersection
        seen = {cur}
        down_path = []
        while cur != owner:
            nxt = down_link.get(cur)
            if nxt is None or nxt in seen:
                direct = nx.shortest_path_length(G, cur, owner)
                down_path.append(f"Direct jump to {owner} (+{direct} hops)")
                down_hops += direct
                break
            down_path.append(nxt)
            down_hops += nx.shortest_path_length(G, cur, nxt)
            seen.add(nxt)
            cur = nxt
        print(f"  Downward path: {down_path}")

        dist_sp = up_hops + down_hops
        dist_opt = nx.shortest_path_length(G, r, owner)
        print(f"  Up hops: {up_hops}")
        print(f"  Down hops: {down_hops}")
        print(f"  Total hops: {dist_sp}")
        print(f"  Shortest path: {dist_opt}")
        print(f"  Stretch: {dist_sp/dist_opt:.3f}")

        stretches.append(dist_sp / dist_opt if dist_opt > 0 else 1.0)
    return np.mean(stretches) if stretches else 1.0

# ---- Prediction and Error Simulation ----
def choose_Vp(G, frac):
    # Randomly select a fraction of nodes as predicted set
    n = G.number_of_nodes()
    k = max(1, int(frac * n))
    return set(random.sample(list(G.nodes()), k))

def sample_Q_within_diameter(G, pred, error):
    # For simplicity, pick a random subset of 'error' nodes from the graph
    candidates = list(set(G.nodes()) - pred)
    k = min(error, len(candidates))
    return random.sample(candidates, k) if k > 0 else []

def calculate_error(pred, act, G):
    # Max shortest path distance from pred to act
    if not act or not pred:
        return 0
    return max(nx.shortest_path_length(G, source=p, target=a) for p in pred for a in act)

# ---- Simulation ----
def simulate(graph_file):
    G = load_graph(graph_file)
    size = int(math.sqrt(G.number_of_nodes()))
    H = build_mesh_hierarchy(size)
    print_clusters(H)
    leaders = assign_cluster_leaders(H)

    results = []
    owner = random.choice(list(G.nodes()))
    publish(owner, leaders)

    for error in ERROR_VALUES:
        for frac in PREDICTION_FRACTIONS:
            for _ in range(NUM_TRIALS):
                pred = choose_Vp(G, frac)
                act = sample_Q_within_diameter(G, pred, error)
                err = calculate_error(pred, act, G)
                for req in act:
                    if req == owner:
                        continue
                    stretch = measure_stretch([req], owner, leaders, G)
                    results.append((frac, error, err, stretch))
                    owner = req
                    publish(owner, leaders)
    return results

# ---- Plotting ----
def plot_results(results):
    df  = pd.DataFrame(results, columns=["Frac","Error","Err","Str"])
    avg = df.groupby(["Frac","Error"]).mean().reset_index()

    xvals = PREDICTION_FRACTIONS

    plt.figure(figsize=(12,6))

    # ---------------- Error vs Fraction ----------------
    plt.subplot(1,2,1)
    for e in ERROR_VALUES_2:
        sub = avg[ avg.Error == e ]
        plt.plot(sub.Frac, sub.Err, '-o', label=f"{e:.1f} Error")
    plt.title("Error vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Error (Max)")
    plt.xticks(xvals, [f"{f:.4f}" for f in xvals], rotation=45)
    plt.ylim(0, max(ERROR_VALUES_2)*1.1)
    plt.grid(True)
    plt.legend(loc="upper right")

    # ---------------- Stretch vs Fraction ----------------
    plt.subplot(1,2,2)
    for e in ERROR_VALUES_2:
        sub = avg[ avg.Error == e ]
        plt.plot(sub.Frac, sub.Str, '-o', label=f"{e:.1f} Stretch")
    plt.title("Stretch vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Stretch")
    plt.xticks(xvals, [f"{f:.4f}" for f in xvals], rotation=45)
    plt.ylim(0.95, 1.05)
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

# ---- Main ----
if __name__ == "__main__":
    res = simulate("64grid_diameter14test.edgelist")
    plot_results(res)
