# spiral_random_adapted.py
import networkx as nx
import os, random, math
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# -------------------- Parameters --------------------
NUM_TRIALS = 50
PREDICTION_FRACTIONS = [0.0312, 0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# -------------------- Helper Functions --------------------
def choose_Vp(G, fraction):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    k = max(1, int(len(nodes) * fraction))
    Vp = nodes[:k]
    
    # Choose owner not in Vp
    remaining = list(set(G.nodes()) - set(Vp))
    owner = random.choice(remaining)
    
    # Insert owner at random position in Vp
    insert_pos = random.randint(0, len(Vp))
    Vp.insert(insert_pos, owner)
    
    return Vp, owner

def sample_Q_within_diameter(G, Vp, error_fraction):
    diam = nx.diameter(G)
    Q = []
    for v in Vp:
        cutoff = diam * (1 - error_fraction)
        try:
            reach = nx.single_source_shortest_path_length(G, v, cutoff=cutoff)
            Q.append(random.choice(list(reach.keys())))
        except:
            Q.append(v)  # Fallback to predicted node
    return Q

# -------------------- Spiral Cluster Construction --------------------
def build_sparse_cover(G, beta=2):
    if not nx.is_connected(G):
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
    cluster_leaders = defaultdict(dict)
    for level, clusters in enumerate(cluster_hierarchy):
        for leader, cluster in clusters:
            for node in cluster:
                cluster_leaders[level][node] = leader
    return cluster_leaders

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

# -------------------- Error Calculation --------------------
# def calculate_error(Vp, Q, G):
#     diam = nx.diameter(G)
#     errors = []
#     for pred, actual in zip(Vp, Q):
#         try:
#             dist = nx.shortest_path_length(G, pred, actual)
#             errors.append(dist / diam)
#         except:
#             errors.append(1.0)  # Max error if no path
#     return max(errors) if errors else 1.0

# -------------------- Simulation --------------------
def simulate(G, cluster_leaders):
    results = []

    for error in ERROR_VALUES:
        for frac in PREDICTION_FRACTIONS:
            for _ in range(NUM_TRIALS):
                Vp, owner = choose_Vp(G, frac)
                Q = sample_Q_within_diameter(G, Vp, error)
                
                # Calculate binary classification error rate
                correct = sum(1 for q in Q if q in Vp)
                error_rate = 1 - (correct / len(Q)) if Q else 1.0
                
                # Publish initial owner
                publish(owner, cluster_leaders)
                
                # Process each requester in chain
                for req in Q:
                    if req == owner:
                        continue
                        
                    stretch = lookup(req, owner, cluster_leaders, G)
                    results.append((frac, error, error_rate, stretch))
                    
                    # Update owner and republish
                    owner = req
                    publish(owner, cluster_leaders)

    return results


# -------------------- Plotting --------------------
def plot_results(results):
    df = pd.DataFrame(results, columns=["frac", "error", "error_rate", "stretch"])
    
    # Aggregate results
    error_agg = df.groupby(['frac', 'error'])['error_rate'].max().reset_index()
    stretch_agg = df.groupby(['frac', 'error'])['stretch'].mean().reset_index()
    
    plt.figure(figsize=(14, 6))
    
    # Error vs Fraction plot
    plt.subplot(1, 2, 1)
    for err_val in ERROR_VALUES:
        subset = error_agg[error_agg['error'] == err_val]
        plt.plot(subset['frac'], subset['error_rate'], 'o-', label=f'ε={err_val}')
    
    plt.title('Error vs Fraction of Predicted Nodes')
    plt.xlabel('Fraction of Predicted Nodes')
    plt.ylabel('Error (Max)')
    plt.xticks(PREDICTION_FRACTIONS)
    plt.grid(True)
    plt.legend()
    
    # Stretch vs Fraction plot
    plt.subplot(1, 2, 2)
    for err_val in ERROR_VALUES:
        subset = stretch_agg[stretch_agg['error'] == err_val]
        plt.plot(subset['frac'], subset['stretch'], 'o-', label=f'ε={err_val}')
    
    plt.title('Stretch vs Fraction of Predicted Nodes')
    plt.xlabel('Fraction of Predicted Nodes')
    plt.ylabel('Stretch')
    plt.xticks(PREDICTION_FRACTIONS)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('spiral_random_results.png')
    plt.show()


# -------------------- Graph Loader --------------------
def load_graph(filename):
    path = os.path.join("graphs", "random", filename)
    
    # Try to detect file format based on content
    try:
        # First try reading as edgelist
        G = nx.read_edgelist(path, nodetype=int, data=False)
        print(f"Loaded {filename} as edgelist")
        return G
    except Exception as e1:
        try:
            # If edgelist fails, try reading as GraphML
            G = nx.read_graphml(path)
            print(f"Loaded {filename} as GraphML")
            
            # Convert node labels to integers
            mapping = {}
            for node in G.nodes():
                try:
                    # Try converting node name to integer
                    mapping[node] = int(node)
                except ValueError:
                    # If conversion fails, use hash value
                    mapping[node] = hash(node) % (10**6)
                    
            G = nx.relabel_nodes(G, mapping)
            return G
        except Exception as e2:
            print(f"Failed to load graph: {e1}\n{e2}")
            raise


# -------------------- Main --------------------
if __name__ == "__main__":
    fname = "64random_diameter38test.edgelist"
    G = load_graph(fname)
    
    # Ensure connected graph
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Using largest connected component with {G.number_of_nodes()} nodes")
    
    clusters = build_sparse_cover(G)
    cluster_leaders = assign_cluster_leaders(clusters)
    
    results = simulate(G, cluster_leaders)
    plot_results(results)