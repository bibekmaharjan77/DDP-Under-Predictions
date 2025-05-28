# good version of run110.py

# multibend_protocol.py
# Updated to ensure stretch is measured correctly through full spiral paths
# Implements MultiBend paper logic: cluster construction (Type-1, Type-2, and root), spiral paths, and routing evaluation

import networkx as nx                    # NetworkX handles graph data structures and algorithms
import math                              # Math provides logarithmic and square root functions
import os                                # OS for file path manipulations
import random                            # Random sampling for simulation experiments
import matplotlib.pyplot as plt          # Matplotlib for visualizing error and stretch plots
from collections import Counter, defaultdict      # Defaultdict for grouping clusters and leader maps
import pandas as pd                      # Pandas for aggregating and analyzing simulation results

# ---------------------------- CONFIGURATION ----------------------------

# Fractions of nodes used as predicted requesters
PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]

# Error rates (inaccuracy of predictions): 0.0 = perfect prediction, 0.5 = total mismatch

ERROR_VALUES = [999999999999999999999999999999999999, 10, 5, 3.33333333333333333333333333, 2.5, 2]

ERROR_VALUES_2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# Number of trials to average for each error rate and prediction fraction
NUM_TRIALS = 2

# Enable or disable detailed debug output
DEBUG = True

# ---------------------------- SUBMESH HIERARCHY ----------------------------

def generate_type1_submeshes(size):
    """
    Creates Type-1 submeshes (aligned blocks) for each level.
    At level l, blocks are of size 2^l x 2^l.
    Each block represents a cluster at that level.
    """

    # Calculate the number of levels based on grid size.
    # For a grid of size N×N, levels go from 0 up to log2(N), inclusive.
    levels = int(math.log2(size)) + 1

    # Initialize a dictionary to store clusters per level.
    # The key is a tuple (level, 2), where '2' denotes Type-1 submeshes.
    hierarchy = defaultdict(list)

    # Iterate over each level of the hierarchy
    for level in range(levels):
        block_size = 2 ** level     # At each level, determine the block size (width and height of submeshes).
        
        # Slide a block of 'block_size' over the grid in both row and column directions.
        for i in range(0, size, block_size):    # Start of the block in row
            for j in range(0, size, block_size):    # Start of the block in column

                # For the current block (i,j), gather all (x,y) coordinates within the block.
                # Ensure we do not exceed the grid boundary using min(..., size)
                nodes = set((x, y) for x in range(i, min(i+block_size, size))
                                     for y in range(j, min(j+block_size, size)))
                
                # Add this block (cluster) to the hierarchy under the appropriate level
                hierarchy[(level, 2)].append(nodes)
    
    # Return the full hierarchy of all Type-1 submeshes
    return hierarchy

def generate_type2_submeshes(size):
    """
    Creates Type-2 submeshes (offset blocks) for each level > 0.
    These are shifted by half the block size (both horizontally and vertically) to create overlap with Type-1 submeshes and cover gaps.
    """
    # Number of valid levels for Type-2 is from level 1 up to log2(size) - 1
    # (we don't define Type-2 submeshes at level 0)
    levels = int(math.log2(size))

    # Initialize a dictionary to collect clusters at each level
    hierarchy = defaultdict(list)

    # Loop through each level starting from 1
    for level in range(1, levels):
        block_size = 2 ** level  # Block size increases exponentially with level
        offset = block_size // 2  # Offset for Type-2 submeshes: half a block

        # Shifted window iteration: move in steps of 'block_size',
        # but start at -offset to introduce the shift
        for i in range(-offset, size, block_size):  # rows
            for j in range(-offset, size, block_size):  # columns

                # Construct nodes in the block starting at (i, j)
                # but only include valid coordinates (inside grid)
                nodes = set((x, y) for x in range(i, i + block_size)
                                     for y in range(j, j + block_size)
                            if 0 <= x < size and 0 <= y < size)        # discard out-of-bound nodes
                
                # Only add non-empty clusters (some blocks at edges may be invalid)
                if nodes:
                    hierarchy[(level, 1)].append(nodes)  # (1) denotes Type-2

    # Return dictionary containing Type-2 clusters at each level
    return hierarchy

def build_mesh_hierarchy(size):
    """
    Combines Type-1 and Type-2 submeshes to form a full hierarchy (complete mesh hierarchy).
    Adds a top-level root cluster containing all nodes.
    This ensures any two nodes always share a common ancestor cluster.
    """

    H = generate_type1_submeshes(size)  # Generate Type-1 clusters: aligned blocks (non-overlapping)

    # Add Type-2 clusters: offset blocks (overlapping)
    # This call updates the same hierarchy dictionary
    H.update(generate_type2_submeshes(size))

    # Create a special cluster at the top level (root)
    # This cluster includes every coordinate (x, y) in the grid
    all_nodes = set((x, y) for x in range(size) for y in range(size))

    # Determine the highest level index, just above all previous levels
    # It is log2(size) + 1 to make sure it’s strictly above all regular levels
    max_level = int(math.log2(size)) + 1

    # Append the root-level cluster into the hierarchy under a new key
    # (max_level, 2) — the '2' here is arbitrary; it's just consistent with Type-1 notation
    H[(max_level, 2)].append(all_nodes)

    # Return the complete hierarchy dictionary:
    # keys → (level, type), values → list of node sets (clusters)
    return H


def assign_cluster_leaders(hierarchy):
    """
    Assigns a leader node (lowest ID) for each cluster at every level.
    Returns a dictionary (leader_map) that maps:
        (level, type) → list of (leader, cluster) pairs.
    This is used to trace spiral paths through the hierarchy.
    """

    # Initialize a dictionary to store leaders for each level/type.
    # Each key is a (level, type) tuple (e.g., (2, 1)), and value is a list of (leader, cluster) pairs.
    leader_map = defaultdict(list)

    # Iterate over each (level, type) and its associated clusters in the hierarchy
    for level_type, clusters in hierarchy.items():

        # For each individual cluster at this level
        for cluster in clusters:

            # Select a leader node within this cluster.
            # Here, leader is simply the node with the minimum ID (using Python's default ordering).
            leader = min(cluster)

            # Store the (leader, cluster) pair in the leader map
            leader_map[level_type].append((leader, cluster))

    # Return the complete map of leaders at each level        
    return leader_map

# ---------------------------- PATH EXTRACTION ----------------------------

def get_spiral_path(node, leader_map):
    """
    Constructs the spiral path for a node by walking up through its cluster leaders.
    One leader per level is added to the path, including the root.
    - The path starts from the node and moves upward to the root-level leader.
    - The result is a list of nodes representing the upward path (i.e., the spiral).
    """

    # Initialize the path starting from the node itself
    path = [node]

    # Maintain a set of visited nodes to prevent duplicate entries
    visited = set(path)

    # Traverse the cluster hierarchy level by level (sorted by level number)
    for level_type in sorted(leader_map):
        for leader, cluster in leader_map[level_type]:   # For each cluster at this level
            if node in cluster:   # If the current node is part of this cluster
                if leader not in visited:   # Add the cluster's leader to the path if not already visited
                    path.append(leader)
                    visited.add(leader)
                break  # Stop at first match in that level i.e., Only one cluster per level should contribute a leader, so we break

    # Ensure the top-level leader is included i.e., Adds the root-level cluster leader explicitly to guarantee a common ancestor
    max_level = max(leader_map.keys(), key=lambda x: x[0])   # Find the highest level key
    for leader, cluster in leader_map[max_level]:
        if node in cluster and leader not in visited:
            path.append(leader)
            break   # Only one leader should be added from the root

    return path  # Return the completed spiral path

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

    # Compute the diameter of the graph — the longest shortest path between any two nodes.
    # Used to normalize distances and scale the error to [0, 1].
    diameter = nx.diameter(G)

    # Initialize a variable to accumulate total normalized distance
    total_error = 0

    # Iterate over paired predicted and actual nodes
    for u, v in zip(pred_nodes, actual_nodes):
        d = nx.shortest_path_length(G, source=u, target=v)  # Compute the shortest path distance between u and v in the graph
        total_error += d / diameter  # Normalize this distance by the diameter and add to total error
    return total_error / len(pred_nodes)  # Return the average normalized error across all pairs



def calculate_error_updated(Vp, Q, G_example):
    diameter_of_G = nx.diameter(G_example, weight='weight')  # Compute the diameter of the graph G_example
    errors = []
    for req, pred in zip(Q, Vp):
        # Using NetworkX to compute the shortest path length in tree T.
        dist = nx.shortest_path_length(G_example, source=req, target=pred, weight='weight')
        error = dist / diameter_of_G
        errors.append(error)
        # print(f"\nDistance between request node {req} and predicted node {pred} is {dist}, error = {error:.4f}")
    
    # print("Diameter of G:", diameter_of_G)
    # print("Diameter of T:", diameter_of_T)
    total_max_error = max(errors) if errors else 0
    total_min_error = min(errors) if errors else 0
    RED = "\033[91m"
    RESET = "\033[0m"
    print(f"{RED}\nOverall max error (max_i(distance_in_G / diameter_G)) = {total_max_error:.4f}{RESET}")
    print(f"{RED}\nOverall min error (min_i(distance_in_G / diameter_G)) = {total_min_error:.4f}{RESET}")
    return total_max_error

def calculate_stretch_spiral(pred_nodes, actual_nodes, leader_map, G):
    """
    Calculates stretch for each requester pair using spiral paths
    (paths up through cluster leaders to their lowest common ancestor).
    Stretch = spiral distance / shortest distance in G.
    """
    # Initialize accumulators for spiral and true shortest distances
    sum_spiral = 0
    sum_shortest = 0
    count = 0

    # Process each pair (predicted node, actual requester)
    for u, v in zip(pred_nodes, actual_nodes):
        if u == v:   # Skip self-pairs (no movement/stretch needed)
            if DEBUG:
                print(f"[WARN] Skipping identical node pair (u={u}, v={v})")
            continue

        # Construct upward spiral paths from both nodes via their cluster leaders
        path_u = get_spiral_path(u, leader_map)
        path_v = get_spiral_path(v, leader_map)

        # Find common nodes (leaders) in the two spiral paths
        common = set(path_u) & set(path_v)
        if not common:
            if DEBUG:
                print(f"[WARN] No common leader in spiral paths for (u={u}, v={v})")
            continue

        # Find the Lowest Common Ancestor (LCA) in the spiral hierarchy
        lca = next(n for n in reversed(path_u) if n in common)

        if lca is None:
            if DEBUG:
                print(f"[WARN] No LCA found for pair (u={u}, v={v})")
            continue


        try:
            # Index of the LCA in both spiral paths
            idx_u = path_u.index(lca)
            idx_v = path_v.index(lca)

            # Spiral path = path from u up to LCA + reversed path from v up to LCA
            spiral_path = path_u[:idx_u+1] + list(reversed(path_v[:idx_v]))
        except ValueError:
            if DEBUG:
                print(f"[ERROR] LCA index not found for u={u}, v={v}, lca={lca}")
            continue

        # If the path is too short to measure meaningful distance (e.g., length 2), skip
        if len(spiral_path) <= 2:
            if DEBUG:
                print(f"[WARN] Spiral path is trivial (length 2) for (u={u}, v={v}), skipping.")
            continue
        
        try:
            # Compute the total length of the spiral path by summing distances between adjacent nodes
            dist_spiral = sum(nx.shortest_path_length(G, spiral_path[i], spiral_path[i+1])
                              for i in range(len(spiral_path)-1))
            dist_shortest = nx.shortest_path_length(G, source=u, target=v)   # Compute the direct shortest path between u and v in the graph
        except nx.NetworkXNoPath:
            if DEBUG:
                print(f"[ERROR] No path in G between nodes in spiral path or from u={u} to v={v}")
            continue

        # Skip invalid cases where shortest path is 0 (shouldn’t happen unless u == v)
        if dist_shortest == 0:
            if DEBUG:
                print(f"[WARN] Shortest path from {u} to {v} is zero, skipping.")
            continue

        # Calculate stretch for this pair
        stretch_value = dist_spiral / dist_shortest

        # Optional debugging logs
        if DEBUG:
            print(f"\n[DEBUG] u={u}, v={v}")
            print(f"Spiral path: {spiral_path}")
            print(f"Spiral path length: {len(spiral_path)}")
            print(f"Spiral path distance: {dist_spiral}")
            print(f"Shortest path distance: {dist_shortest}")
            print(f"Stretch: {stretch_value:.3f}")

        # Accumulate distances for average computation
        sum_spiral += dist_spiral
        sum_shortest += dist_shortest
        count += 1

    # Final debug summary
    if DEBUG:
        print(f"\n[INFO] Total valid pairs used: {count}")
        print(f"[INFO] Total spiral distance: {sum_spiral}")
        print(f"[INFO] Total shortest distance: {sum_shortest}")

    # Return average stretch across all valid pairs 
    return sum_spiral / sum_shortest if sum_shortest > 0 else 1.0


def choose_Vp(G, fraction):
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    random.shuffle(nodes)  # Shuffle the nodes to ensure randomness
    total_nodes = len(nodes)
    vp_size = int(total_nodes * fraction) # Fraction of nodes to be chosen as Vp
    original_Vp = list(random.choices(nodes, k=vp_size))
    random.shuffle(original_Vp)  # Shuffle Vp to ensure randomness

    reduced_Vp = set(original_Vp)

    reduced_Vp = list(reduced_Vp)  # Convert back to a list for indexing
    random.shuffle(reduced_Vp)  # Shuffle Vp to ensure randomness

    # Choose an owner node that is not in Vp
    remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(remaining))

    # Insert owner to reduced_Vp list at a random position
    insert_position = random.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    S = reduced_Vp.copy()
    S = set(S)  # Convert to a set for uniqueness

    return original_Vp


def count_duplicates(input_list):
    """
    Checks for duplicate elements in a list and returns their counts.

    Args:
        input_list: The list to check for duplicates.

    Returns:
        A dictionary where keys are the duplicate elements and values are their counts.
        Returns an empty dictionary if no duplicates are found.
    """
    counts = Counter(input_list)
    duplicates = {element: count for element, count in counts.items() if count > 1}
    return duplicates



def sample_Q_within_diameter(G, Vp, error_cutoff):
    diam = nx.diameter(G, weight='weight')
    max_iter = 100000  # Maximum number of iterations to avoid infinite loop

    for attempt in range(1, max_iter+1):
        # 1) sample one random reachable node per v
        Q = []
        for v in Vp:
            dist_map = nx.single_source_dijkstra_path_length(G, v, cutoff=float(diam/error_cutoff), weight="weight")
            Q.append(random.choice(list(dist_map.keys())))

        # 2) compute overlap
        dup_counts = count_duplicates(Q)
        # extra dups = sum of (count - 1) for each duplicated element
        extra_dups = sum(cnt for cnt in dup_counts.values())
        current_overlap = extra_dups / len(Q) * 100

        # if dup_counts:
        #     print("Duplicate elements in Q and their counts:")
        #     for element, count in dup_counts.items():
        #         print(f"{element}: {count}")
        # else:
        #     print("No duplicate elements found.")

        # print("Extra dups: ", extra_dups)
        # print("Current overlap: ", current_overlap)

        # 3) check if within tolerance
        if current_overlap <= 100:
            return Q

    # print(f"Could not reach {overlap}% overlap after {max_iter} tries.")
    # print(f"Last overlap was {current_overlap:.2f}%, duplicates:", dup_counts)
    random.shuffle(Q)  # Shuffle the list to ensure randomness
    return Q


# ---------------------------- SIMULATION ----------------------------

def simulate_multibend_eval(graph_file):
    """
    Main experiment loop.
    For each error rate and prediction fraction:
      - Generates predicted and actual requester node lists
      - Computes error and spiral stretch
    """
    G = load_graph_from_directory(graph_file)  # Load the graph from file and relabel nodes as integers
    size = int(math.sqrt(len(G)))  # Infer the grid size (since the graph is a square grid with size x size nodes)
    hierarchy = build_mesh_hierarchy(size)  # Build the full hierarchical clustering (Type-1, Type-2, and top-level root)
    leader_map = assign_cluster_leaders(hierarchy)  # Assign a leader to each cluster (lowest node ID)
    node_list = list(G.nodes())  # Get a flat list of all nodes in the graph
    results = []  # Initialize a list to store results from each experiment run

    # Outer loop: iterate over different simulated prediction error rates (0.0 to 0.5)
    for error_rate in ERROR_VALUES:

        # For each error rate, iterate over different prediction fractions
        for frac in PREDICTION_FRACTIONS:
            pred_nodes = choose_Vp(G, frac)  # Choose a set of predicted nodes (Vp) based on the fraction
            actual_nodes = sample_Q_within_diameter(G, pred_nodes, error_rate)
            # num_nodes = int(frac * len(G))   # Calculate the number of nodes to sample based on the fraction

            err = calculate_error_updated(pred_nodes, actual_nodes, G)

            # Compute stretch using spiral path routing
            stretch = calculate_stretch_spiral(pred_nodes, actual_nodes, leader_map, G)

            # Record the results of this run
            if error_rate > 15:
                results.append((frac, 0, err, stretch))
            else:
                results.append((frac, 1/error_rate, err, stretch))
           
    return results   # Return the complete list of results for plotting and analysis

# ---------------------------- PLOTTING ----------------------------

def plot_results(results):
    """
    Aggregates and plots average error and stretch metrics.
    Produces two subplots:
      - Error vs. Prediction Fraction (one curve per error rate)
      - Stretch vs. Prediction Fraction (one curve per error rate)
    """
    
    df = pd.DataFrame(results, columns=["Fraction", "ErrorRate", "Error", "Stretch"])
    filename = 'data.xlsx'
    df.to_excel(filename)
    avg_df = df.groupby(["Fraction", "ErrorRate"]).mean().reset_index()

    plt.figure(figsize=(12, 6))

    # Error Plot: how far off predicted requesters are from actual ones
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

    # Stretch Plot: how efficient spiral paths are compared to shortest paths
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
    """
    Entry point for executing the full evaluation pipeline:
      - Loads graph
      - Runs simulation
      - Plots results
    """
    results = simulate_multibend_eval("256grid_diameter30test.edgelist")  # Adjust to your file name
    plot_results(results)

if __name__ == "__main__":
    main()