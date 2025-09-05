# good version till now

import networkx as nx
import math, os, random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# ---------------------------- CONFIGURATION ----------------------------
PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES = [999999999999999999999999999999999999, 10, 5, 3.33333333333333333333333333, 2.5, 2]
ERROR_VALUES_2       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_TRIALS           = 10
DEBUG                = True
down_link = {}

# ---------------------------- SUBMESH HIERARCHY ----------------------------

def xy_to_id(x,y,size): 
    return x*size + y


def generate_type1_submeshes(size):
    levels = int(math.log2(size)) + 1
    hierarchy = defaultdict(list)
    for level in range(levels):
        b = 2**level
        for i in range(0, size, b):
            for j in range(0, size, b):
                nodes = {
                    # (x,y)
                    xy_to_id(x,y,size)
                    for x in range(i, min(i+b, size))
                    for y in range(j, min(j+b, size))
                }
                hierarchy[(level,2)].append(nodes)
    return hierarchy

def generate_type2_submeshes(size):
    levels = int(math.log2(size))
    hierarchy = defaultdict(list)
    for level in range(1, levels):
        b = 2**level
        off = b//2
        for i in range(-off, size, b):
            for j in range(-off, size, b):
                nodes = {
                    # (x,y)
                    xy_to_id(x,y,size)
                    for x in range(i, i+b)
                    for y in range(j, j+b)
                    if 0 <= x < size and 0 <= y < size
                }
                if nodes:
                    hierarchy[(level,1)].append(nodes)
    return hierarchy

def build_mesh_hierarchy(size):
    """
    Combines Type-1 and Type-2 submeshes to form a full hierarchy.
    """
    H = generate_type1_submeshes(size)
    H.update(generate_type2_submeshes(size))

    # make level 0 match
    H[(0,1)] = list(H[(0,2)])

    # sanity check bottom & interior
    levels = sorted({lvl for (lvl,_) in H})
    lo, hi = levels[0], levels[-1]
    if H[(lo,1)] != H[(lo,2)]:
        raise RuntimeError("Level 0 must have identical Type-1/2 clusters")
    for lvl in levels[1:-1]:
        if H[(lvl,1)] == H[(lvl,2)]:
            raise RuntimeError(f"Level {lvl} Type-1 == Type-2; they must differ")

    # now add the root cluster *to both* so they compare equal
    # all_nodes = {(x, y) for x in range(size) for y in range(size)}
    # being consistent with xy to id everyehere (even in root)
    all_nodes = {xy_to_id(x, y, size) for x in range(size) for y in range(size)}
    root_level = hi + 1
    H[(root_level,1)].append(all_nodes)
    H[(root_level,2)].append(all_nodes)
    if H[(root_level,1)] != H[(root_level,2)]:
        raise RuntimeError("Root level must have identical Type-1/2 clusters")

    return H

def print_clusters(H):
    print("=== Cluster Hierarchy ===")
    levels = sorted(set(lvl for (lvl, _) in H))
    for lvl in levels:
        for t in [1, 2]:
            key = (lvl, t)
            if key in H:
                print(f"Level {lvl} Type-{t}: {len(H[key])} clusters")
                for idx, cl in enumerate(H[key]):
                    print(f"  Cluster {idx}: {sorted(cl)}")
    print("="*40)




# def assign_cluster_leaders(H):
#     M = defaultdict(list)
#     for lvl_type, clusters in H.items():
#         for cl in clusters:
#             # pick e.g. the minimum node as our deterministic leader
#             leader = min(cl)
#             M[lvl_type].append((leader,cl))
#     return M

def assign_cluster_leaders(H, seed=42):
    rng = random.Random(seed)
    M = defaultdict(list)
    for lvl_type, clusters in H.items():
        for cl in clusters:
            leader = rng.choice(tuple(cl))  # uniform within cluster
            M[lvl_type].append((leader, cl))
    return M


# ---------------------------- PUBLISH / DOWNWARD LINKS ----------------------------

def get_spiral(node, leader_map, verbose=False):
    path = [node]
    seen = {node}
    for lvl_type in sorted(leader_map):
        for leader, cl in leader_map[lvl_type]:
            if node in cl and leader not in seen:
                path.append(leader)
                seen.add(leader)
                break
    if verbose:
        print(f"Spiral upward path for {node}: {path}")
    return path

def publish(owner, leader_map):
    """
    Store downward pointers at every cluster leader the owner belongs to.
    """
    global down_link
    down_link.clear()
    sp = get_spiral(owner, leader_map) # spiral is bottom-up
    # Store a pointer at every leader in the spiral (except the last, which is the owner)
    for i in range(len(sp)-1, 0, -1):
        down_link[sp[i]] = sp[i-1]

# ---------------------------- STRETCH MEASUREMENT ----------------------------

def measure_stretch(requesters, owner, leader_map, G):
    global down_link
    total_up_down = 0
    total_opt     = 0

    for r in requesters:
        if r == owner:
            continue

        # 1) climb spiral
        sp = get_spiral(r, leader_map)
        print(f"\nRequester {r}:")
        print(f"  Upward path: {sp}")

        up_hops = 0
        intersection = None
        for i in range(len(sp)-1):
            up_hops += nx.shortest_path_length(G, sp[i], sp[i+1])
            if sp[i+1] in down_link:
                intersection = sp[i+1]
                break
        if intersection is None:
            intersection = sp[-1]
        print(f"  Intersection at: {intersection}")

        # 2) descend via down_link chain (with fallback)
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

        dist_sp  = up_hops + down_hops
        dist_opt = nx.shortest_path_length(G, r, owner)

        print(f"  Up hops: {up_hops}")
        print(f"  Down hops: {down_hops}")
        print(f"  Total hops: {dist_sp}")
        print(f"  Shortest path: {dist_opt}")
        print(f"  Stretch: {dist_sp/dist_opt:.3f}")

        total_up_down += dist_sp
        total_opt     += dist_opt

        owner = r
        publish(owner, leader_map)

    return (total_up_down/total_opt) if total_opt>0 else 1.0



# ---------------------------- GRAPH LOADING ----------------------------

def load_graph(dfile):
    G = nx.read_graphml(os.path.join("graphs","grid",dfile))
    return nx.relabel_nodes(G, lambda x:int(x))

# ---------------------------- ERRORS & HELPERS ----------------------------

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

        # 3) check if within tolerance
        if current_overlap <= 100:
            return Q

    random.shuffle(Q)  # Shuffle the list to ensure randomness
    return Q

def sample_actual(G, Vp, error):
    diam = nx.diameter(G)
    act = []
    for v in Vp:
        cutoff = int(diam/error) if error>0 else diam
        lengths = nx.single_source_shortest_path_length(G, v, cutoff=cutoff)
        act.append(random.choice(list(lengths.keys())))
    return act

def calculate_error(Vp, Q, G_example):
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

# ---------------------------- SIMULATION ----------------------------

def simulate(graph_file):
    G = load_graph(graph_file)
    size = int(math.sqrt(G.number_of_nodes()))
    H = build_mesh_hierarchy(size)
    print_clusters(H)
    leaders = assign_cluster_leaders(H)

    results = []
    for error in ERROR_VALUES:
      for frac in PREDICTION_FRACTIONS:
        # ←– ADD THESE TWO LINES HERE
        owner = random.choice(list(G.nodes()))
        publish(owner, leaders)

        for _ in range(NUM_TRIALS):
          pred = choose_Vp(G, frac)
          act  = sample_Q_within_diameter(G, pred, error)
          err  = calculate_error(pred, act, G)

          for req in act:
            if req == owner: 
              continue
            stretch = measure_stretch([req], owner, leaders, G)

            if error > 15:
              err_rate = 0.0
            else:
              err_rate = round(1.0 / error, 1)

            results.append((frac, err_rate, err, stretch))

            owner = req
            publish(owner, leaders)

    return results


# ---------------------------- PLOTTING ----------------------------

def plot_results(results):
    df  = pd.DataFrame(results, columns=["Frac","ErrRate","Err","Str"])
    avg = df.groupby(["Frac","ErrRate"]).mean().reset_index()

    # use your global list here
    xvals = PREDICTION_FRACTIONS

    plt.figure(figsize=(12,6))

    # ---------------- Error vs Fraction ----------------
    plt.subplot(1,2,1)
    for e in ERROR_VALUES_2:
        sub = avg[ avg.ErrRate == e ]
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
    # for e in ERROR_VALUES_2:
    #     sub = avg[ avg.ErrRate == e ]
    #     plt.plot(sub.Frac, sub.Str, '-o', label=f"{e:.1f} Stretch")

    # loop over each unique ErrRate in your aggregated frame

    for err_rate, group in avg.groupby("ErrRate"):
        plt.plot(
            group.Frac, 
            group.Str, 
            "-o", 
            label=f"{err_rate:.1f} Stretch"
        )

    


    plt.title("Stretch vs Fraction of Predicted Nodes")
    plt.xlabel("Fraction of Predicted Nodes")
    plt.ylabel("Stretch")
    plt.xticks(xvals, [f"{f:.4f}" for f in xvals], rotation=45)
    # plt.ylim(0.95, 1.05)
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# ---------------------------- MAIN ----------------------------

if __name__ == "__main__":
    # res = simulate("64grid_diameter14test.edgelist")
    # res = simulate("144grid_diameter22test.edgelist")
    # res = simulate("256grid_diameter30test.edgelist")
    # res = simulate("576grid_diameter46test.edgelist")
    res = simulate("1024grid_diameter62test.edgelist")
    print("I collected", len(res), "data points")
    df  = pd.DataFrame(res, columns=["Frac","ErrRate","Err","Str"])
    print(df.head())
    plot_results(res)
