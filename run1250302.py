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
NUM_TRIALS           = 50
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
    # iterate by level, and inside each level visit Type-1 (2) then Type-2 (1)
    levels = sorted({lvl for (lvl, _) in leader_map})
    for lvl in levels:
        for t in (2, 1):  # Type-1, then Type-2
            key = (lvl, t)
            if key not in leader_map: 
                continue
            for leader, cl in leader_map[key]:
                if node in cl:
                    if leader not in seen:
                        path.append(leader)
                        seen.add(leader)
                    break  # there is only one containing cluster per (lvl,t)

    # force-append the root leader so publish always reaches the root
    top_level = max(lvl for (lvl, _) in leader_map)
    root_key  = (top_level, 2) if (top_level, 2) in leader_map else (top_level, 1)
    root_leader = leader_map[root_key][0][0]
    if path[-1] != root_leader:
        path.append(root_leader)

    
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

def measure_stretch(requesters, owner, leader_map, G, weight=None, trace=True):
    """
    Compute average stretch over `requesters` w.r.t. current `owner`.
    - weight: pass "weight" if the graph has weighted edges; default None (unweighted).
    - trace:  print per-segment costs for UP (spiral climb) and DOWN (pointer chain).
    """
    global down_link
    total_up_down = 0
    total_opt     = 0

    for r in requesters:
        if r == owner:
            continue

        # 1) Upward climb along the spiral until we meet the published path
        sp = get_spiral(r, leader_map)
        up_segments = []
        up_hops = 0
        intersection = None

        if trace:
            print(f"\nRequester {r}:")
            print(f"  Spiral: {sp}")

        for i in range(len(sp) - 1):
            u, v = sp[i], sp[i+1]
            d = nx.shortest_path_length(G, u, v, weight=weight)
            up_segments.append((u, v, d))
            up_hops += d
            if v in down_link:
                intersection = v
                break

        if intersection is None:
            intersection = sp[-1]  # root leader

        if trace:
            print("  Upward segments:")
            for u, v, d in up_segments:
                print(f"    {u} -> {v} = {d}")
            print(f"  Intersection at: {intersection}")

        # 2) Downward: follow published pointers; if missing, jump directly
        down_segments = []
        down_hops = 0
        cur = intersection
        seen = {cur}
        while cur != owner:
            nxt = down_link.get(cur)
            if nxt is None or nxt in seen:
                d = nx.shortest_path_length(G, cur, owner, weight=weight)
                down_segments.append((cur, owner, d, "DIRECT"))
                down_hops += d
                break
            d = nx.shortest_path_length(G, cur, nxt, weight=weight)
            down_segments.append((cur, nxt, d, "POINTER"))
            down_hops += d
            seen.add(nxt)
            cur = nxt

        if trace:
            print("  Downward segments:")
            for u, v, d, kind in down_segments:
                extra = " [DIRECT]" if kind == "DIRECT" else ""
                print(f"    {u} -> {v} = {d}{extra}")

        dist_sp  = up_hops + down_hops
        dist_opt = nx.shortest_path_length(G, r, owner, weight=weight)
        stretch  = dist_sp / dist_opt if dist_opt > 0 else 1.0

        if trace:
            print(f"  Up hops: {up_hops}")
            print(f"  Down hops: {down_hops}")
            print(f"  Total hops: {dist_sp}")
            print(f"  Shortest path (opt): {dist_opt}")
            print(f"  Stretch: {stretch:.3f}")

        total_up_down += dist_sp
        total_opt     += dist_opt

        # Move ownership and republish for the next requester
        owner = r
        publish(owner, leader_map)

    return (total_up_down / total_opt) if total_opt > 0 else 1.0




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

# ---------------------------- HALVING MODE ADD-ONS ----------------------------

def halving_counts(n: int):
    """Return [n/2, n/4, ..., 1] using integer division."""
    counts = []
    k = n // 2
    while k >= 1:
        counts.append(k)
        k //= 2
    return counts

def choose_Vp_halving(G, k: int):
    """
    Halving-mode version of choose_Vp. Intentionally mirrors the original:
    - uses random.choices (with replacement)
    - dedups to build reduced_Vp
    - selects an owner not in reduced_Vp and inserts it (for parity with original)
    - returns *original_Vp* (same as your original choose_Vp)
    """
    G = nx.relabel_nodes(G, lambda x: int(x))
    nodes = list(G.nodes())
    random.shuffle(nodes)  # Shuffle for randomness
    total_nodes = len(nodes)

    # mirror the original: clamp to [1, n] and use as-is
    vp_size = int(k)
    vp_size = max(1, min(vp_size, total_nodes))

    # with replacement (exactly like your original)
    original_Vp = list(random.choices(nodes, k=vp_size))
    random.shuffle(original_Vp)

    # build reduced_Vp & owner (same as original structure)
    reduced_Vp = set(original_Vp)
    reduced_Vp = list(reduced_Vp)
    random.shuffle(reduced_Vp)

    remaining = set(nodes) - set(reduced_Vp)
    owner = random.choice(list(remaining)) if remaining else random.choice(nodes)

    insert_position = random.randint(0, len(reduced_Vp))
    reduced_Vp.insert(insert_position, owner)
    S = set(reduced_Vp)  # kept for parity; not returned/used (same as original)

    # IMPORTANT: return signature matches your original choose_Vp
    return original_Vp


def simulate_halving(graph_file):
    """
    Same outputs as simulate(): list of (Frac, ErrRate, Err, Str).
    Only difference: |P| runs over {n/2, n/4, ..., 1}, with Frac = k/n.
    """
    G = load_graph(graph_file)
    n = G.number_of_nodes()
    size = int(math.sqrt(n))
    H = build_mesh_hierarchy(size)
    print_clusters(H)
    leaders = assign_cluster_leaders(H)

    k_values = halving_counts(n)
    results = []

    for error in ERROR_VALUES:
        for k in k_values:
            frac = k / n  # so plots are still vs fraction

            # reset ownership for this bucket
            owner = random.choice(list(G.nodes()))
            publish(owner, leaders)

            for _ in range(NUM_TRIALS):
                P = choose_Vp_halving(G, k)              # <- mirrors original choose_Vp
                Q = sample_Q_within_diameter(G, P, error)  # |Q| == |P|
                err = calculate_error(P, Q, G)

                for req in Q:
                    if req == owner:
                        continue
                    stretch = measure_stretch([req], owner, leaders, G, trace=False)

                    err_rate = 0.0 if error > 15 else round(1.0 / error, 1)
                    results.append((frac, err_rate, err, stretch))

                    owner = req
                    publish(owner, leaders)

    return results



def plot_results_halving_counts(results, n=None, title_suffix=" (halving mode)"):
    """
    Same input schema as your other plotters: results = [(Frac, ErrRate, Err, Str), ...]
    Shows x-axis as |P| (k) taking values 1, 2, 4, ..., n/2.
    If n is not given, it will try to infer n as round(1 / min(Frac)).
    """
    df = pd.DataFrame(results, columns=["Frac","ErrRate","Err","Str"])

    # infer n if not provided (works for halving runs that include 1/n)
    if n is None:
        min_frac = df["Frac"].min()
        n = int(round(1.0 / min_frac)) if min_frac > 0 else None
        if not n:
            raise ValueError("Please pass n explicitly to plot_results_halving_counts().")

    # convert fractions back to counts k = |P| and group by count
    df["Count"] = (df["Frac"] * n).round().astype(int)
    avg = df.groupby(["Count","ErrRate"]).mean().reset_index()

    # x-axis ticks in ascending halving order: 1,2,4,...,n/2
    xvals = sorted(df["Count"].unique())

    plt.figure(figsize=(12,6))

    # ---------------- Error vs |P| ----------------
    plt.subplot(1,2,1)
    for e in ERROR_VALUES_2:
        sub = avg[ avg.ErrRate == e ].sort_values("Count", ascending=True)
        plt.plot(sub.Count, sub.Err, "-o", label=f"{e:.1f} Error")
    plt.title("Error vs |P| (halving)"+title_suffix)
    plt.xlabel("|P| (n/2, n/4, ..., 1)")
    plt.ylabel("Error (Max)")
    plt.xticks(xvals, [str(x) for x in xvals], rotation=45)
    plt.ylim(0, max(ERROR_VALUES_2)*1.1)
    plt.grid(True, axis="both")
    plt.legend(loc="upper right")

    # ---------------- Stretch vs |P| ----------------
    plt.subplot(1,2,2)
    for err_rate, group in avg.groupby("ErrRate"):
        sub = group.sort_values("Count", ascending=True)
        plt.plot(sub.Count, sub.Str, "-o", label=f"{err_rate:.1f} Stretch")

    plt.title("Stretch vs |P| (halving)"+title_suffix)
    plt.xlabel("|P| (n/2, n/4, ..., 1)")
    plt.ylabel("Stretch")
    plt.xticks(xvals, [str(x) for x in xvals], rotation=45)
    plt.grid(True, axis="both")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()



# ---------------------------- MAIN ----------------------------

if __name__ == "__main__":
    # res = simulate("64grid_diameter14test.edgelist")
    # res = simulate("144grid_diameter22test.edgelist")
    # res = simulate("256grid_diameter30test.edgelist")
    # res = simulate("576grid_diameter46test.edgelist")
    # res = simulate("1024grid_diameter62test.edgelist")
    # print("I collected", len(res), "data points")
    # df  = pd.DataFrame(res, columns=["Frac","ErrRate","Err","Str"])
    # print(df.head())
    # plot_results(res)

    # ===== Halving mode (new flow) =====
    # res_half = simulate_halving("256grid_diameter30test.edgelist")
    res_half = simulate_halving("576grid_diameter46test.edgelist")
    print("Halving mode collected", len(res_half), "points")
    df  = pd.DataFrame(res_half, columns=["Frac","ErrRate","Err","Str"])
    print(df.head())
    # plot_results_halving(res_half)
    plot_results_halving_counts(res_half, n=576)
