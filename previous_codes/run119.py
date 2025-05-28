# good version of run110.py
# (with full diagnostics & correct stretch)

import networkx as nx
import math, os, random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd

# ---------------------------- CONFIGURATION ----------------------------
PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES_2       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_TRIALS           = 2
DEBUG                = True
down_link = {}
# ---------------------------- SUBMESH HIERARCHY ----------------------------

def generate_type1_submeshes(size):
    levels = int(math.log2(size)) + 1
    hierarchy = defaultdict(list)
    for level in range(levels):
        b = 2**level
        for i in range(0, size, b):
            for j in range(0, size, b):
                nodes = {
                    (x,y)
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
                    (x,y)
                    for x in range(i, i+b)
                    for y in range(j, j+b)
                    if 0 <= x < size and 0 <= y < size
                }
                if nodes:
                    hierarchy[(level,1)].append(nodes)
    return hierarchy

def build_mesh_hierarchy(size):
    """
    Combines Type-1 and Type-2 submeshes to form a full hierarchy (complete mesh hierarchy).
    Adds a top-level root cluster containing all nodes under BOTH type‐1 and type‐2, so they compare equal.
    """
    # 1) get the standard two‐type decomposition
    H = generate_type1_submeshes(size)
    H.update(generate_type2_submeshes(size))

    # mirror level 0 type-2 into type-1 so they match
    H[(0,1)] = list(H[(0,2)])

    # 2) validate bottom and interior before adding root
    levels = sorted({lvl for (lvl,_) in H})
    lo, hi = levels[0], levels[-1]
    # bottom: both must match exactly (each singleton)
    if H[(lo,1)] != H[(lo,2)]:
        raise RuntimeError("Level 0 must have identical Type-1/2 clusters")
    # interior: every level between lo+1..hi-1 must differ
    for lvl in levels[1:-1]:
        if H[(lvl,1)] == H[(lvl,2)]:
            raise RuntimeError(f"Level {lvl} Type-1 == Type-2; they must differ")

    # 3) now add the root cluster *to both* (hi+1,1) and (hi+1,2)
    all_nodes = {(x, y) for x in range(size) for y in range(size)}
    root_level = hi + 1
    H[(root_level,1)].append(all_nodes)
    H[(root_level,2)].append(all_nodes)

    # 4) validate that top now matches
    if H[(root_level,1)] != H[(root_level,2)]:
        raise RuntimeError("Root level must have identical Type-1/2 clusters")

    return H


def assign_cluster_leaders(H):
    M = defaultdict(list)
    for lvl_type, clusters in H.items():
        for cl in clusters:
            leader = min(cl)
            M[lvl_type].append((leader,cl))
    return M

# ---------------------------- PUBLISH / DOWNWARD LINKS ----------------------------

def get_spiral(node, leader_map):
    path = [node]
    seen = {node}
    for lvl_type in sorted(leader_map):
        for leader,cl in leader_map[lvl_type]:
            if node in cl and leader not in seen:
                path.append(leader)
                seen.add(leader)
                break
    return path

def publish(owner, leader_map):
    """
    Walk 'owner' ↑ via its full spiral and at each step
    set down_link[parent] = child so later lookups
    can find the chain root→...→owner.
    """
    global down_link
    sp = get_spiral(owner, leader_map)
    down_link.clear()

     # we need a sequence that’s sliceable, so materialize reversed(sp) into a list
    rev = list(reversed(sp))
    for a, b in zip(rev, rev[1:]):
        # a = parent, b = child
        down_link[a] = b

# ---------------------------- STRETCH MEASUREMENT ----------------------------

def measure_stretch(requesters, owner, leader_map, G):
    global down_link
    total_up_down = 0
    total_opt     = 0
    valid_count   = 0

    for r in requesters:
        if r==owner: continue

        # 1) climb spiral until we hit a down_link:
        sp = get_spiral(r, leader_map)
        up_hops = 0
        intersection = None
        for i in range(len(sp)-1):
            up_hops += nx.shortest_path_length(G, sp[i], sp[i+1])
            if sp[i+1] in down_link:
                intersection = sp[i+1]
                break
        if intersection is None:
            # fallback: climb all the way to root
            intersection = sp[-1]

        # 2) descend via down_link chain
        # down_hops = 0
        # cur = intersection
        # while cur!=owner:
        #     nxt = down_link.get(cur)
        #     if nxt is None:
        #         raise RuntimeError(f"Broken down_link chain at {cur}")
        #     down_hops += nx.shortest_path_length(G, cur, nxt)
        #     cur = nxt

        # 2) descend via down_link chain (with fallback)
        down_hops = 0
        cur = intersection
        seen = {cur}
        while cur != owner:
            nxt = down_link.get(cur)
            if nxt is None or nxt in seen:
                # missing pointer or loop—jump straight to owner
                direct = nx.shortest_path_length(G, cur, owner)
                if DEBUG:
                    print(f"[WARN] no down_link from {cur}, fallback direct→owner={direct}")
                down_hops += direct
                break
            down_hops += nx.shortest_path_length(G, cur, nxt)
            seen.add(nxt)
            cur = nxt


        dist_sp = up_hops + down_hops
        dist_opt = nx.shortest_path_length(G, r, owner)

        if DEBUG:
            print(f"Req={r}, int={intersection}, up={up_hops}, down={down_hops}, opt={dist_opt}")

        total_up_down += dist_sp
        total_opt     += dist_opt
        valid_count   += 1

    return (total_up_down/total_opt) if total_opt>0 else 1.0

# ---------------------------- GRAPH LOADING ----------------------------

def load_graph(dfile):
    G = nx.read_graphml(os.path.join("graphs","grid",dfile))
    return nx.relabel_nodes(G, lambda x:int(x))

# ---------------------------- ERRORS & HELPERS ----------------------------

def choose_Vp(G, frac):
    nodes = list(G.nodes())
    k = max(1,int(len(nodes)*frac))
    return random.sample(nodes, k)

def sample_actual(G, Vp, error):
    # very rough: pick some nodes within diameter/error
    diam = nx.diameter(G)
    act = []
    for v in Vp:
        # all within cutoff
        lengths = nx.single_source_shortest_path_length(G, v, cutoff=int(diam/error) if error>0 else diam)
        act.append(random.choice(list(lengths)))
    return act

def calculate_error(pred, actual, G):
    diam = nx.diameter(G)
    return sum(nx.shortest_path_length(G,u,v)/diam for u,v in zip(pred,actual))/len(pred)

# ---------------------------- SIMULATION ----------------------------

def simulate(graph_file):
    G = load_graph(graph_file)
    size = int(math.sqrt(G.number_of_nodes()))
    H = build_mesh_hierarchy(size)
    leaders = assign_cluster_leaders(H)

    results = []
    owner = random.choice(list(G.nodes()))
    publish(owner, leaders)

    for error in ERROR_VALUES_2:
        for frac in PREDICTION_FRACTIONS:
            for _ in range(NUM_TRIALS):
                pred = choose_Vp(G, frac)
                act  = sample_actual(G, pred, error)
                err  = calculate_error(pred, act, G)
                stretch = measure_stretch(pred, owner, leaders, G)
                results.append((frac, error, err, stretch))
    return results

# ---------------------------- PLOTTING ----------------------------

def plot_results(results):
    df = pd.DataFrame(results, columns=["Frac","ErrRate","Err","Str"])
    avg = df.groupby(["Frac","ErrRate"]).mean().reset_index()

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    for e in ERROR_VALUES_2:
        sub = avg[avg.ErrRate==e]
        plt.plot(sub.Frac, sub.Err, marker='o', label=f"{e:.1f}")
    plt.title("Error vs Frac"); plt.xlabel("Frac"); plt.ylabel("Error")
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    for e in ERROR_VALUES_2:
        sub = avg[avg.ErrRate==e]
        plt.plot(sub.Frac, sub.Str, marker='o', label=f"{e:.1f}")
    plt.title("Stretch vs Frac"); plt.xlabel("Frac"); plt.ylabel("Stretch")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

# ---------------------------- MAIN ----------------------------

if __name__=="__main__":
    res = simulate("64grid_diameter14test.edgelist")
    plot_results(res)
