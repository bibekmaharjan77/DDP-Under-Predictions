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
    all_nodes = {(x, y) for x in range(size) for y in range(size)}
    root_level = hi + 1
    H[(root_level,1)].append(all_nodes)
    H[(root_level,2)].append(all_nodes)
    if H[(root_level,1)] != H[(root_level,2)]:
        raise RuntimeError("Root level must have identical Type-1/2 clusters")

    return H

def assign_cluster_leaders(H):
    M = defaultdict(list)
    for lvl_type, clusters in H.items():
        for cl in clusters:
            # pick e.g. the minimum node as our deterministic leader
            leader = min(cl)
            M[lvl_type].append((leader,cl))
    return M

# ---------------------------- PUBLISH / DOWNWARD LINKS ----------------------------

def get_spiral(node, leader_map):
    path = [node]
    seen = {node}
    for lvl_type in sorted(leader_map):
        # find the one cluster at that sub-level which contains `node`
        for leader, cl in leader_map[lvl_type]:
            if node in cl and leader not in seen:
                path.append(leader)
                seen.add(leader)
                break
    return path

def publish(owner, leader_map):
    """
    Build our single linked-list from root→…→owner
    so that lookups can follow down_link[] afterwards.
    """
    global down_link
    sp = get_spiral(owner, leader_map)
    down_link.clear()
    rev = list(reversed(sp))
    for parent, child in zip(rev, rev[1:]):
        down_link[parent] = child

# ---------------------------- STRETCH MEASUREMENT ----------------------------

def measure_stretch(requesters, owner, leader_map, G):
    """
    For each r in `requesters`, climb its spiral until we
    hit a down_link[], then descend that chain.  Compare
    total hops vs true shortest‐path hops.
    """
    global down_link
    total_up_down = 0
    total_opt     = 0

    for r in requesters:
        if r == owner:
            continue

        # 1) climb spiral
        sp = get_spiral(r, leader_map)
        up_hops = 0
        intersection = None
        for i in range(len(sp)-1):
            up_hops += nx.shortest_path_length(G, sp[i], sp[i+1])
            if sp[i+1] in down_link:
                intersection = sp[i+1]
                break
        if intersection is None:
            intersection = sp[-1]

        # 2) descend via down_link chain (with fallback)
        down_hops = 0
        cur = intersection
        seen = {cur}
        while cur != owner:
            nxt = down_link.get(cur)
            if nxt is None or nxt in seen:
                # broken chain → jump straight to owner
                direct = nx.shortest_path_length(G, cur, owner)
                if DEBUG:
                    print(f"[WARN] fallback direct from {cur} → {owner} = {direct}")
                down_hops += direct
                break
            down_hops += nx.shortest_path_length(G, cur, nxt)
            seen.add(nxt)
            cur = nxt

        dist_sp  = up_hops + down_hops
        dist_opt = nx.shortest_path_length(G, r, owner)

        if DEBUG:
            print(f"Req={r}, int={intersection}, up={up_hops}, down={down_hops}, opt={dist_opt}")

        total_up_down += dist_sp
        total_opt     += dist_opt

    return (total_up_down/total_opt) if total_opt>0 else 1.0

# ---------------------------- GRAPH LOADING ----------------------------

def load_graph(dfile):
    G = nx.read_graphml(os.path.join("graphs","grid",dfile))
    return nx.relabel_nodes(G, lambda x:int(x))

# ---------------------------- ERRORS & HELPERS ----------------------------

def choose_Vp(G, frac):
    k = max(1, int(len(G)*frac))
    return random.sample(list(G), k)

def sample_actual(G, Vp, error):
    diam = nx.diameter(G)
    act = []
    for v in Vp:
        cutoff = int(diam/error) if error>0 else diam
        lengths = nx.single_source_shortest_path_length(G, v, cutoff=cutoff)
        act.append(random.choice(list(lengths.keys())))
    return act

def calculate_error(pred, actual, G):
    diam = nx.diameter(G)
    return sum(nx.shortest_path_length(G,u,v)/diam
               for u,v in zip(pred,actual)) / len(pred)

# ---------------------------- SIMULATION ----------------------------

def simulate(graph_file):
    G = load_graph(graph_file)
    size = int(math.sqrt(G.number_of_nodes()))
    H = build_mesh_hierarchy(size)
    leaders = assign_cluster_leaders(H)

    results = []
    owner = random.choice(list(G.nodes()))
    # build our one directory once
    publish(owner, leaders)

    for error in ERROR_VALUES_2:
        for frac in PREDICTION_FRACTIONS:
            for _ in range(NUM_TRIALS):
                pred    = choose_Vp(G, frac)
                act     = sample_actual(G, pred, error)
                err     = calculate_error(pred, act, G)

                # ==== FIX #1: measure stretch on the ACTUAL path, not the predicted set! ====
                stretch = measure_stretch(act, owner, leaders, G)

                results.append((frac, error, err, stretch))
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
    for e in ERROR_VALUES_2:
        sub = avg[ avg.ErrRate == e ]
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


# ---------------------------- MAIN ----------------------------

if __name__ == "__main__":
    res = simulate("64grid_diameter14test.edgelist")
    plot_results(res)
