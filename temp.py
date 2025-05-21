# run110_merged.py
# Combined version: uses labmate's error plotting and original spiral/multibend logic

import networkx as nx
import math, os, random
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd

# ---------------------------- CONFIGURATION ----------------------------
PREDICTION_FRACTIONS = [0.03125, 0.0625, 0.125, 0.25, 0.5]
ERROR_VALUES_2       = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NUM_TRIALS           = 1
DEBUG                = True

# ---------------------------- SUBMESH HIERARCHY ----------------------------
def generate_type1_submeshes(size):
    levels = int(math.log2(size)) + 1
    hierarchy = defaultdict(list)
    for level in range(levels):
        b = 2**level
        for i in range(0, size, b):
            for j in range(0, size, b):
                nodes = {(x,y) for x in range(i, min(i+b, size))
                                for y in range(j, min(j+b, size))}
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
                nodes = {(x,y) for x in range(i, i+b)
                                for y in range(j, j+b)
                                if 0 <= x < size and 0 <= y < size}
                if nodes:
                    hierarchy[(level,1)].append(nodes)
    return hierarchy

def build_mesh_hierarchy(size):
    H = generate_type1_submeshes(size)
    H.update(generate_type2_submeshes(size))

    # level 0 unify
    H[(0,1)] = list(H[(0,2)])
    # root level
    hi = int(math.log2(size))
    all_nodes = {(x,y) for x in range(size) for y in range(size)}
    H[(hi+1,1)] = [all_nodes]
    H[(hi+1,2)] = [all_nodes]
    return H

def assign_cluster_leaders(H):
    M = defaultdict(list)
    for lvl_type, clusters in H.items():
        for cl in clusters:
            leader = min(cl)
            M[lvl_type].append((leader,cl))
    return M

# ---------------------------- PUBLISH / DOWNWARD LINKS ----------------------------
down_link = {}
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

def publish(owner, leader_map):
    global down_link
    sp = get_spiral(owner, leader_map)
    down_link.clear()
    rev = list(reversed(sp))
    for parent, child in zip(rev, rev[1:]):
        down_link[parent] = child

# ---------------------------- ROUTING METRICS ----------------------------
def calculate_error_updated(Vp, Q, G):
    diameter = nx.diameter(G)
    errors = []
    for req, pred in zip(Q, Vp):
        d = nx.shortest_path_length(G, source=req, target=pred)
        errors.append(d/diameter)
    return max(errors) if errors else 0

def calculate_stretch_spiral(pred_nodes, act_nodes, leader_map, G):
    sum_spiral = sum_shortest = count = 0
    for u,v in zip(pred_nodes, act_nodes):
        if u==v: continue
        pu = get_spiral(u, leader_map)
        pv = get_spiral(v, leader_map)
        common = set(pu)&set(pv)
        if not common: continue
        lca = next(n for n in reversed(pu) if n in common)
        iu,iv = pu.index(lca), pv.index(lca)
        path = pu[:iu+1]+list(reversed(pv[:iv]))
        if len(path)<=2: continue
        dsp = sum(nx.shortest_path_length(G,path[i],path[i+1]) for i in range(len(path)-1))
        ds  = nx.shortest_path_length(G,u,v)
        sum_spiral += dsp; sum_shortest += ds; count+=1
    return sum_spiral/sum_shortest if sum_shortest>0 else 1.0

# ---------------------------- GRAPH LOADING ----------------------------
def load_graph(file):
    G = nx.read_graphml(os.path.join("graphs","grid",file))
    return nx.relabel_nodes(G, lambda x:int(x))

# ---------------------------- SAMPLING ----------------------------
def choose_Vp(G, fraction):
    nodes = list(G.nodes()); random.shuffle(nodes)
    k = int(len(nodes)*fraction)
    orig = random.choices(nodes, k=k)
    return orig

def sample_Q_within_diameter(G, Vp, error):
    diam = nx.diameter(G)
    for _ in range(10000):
        Q = []
        for v in Vp:
            dist = nx.single_source_shortest_path_length(G,v,cutoff=int(diam/error) if error>0 else diam)
            Q.append(random.choice(list(dist.keys())))
        dups = Counter(Q)
        extra = sum(c-1 for c in dups.values())
        if extra/len(Q)*100<=100:
            return Q
    random.shuffle(Q)
    return Q

# ---------------------------- SIMULATION ----------------------------
def simulate(graph_file):
    G = load_graph(graph_file)
    size = int(math.sqrt(G.number_of_nodes()))
    H = build_mesh_hierarchy(size)
    leaders = assign_cluster_leaders(H)

    owner = random.choice(list(G.nodes()))
    publish(owner, leaders)
    results = []
    for err in ERROR_VALUES_2:
        for frac in PREDICTION_FRACTIONS:
            for _ in range(NUM_TRIALS):
                Vp = choose_Vp(G, frac)
                Q  = sample_Q_within_diameter(G, Vp, err)
                er = calculate_error_updated(Vp,Q,G)
                st = calculate_stretch_spiral(Vp,Q,leaders,G)
                results.append((frac,err,er,st))
    return results

# ---------------------------- PLOTTING ----------------------------
def plot_results(results):
    df = pd.DataFrame(results, columns=["Frac","ErrRate","Error","Stretch"])
    avg = df.groupby(["Frac","ErrRate"]).mean().reset_index()

    plt.figure(figsize=(12,5))
    # Error
    plt.subplot(1,2,1)
    for e in ERROR_VALUES_2:
        sub=avg[avg.ErrRate==e]
        plt.plot(sub.Frac, sub.Error,'-o',label=f"{e:.1f}")
    plt.title("Error vs Fraction")
    plt.xlabel("Fraction")
    plt.ylabel("Error")
    plt.legend()
    # Stretch
    plt.subplot(1,2,2)
    for e in ERROR_VALUES_2:
        sub=avg[avg.ErrRate==e]
        plt.plot(sub.Frac, sub.Stretch,'-o',label=f"{e:.1f}")
    plt.title("Stretch vs Fraction")
    plt.xlabel("Fraction")
    plt.ylabel("Stretch")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------- MAIN ----------------------------
if __name__=="__main__":
    res=simulate("64grid_diameter14test.edgelist")
    plot_results(res)
