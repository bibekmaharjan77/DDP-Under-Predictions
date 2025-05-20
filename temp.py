import math
import random
import networkx as nx
import matplotlib.pyplot as plt

# global down_link mapping
down_link = {}

# -----------------------------------------------------------------------------
# Spiral/path utilities (assumed already implemented)
# -----------------------------------------------------------------------------
def get_spiral(node, leader_map):
    # returns the list of leaders from node up to root
    # (implementation depends on your hierarchy)
    raise NotImplementedError

# -----------------------------------------------------------------------------
# Publish: build the directory path for the 'owner'
# -----------------------------------------------------------------------------

def publish(owner, leader_map):
    """
    Walk 'owner' ↑ via its full spiral and at each step
    set down_link[parent] = child so later lookups
    and moves can reconstruct the root→...→owner chain.
    """
    global down_link
    down_link.clear()
    sp = get_spiral(owner, leader_map)
    # we must convert reversed(sp) to a list so slicing works
    rev = list(reversed(sp))
    # link each parent→child pair
    for parent, child in zip(rev, rev[1:]):
        down_link[parent] = child

# -----------------------------------------------------------------------------
# Stretch measurement: follow up-phase then down_link chain
# -----------------------------------------------------------------------------
def measure_stretch(requesters, owner, leader_map, G, DEBUG=False):
    """
    For each requester in 'requesters', climb its spiral until
    hitting a down_link, then descend via down_link to 'owner'.
    Compare that path-length to the direct shortest-path in G.
    Returns the ratio total_traversed / total_direct.
    """
    global down_link
    total_up_down = 0
    total_opt     = 0

    for r in requesters:
        if r == owner:
            continue

        # 1) climb spiral until we hit a down_link
        sp = get_spiral(r, leader_map)
        up_hops = 0
        intersection = None
        for i in range(len(sp)-1):
            u, v = sp[i], sp[i+1]
            up_hops += nx.shortest_path_length(G, u, v)
            if v in down_link:
                intersection = v
                break
        if intersection is None:
            # fallback: climb all the way to root
            intersection = sp[-1]

        # 2) descend via down_link chain
        down_hops = 0
        cur = intersection
        while cur != owner:
            nxt = down_link.get(cur)
            if nxt is None:
                raise RuntimeError(f"Broken down_link chain at {cur}")
            down_hops += nx.shortest_path_length(G, cur, nxt)
            cur = nxt

        dist_sp  = up_hops + down_hops
        dist_opt = nx.shortest_path_length(G, r, owner)

        if DEBUG:
            print(f"Req={r}, int={intersection}, up={up_hops}, down={down_hops}, opt={dist_opt}")

        total_up_down += dist_sp
        total_opt     += dist_opt

    return (total_up_down / total_opt) if total_opt > 0 else 1.0

# -----------------------------------------------------------------------------
# Simulation driver
# -----------------------------------------------------------------------------
def simulate(graph_file,
             ERROR_VALUES_2,
             PREDICTION_FRACTIONS,
             NUM_TRIALS):
    G = load_graph(graph_file)
    size = int(math.sqrt(G.number_of_nodes()))
    H = build_mesh_hierarchy(size)
    leaders = assign_cluster_leaders(H)

    results = []
    owner = random.choice(list(G.nodes()))

    for error in ERROR_VALUES_2:
        for frac in PREDICTION_FRACTIONS:
            for _ in range(NUM_TRIALS):
                pred = choose_Vp(G, frac)
                act  = sample_actual(G, pred, error)

                # rebuild directory for this trial
                publish(owner, leaders)

                err     = calculate_error(pred, act, G)
                stretch = measure_stretch(act, owner, leaders, G)

                results.append((frac, err, stretch))

    return results

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    graph_file = 'mesh16x16.edgelist'
    ERROR_VALUES_2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    PREDICTION_FRACTIONS = [0.05, 0.10, 0.25, 0.50]
    NUM_TRIALS         = 50

    data = simulate(graph_file,
                    ERROR_VALUES_2,
                    PREDICTION_FRACTIONS,
                    NUM_TRIALS)

    # unpack for plotting
    fracs, errs, stretches = zip(*data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    ax1.plot(fracs, errs, 'o-')
    ax1.set_title('Error vs Fraction')
    ax1.set_xlabel('Fraction')
    ax1.set_ylabel('Measured Error')

    ax2.plot(fracs, stretches, 'o-')
    ax2.set_title('Stretch vs Fraction')
    ax2.set_xlabel('Fraction')
    ax2.set_ylabel('Stretch')

    plt.tight_layout()
    plt.show()
