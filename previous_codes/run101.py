# Spiral Paper - research faithful Laminar cover hierarchy

import networkx as nx
import random
import math
from collections import defaultdict, deque

# --------------------- Helper Functions ---------------------

#Computes the ceiling of log base 2. Used for defining hierarchy depth (levels).
def log2_ceil(n):
    return math.ceil(math.log2(n))

#Returns all nodes within distance k of a given node. 
#Crucial for cluster formation (Spiral protocol needs locality-based clusters).
def get_k_neighborhood(G, node, k):
    return nx.single_source_shortest_path_length(G, node, cutoff=k).keys()

# --------------------- Laminar Hierarchy Construction ---------------------

def build_laminar_cover_hierarchy(G):
    D = nx.diameter(G)
    max_level = log2_ceil(D) + 1
    sigma = math.ceil(math.log2(len(G)))
    chi = math.ceil(math.log2(len(G)))

    hierarchy = defaultdict(list)
    hierarchy[0] = [(i % chi, {v}) for i, v in enumerate(G.nodes())]

    for level in range(1, max_level + 1):
        gamma = 2 ** (level - 1)
        prev_clusters = [cluster for _, cluster in hierarchy[level - 1]]
        cluster_map = {}
        new_clusters = []

        for cluster in prev_clusters:
            if frozenset(cluster) in cluster_map:
                continue
            center = next(iter(cluster))
            neighborhood = set(get_k_neighborhood(G, center, gamma))
            merged = set()
            for c in prev_clusters:
                if neighborhood & c:
                    merged |= c
                    cluster_map[frozenset(c)] = True
            new_clusters.append(merged)

        hierarchy[level] = [(i % chi, cluster) for i, cluster in enumerate(new_clusters)]

    return hierarchy, sigma, chi

# --------------------- Leader Assignment ---------------------

def assign_leaders(hierarchy):
    """
    Select a leader for each cluster in each level.
    Returns:
        leader_map[level][label] = leader_node
    """
    leader_map = defaultdict(dict)
    for level, clusters in hierarchy.items():
        for label, cluster in clusters:
            leader = min(cluster)  # Simple deterministic choice
            leader_map[level][label] = leader
    return leader_map

# --------------------- Spiral Path Construction ---------------------

def construct_spiral_path(u, hierarchy, leader_map, chi):
    """
    Returns the spiral path of node u as a list of nodes.
    """
    path = [u]
    levels = sorted(hierarchy.keys())
    for level in levels[1:]:  # skip level 0
        # Sort labels for this node
        labels = sorted([label for label, cluster in hierarchy[level] if u in cluster])
        for label in labels:
            leader = leader_map[level][label]
            if leader != path[-1]:  # avoid repeats
                path.append(leader)
    return path

# --------------------- Protocol Operations ---------------------

class SpiralProtocol:
    def __init__(self, G):
        self.G = G
        self.hierarchy, self.sigma, self.chi = build_laminar_cover_hierarchy(G)
        self.leader_map = assign_leaders(self.hierarchy)
        self.directory_path = {}  # {node: child}, forming a chain to the owner
        self.owner = None
        self.object_id = "X"  # placeholder object ID

    def publish(self, node):
        """Sets up the initial directory path pointing to the publishing node."""
        spiral_path = construct_spiral_path(node, self.hierarchy, self.leader_map, self.chi)
        print(f"[PUBLISH] Spiral path for node {node}: {spiral_path}")
        for i in range(len(spiral_path) - 1):
            self.directory_path[spiral_path[i+1]] = spiral_path[i]
        self.owner = node
        print(f"[PUBLISH] Object published by node {node}.")

    def lookup(self, requester):
        """
        Traverses spiral path to find the object owner.
        Follows directory pointers from the top.
        """
        spiral_path = construct_spiral_path(requester, self.hierarchy, self.leader_map, self.chi)
        for node in reversed(spiral_path):
            if node in self.directory_path:
                next_node = self.directory_path[node]
                print(f"[LOOKUP] Found pointer at {node} → {next_node}")
                return self.owner
        print("[LOOKUP] Owner not found in path — going to root")
        return self.owner

    def move(self, requester):
        """
        Moves ownership to the requester.
        Rebuilds the directory path to point to the new owner.
        """
        print(f"[MOVE] Requester {requester} wants to move object.")
        self.publish(requester)  # Rebuild path to new owner
        print(f"[MOVE] Object moved. New owner is {requester}.")


#----------------------get the graph----------------------
def load_graph_from_directory(filename="16grid_diameter6test.edgelist"):
    import os
    filepath = os.path.join("graphs", "grid", filename)
    return nx.read_graphml(filepath)

# --------------------- Example Simulation ---------------------

def simulate_spiral_protocol():
    # # Build a sample 4x4 grid
    # G = nx.grid_2d_graph(4, 4)
    # G = nx.convert_node_labels_to_integers(G)

    # Load the graph from a file
    # G = load_graph_from_directory("16grid_diameter6test.edgelist")
    G = load_graph_from_directory("64grid_diameter14test.edgelist")
    G = nx.relabel_nodes(G, lambda x: int(x))  # Convert string IDs to ints

    protocol = SpiralProtocol(G)
    protocol.publish(0)  # Node 0 owns the object

    # Node 5 performs lookup
    found_owner = protocol.lookup(5)
    print(f"[SIM] Node 5 found object at owner: {found_owner}")

    # Node 10 requests move (write access)
    protocol.move(10)

    # Node 3 performs lookup again
    found_owner = protocol.lookup(3)
    print(f"[SIM] Node 3 found object at new owner: {found_owner}")

# Run the simulation
if __name__ == "__main__":
    simulate_spiral_protocol()
