import random
from collections import defaultdict
import math

# Documentation and Implementation for Cluster Construction

# This function computes the k-neighborhood of a node in the graph.
def get_k_neighborhood(G, node, k):
    """
    Returns the set of nodes within distance ≤ k from the given node.
    """
    return nx.single_source_shortest_path_length(G, node, cutoff=k).keys()

# Function to construct a sparse cover for a given level
def construct_sparse_cover(G, level, sigma=2):
    """
    Constructs a sparse cover for a given level.
    
    Parameters:
    - G: The input graph.
    - level: The level index in the hierarchy.
    - sigma: Factor used to define max cluster diameter (default 2).
    
    Returns:
    - clusters: A list of clusters, each cluster is a set of node IDs.
    """
    gamma = 2 ** (level - 1)
    uncovered = set(G.nodes())
    clusters = []
    while uncovered:
        center = random.choice(list(uncovered))
        neighborhood = set(get_k_neighborhood(G, center, gamma))
        clusters.append(neighborhood)
        uncovered -= neighborhood
    return clusters, gamma, sigma

# Hierarchy construction function
def build_labeled_cover_hierarchy(G, max_level=None):
    """
    Builds a (σ, χ)-labeled cover hierarchy as described in the Spiral protocol.
    
    Returns:
    - hierarchy: A dict {level: [(label, cluster_set), ...]}
    """
    if max_level is None:
        max_level = log2_ceil(nx.diameter(G)) + 1

    hierarchy = defaultdict(list)
    sigma = math.ceil(math.log2(len(G)))
    chi = math.ceil(math.log2(len(G)))

    for level in range(1, max_level + 1):
        clusters, gamma, _ = construct_sparse_cover(G, level, sigma)
        for idx, cluster in enumerate(clusters):
            hierarchy[level].append((idx % chi, cluster))  # Label clusters cyclically up to χ

    return hierarchy, sigma, chi

# Build the hierarchy on the loaded graph
cover_hierarchy, SIGMA, CHI = build_labeled_cover_hierarchy(G)

# Show a summary: how many clusters per level
cluster_summary = {level: len(clusters) for level, clusters in cover_hierarchy.items()}
cluster_summary
