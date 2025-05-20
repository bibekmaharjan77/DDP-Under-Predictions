import random
import math
import matplotlib.pyplot as plt

# **1. Define the Network Topology and Hierarchical Clusters**
# Here we simulate a 16x16 mesh network. We partition it into Type-1 submeshes (clusters) of size 4x4,
# and a single Type-2 submesh that covers the entire network (root cluster).
# Each Type-1 cluster has a designated leader (we choose the top-left node of the cluster),
# and the Type-2 cluster (root) has a leader (chosen here as node (0,0) for simplicity).
N = 16
cluster_size = 4
cluster_count = N // cluster_size  # should be 4 for N=16 and cluster_size=4

# Map each cluster (by its index in the grid of clusters) to its leader node coordinates.
cluster1_leaders = {}
for cx in range(cluster_count):
    for cy in range(cluster_count):
        leader_node = (cx * cluster_size, cy * cluster_size)  # top-left of the cluster
        cluster1_leaders[(cx, cy)] = leader_node

# Define the root cluster leader for the Type-2 submesh (the entire 16x16 mesh).
root_leader = (0, 0)  # using (0,0) as the root leader (this node also happens to be a Type-1 leader of cluster (0,0)).

# Helper functions for cluster lookups and distance in the mesh.
def cluster1_index(node):
    """Return the (cx, cy) index of the Type-1 cluster containing the given node."""
    x, y = node
    return (x // cluster_size, y // cluster_size)

def cluster1_leader(node):
    """Return the leader node of the Type-1 cluster containing the given node."""
    return cluster1_leaders[cluster1_index(node)]

def distance(a, b):
    """Manhattan distance (number of hops) between two nodes a and b in the 2D mesh."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# **2. Define Functions to Simulate Publish and Move Operations with Downward Pointers**
# We maintain a dictionary `pointer` to represent downward pointers in the hierarchy.
# After a publish operation from the owner, `pointer[root_leader]` points to the owner’s Type-1 cluster leader (or directly to owner if in root’s cluster),
# and `pointer[cluster_leader]` (for the owner's cluster) points to the owner node.
# The `perform_move` function simulates a move request:
#    - It builds the upward path from the requester to its cluster leaders, checking for a downward pointer at each level.
#    - If a pointer is found at the requester’s cluster leader, it follows it down to the owner immediately.
#    - If not, it goes up to the root and uses the root’s pointer to descend to the owner’s cluster leader and then to the owner.
#    - The function returns the total path length (hops) taken by this request (upward + downward path) **plus** the final hop from owner to requester (object transfer),
#      and also returns the Manhattan shortest path distance between original owner and requester (for stretch calculation).
# After the move, it updates the `pointer` dictionary by performing a publish from the new owner (setting downward pointers from root to the new owner).
def perform_move(requester_node, current_owner, pointer):
    """
    Simulate a move request initiated by `requester_node` for the object currently at `current_owner`.
    Returns a tuple (new_owner, alg_distance, opt_distance):
      - new_owner: the requester_node (now the new owner after the move).
      - alg_distance: total hops taken by the request (upward + downward traversal plus object delivery).
      - opt_distance: shortest path hops between the original owner and the requester (optimal direct distance).
    Updates the pointer dictionary to reflect a publish from the new owner after the move.
    """
    orig_owner = current_owner
    alg_distance = 0

    # Upward phase: requester goes up to its Type-1 cluster leader.
    req_cluster_leader = cluster1_leader(requester_node)
    alg_distance += distance(requester_node, req_cluster_leader)

    # Check for downward pointer at the Type-1 cluster leader.
    if req_cluster_leader in pointer and pointer[req_cluster_leader] is not None:
        # Downward pointer found at cluster leader -> follow it.
        target = pointer[req_cluster_leader]
        if target != orig_owner:
            # If pointer leads to an intermediate leader (in this 2-level setup this shouldn't happen,
            # because any pointer at a Type-1 leader should point directly to the owner).
            # Nonetheless, handle generically: follow pointer chain downward.
            alg_distance += distance(req_cluster_leader, target)
            if target in pointer and pointer[target] is not None:
                alg_distance += distance(target, pointer[target])
                target = pointer[target]
            # Now target should be the owner node.
            alg_distance += distance(target, orig_owner)
        else:
            # Pointer directly leads to the owner.
            alg_distance += distance(req_cluster_leader, orig_owner)
    else:
        # No pointer at requester’s cluster leader -> go up to root leader.
        alg_distance += distance(req_cluster_leader, root_leader)
        # At root, there must be a pointer down to the owner's cluster (from the last publish).
        target = pointer.get(root_leader)
        if target is not None:
            alg_distance += distance(root_leader, target)
            if target != orig_owner:
                # If the root's pointer goes to an intermediate cluster leader (owner's cluster leader),
                # add distance from that leader down to the owner.
                alg_distance += distance(target, orig_owner)
        # (If `target` were None, it would mean no publish info is available, which should not happen after initialization.)

    # Downward phase complete; now the object (data) is sent from the owner to the requester.
    alg_distance += distance(orig_owner, requester_node)

    # Compute optimal distance for this move (direct shortest path from original owner to requester).
    opt_distance = distance(orig_owner, requester_node)

    # Update owner to the requester (the object moves here).
    new_owner = requester_node

    # **Publish**: reset downward pointers from root to the new owner.
    pointer.clear()
    owner_cl_leader = cluster1_leader(new_owner)
    if owner_cl_leader == root_leader:
        # If new owner is in the root's cluster (i.e., root and owner share the same Type-1 cluster),
        # set root's pointer directly to the owner.
        pointer[root_leader] = new_owner
    else:
        # Otherwise, set root pointer to the owner's cluster leader, and that leader to the owner.
        pointer[root_leader] = owner_cl_leader
        pointer[owner_cl_leader] = new_owner

    return new_owner, alg_distance, opt_distance

# **3. Simulation of Move Requests with Predictions and Stretch Calculation**
# The `simulate_moves` function runs a sequence of move requests on the network, with a given prediction fraction and prediction error rate.
# - pred_fraction: fraction of move requests that are predicted (proactively initiated before the actual request).
# - error_rate: probability that a prediction is incorrect (the object is moved to the wrong node).
# The simulation chooses random requesting nodes for each move (uniformly over the mesh).
# For each actual move request:
#    - If a prediction is to be made (based on pred_fraction), simulate a predicted move:
#         * If the prediction is correct (with probability 1 - error_rate), move the object to the actual requester in advance.
#           (The upcoming request will find the object already there, incurring no additional cost at request time.)
#         * If the prediction is wrong, move the object to a wrong node, and later when the actual request arrives, a second move is performed from that wrong node to the real requester.
#    - If no prediction is made, simply perform the move when requested.
# The stretch is calculated as (total path length taken by the algorithm) / (shortest path length if the object moved directly for each request).
# We aggregate total algorithm path length and total optimal path length for all move requests, then compute their ratio as the average stretch.
def simulate_moves(num_moves, pred_fraction, error_rate):
    # Initialize a random object owner.
    current_owner = (random.randrange(N), random.randrange(N))
    # Perform initial publish from this owner (set initial downward pointers from root to owner).
    pointer = {}
    owner_cl_leader = cluster1_leader(current_owner)
    if owner_cl_leader == root_leader:
        pointer[root_leader] = current_owner
    else:
        pointer[root_leader] = owner_cl_leader
        pointer[owner_cl_leader] = current_owner

    total_alg_distance = 0
    total_opt_distance = 0

    for _ in range(num_moves):
        # Choose a random requester node for the move.
        requester = (random.randrange(N), random.randrange(N))
        if requester == current_owner:
            # If the requester already holds the object (current owner), skip the move (no cost or change in ownership).
            continue

        original_owner = current_owner  # save for computing optimal distance

        # Decide if we predict this move ahead of time.
        if random.random() < pred_fraction:
            # **Prediction phase**: attempt to move the object before the actual request.
            if random.random() < (1 - error_rate):
                # Prediction is correct – we predict the actual requester.
                predicted_target = requester
                # Perform the move to predicted_target (object moves to requester in advance).
                current_owner, alg_dist, opt_dist = perform_move(predicted_target, current_owner, pointer)
                total_alg_distance += alg_dist
                total_opt_distance += opt_dist
                # (The actual request will find the object at requester with no additional cost.)
            else:
                # Prediction is incorrect – choose a wrong target to move the object.
                predicted_target = requester
                while predicted_target == requester:
                    # Pick a random node that is not the true requester.
                    predicted_target = (random.randrange(N), random.randrange(N))
                # Perform the move to the wrong predicted target.
                current_owner, alg_dist1, opt_dist1 = perform_move(predicted_target, current_owner, pointer)
                total_alg_distance += alg_dist1
                total_opt_distance += opt_dist1
                # Now the object is at the wrong node. When the actual request arrives, we must move it again.
                current_owner, alg_dist2, opt_dist2 = perform_move(requester, current_owner, pointer)
                total_alg_distance += alg_dist2
                total_opt_distance += opt_dist2
        else:
            # **No prediction**: perform the move when requested.
            current_owner, alg_dist, opt_dist = perform_move(requester, current_owner, pointer)
            total_alg_distance += alg_dist
            total_opt_distance += opt_dist

    # Compute average stretch as the ratio of total algorithm path length to total optimal path length.
    # (If no moves were performed, stretch can be None or 0 – here we handle the case to avoid division by zero.)
    if total_opt_distance == 0:
        return 0.0
    return total_alg_distance / total_opt_distance

# **4. Run Simulations for Various Prediction Fractions and Error Rates, Collecting Stretch Results**
prediction_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
error_rates = [0.0, 0.25, 0.5, 0.75, 1.0]

# We will average multiple simulation runs for each scenario to smooth out randomness.
runs_per_scenario = 5
moves_per_run = 500

# Dictionary to hold averaged stretch results: keys are error rates, values are lists of stretch values for each prediction fraction.
stretch_results = {er: [] for er in error_rates}

for er in error_rates:
    for pf in prediction_fractions:
        total_stretch = 0.0
        for _ in range(runs_per_scenario):
            # Run simulation for a given pred_fraction and error_rate.
            stretch = simulate_moves(moves_per_run, pf, er)
            total_stretch += stretch
        avg_stretch = total_stretch / runs_per_scenario
        stretch_results[er].append(avg_stretch)

# **5. Plot the Stretch Results**
plt.figure(figsize=(6, 4))
for er in error_rates:
    plt.plot(prediction_fractions, stretch_results[er], marker='o', label=f'Error Rate = {er}')
plt.xlabel('Prediction Fraction')
plt.ylabel('Average Stretch')
plt.title('Stretch vs Prediction Fraction for Various Error Rates')
plt.xticks(prediction_fractions)  # show all fraction values on x-axis
plt.legend(title='Legend')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
