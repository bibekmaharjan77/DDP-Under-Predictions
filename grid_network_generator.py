import sys
import os
import math
from random import randint
from matplotlib import pyplot as plt
import networkx as nx
import random

num_nodes = 576

def add_edge_weights(graph):
    for e in graph.edges:
        w = 1
        graph.add_edge(e[0], e[1], weight=w)


def build_grid_graph():
    grid_row = math.ceil(math.sqrt(num_nodes))
    grid_col = math.floor(math.sqrt(num_nodes))
    grid_graph = nx.grid_graph([grid_row, grid_col])
    grid_graph = nx.convert_node_labels_to_integers(grid_graph)
    # grid_graph = nx.grid_2d_graph(grid_row, grid_col)
    assert nx.is_connected(grid_graph)
    add_edge_weights(grid_graph)
    return grid_graph

def write_to_a_file(graph, param):
    diameter = nx.diameter(graph, weight='weight')
    nx.write_graphml(graph, './graphs/grid/' + str(num_nodes) + str(param) + '_diameter' + str(diameter) + 'test.edgelist')


def draw(graph):
    plt.axis('off')

    node_pos = nx.spring_layout(graph)

    edge_weight = nx.get_edge_attributes(graph, 'weight')
    # Draw the nodes
    nx.draw_networkx(graph, node_pos, node_color='grey', node_size=100)

    # Draw the edges
    nx.draw_networkx_edges(graph, node_pos, edge_color='black')

    # Draw the edge labels
    nx.draw_networkx_edge_labels(graph, node_pos, edge_labels=edge_weight)
    print(edge_weight)
    print(graph)
    print(graph.nodes())
    print("THE GRAPH")
    plt.show()

def build_graphs():
    grid_graph = build_grid_graph()
    write_to_a_file(grid_graph, "grid")
    draw(grid_graph)

if __name__ == '__main__':
    build_graphs()