# network_mining.py
# Simple Network Mining on Les Miserables Graph Dataset

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load the Les Miserables GML graph file
G = nx.read_gml('data/lesmiserables.gml', label='label')

# Draw the full graph with basic styling
plt.figure(figsize=(10, 7))
nx.draw(G, node_size=50, edge_color='b', alpha=0.2, font_size=8, with_labels=True)
plt.title('Les Miserables Character Network')
plt.show()

# Degree analysis
degree_dict = dict(G.degree())
degrees = np.array(list(degree_dict.values()))

print(f"Minimum degree: {degrees.min()}")
print(f"25th percentile: {np.percentile(degrees, 25)}")
print(f"Median degree: {np.median(degrees)}")
print(f"75th percentile: {np.percentile(degrees, 75)}")
print(f"Maximum degree: {degrees.max()}")

# Filter graph to nodes with degree > 10
Gt = G.copy()
nodes_to_remove = [node for node, deg in degree_dict.items() if deg <= 10]
Gt.remove_nodes_from(nodes_to_remove)

# Draw filtered graph
plt.figure(figsize=(10, 7))
nx.draw(Gt, node_size=50, edge_color='b', alpha=0.4, font_size=10, with_labels=True)
plt.title('Filtered Les Miserables Network (Degree > 10)')
plt.show()
