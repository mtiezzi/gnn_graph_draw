import os
import random
from itertools import groupby, combinations

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


def create_random_graph(n: int = 20, p: float = 0.1, show=False):
	G = nx.generators.fast_gnp_random_graph(n, p)
	connected_component = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)][0]
	G = G.subgraph(connected_component).copy()
	G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')
	if show:
		nx.draw_kamada_kawai(G)
		plt.show()
	return G


if __name__ == "__main__":
	n_graph = 10000
	max_nodes = 60
	min_nodes = 10
	min_p = 0.01
	max_p = 0.05

	folder = os.path.join("data", "random_graph")
	image_folder = os.path.join("data", "random_graph", "images")
	if not os.path.exists(folder):
		os.makedirs(folder)
	if not os.path.exists(image_folder):
		os.makedirs(image_folder)
		print("Created folder")

	i = 0
	stats = {
		"n_nodes": [],
		"n_edges": [],
	}
	while i <= n_graph:
		file = f"random_graph_{i}"
		path = os.path.join(folder, file)
		image_path = os.path.join(image_folder, file)

		n_i = random.randint(min_nodes, max_nodes)
		p_i = random.uniform(min_p, max_p)
		G_i = create_random_graph(n_i, p_i)

		created_nodes = len(G_i)
		created_edges = len(G_i.edges)
		if created_nodes > min_nodes and created_nodes < 30 and not (created_nodes > 30 and created_edges > 120):
			print(f"{i}) n = {n_i}, p = {p_i} nodes {created_nodes} edges {created_edges}")
			nx.write_gpickle(G_i, path + ".gpickle")
			stats['n_nodes'].append(created_nodes)
			stats['n_edges'].append(created_edges)
			nx.draw_kamada_kawai(G_i)
			plt.savefig(image_path + ".png")
			#plt.show()
			i += 1
			plt.close()

	file_csv = os.path.join(folder, "stats.csv")
	pd.DataFrame(stats).to_csv(file_csv, index=False)

