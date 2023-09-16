import os

import numpy as np
import networkx as nx
import abc
from abc import ABC, abstractmethod
from itertools import combinations, groupby
import random

from matplotlib import pyplot as plt

np.random.seed(0)

class GraphFactory:
    @staticmethod
    def create_graph(config):
        if config['input_graph']["name"] == "simple":
            return SimpleGraph(config)
        if config['input_graph']["name"] == "karate":
            return KarateClub(config)
        if config['input_graph']["name"] == "cycle":
            return Cycle(config)
        if config['input_graph']["name"] == "cube":
            return Cube(config)
        if config['input_graph']["name"] == "dodecahedral":
            return Dodecahedral(config)
        if config['input_graph']["name"] == "star":
            return Star(config)
        if config['input_graph']["name"] == "grid":
            return Grid(config)
        if config['input_graph']["name"] == "random":
            return Random(config)
        if config['input_graph']["name"] == "barbell":
            return Barbell_graph(config)
        if config['input_graph']["name"] == "petersen":
            return Petersen_graph(config)
        if config['input_graph']["name"] == "tutte":
            return Tutte_graph(config)
        else:
            raise AttributeError(f"Graph Type {config['input_graph']['name']} is unknown.")

        # SIMPLE GRAPH


class Graph(ABC):

    def __init__(self, config):
        self.config = config
        self.E = None
        self.num_nodes = None
        self.shortest_paths = None
        self.G = None

    @abstractmethod
    def create_arcs_list(self):
        pass

    def get_arcs_list(self):
        return self.E

    def __call__(self, *args, **kwargs):
        self.create_arcs_list()
        self.num_nodes = np.max(self.E) + 1


class SimpleGraph(Graph):

    def create_arcs_list(self):
        e = [[0, 1, ], [0, 2, ], [0, 4, ], [1, 2, ], [1, 3, ], [2, 3, ], [2, 4, ]]
        self.E = np.asarray(e)


class KarateClub(Graph):

    def create_arcs_list(self):
        src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
                        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
                        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
                        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
                        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
        dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
                        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
                        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
                        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
                        31, 32])
        # Edges are directional in DGL; Make them bi-directional.
        u = np.column_stack([src, dst])
        self.E = u
        self.G = nx.karate_club_graph()
        if self.config['input_graph']["bidirectional"]:
            self.G = self.G.to_directed()
        if self.config["graph_drawer"]["stress"]:
            G = nx.karate_club_graph()
            self.num_nodes = G.number_of_nodes()
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az
            self.G = G


class Cycle(Graph):

    def create_arcs_list(self):
        G = nx.cycle_graph(n=10)
        if self.config['input_graph']["bidirectional"]:
            G = G.to_directed()
        self.E = np.asarray(G.edges)
        self.num_nodes = G.number_of_nodes()
        self.G = G
        # Edges are directional in DGL; Make them bi-directional.
        if self.config["graph_drawer"]["stress"]:
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az


class Cube(Graph):

    def create_arcs_list(self):
        G = nx.cubical_graph()
        if self.config['input_graph']["bidirectional"]:
            G = G.to_directed()
        self.E = np.asarray(G.edges)
        self.num_nodes = G.number_of_nodes()
        self.G = G
        # Edges are directional in DGL; Make them bi-directional.
        if self.config["graph_drawer"]["stress"]:
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az


class Dodecahedral(Graph):

    def create_arcs_list(self):
        G = nx.dodecahedral_graph()
        if self.config['input_graph']["bidirectional"]:
            G = G.to_directed()
        self.E = np.asarray(G.edges)
        self.num_nodes = G.number_of_nodes()
        self.G = G
        # Edges are directional in DGL; Make them bi-directional.
        if self.config["graph_drawer"]["stress"]:
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az

class Barbell_graph(Graph):

    def create_arcs_list(self):
        G = nx.barbell_graph(7, 2)
        if self.config['input_graph']["bidirectional"]:
            G = G.to_directed()
        self.E = np.asarray(G.edges)
        self.num_nodes = G.number_of_nodes()
        self.G = G
        # Edges are directional in DGL; Make them bi-directional.
        if self.config["graph_drawer"]["stress"]:
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az

class Petersen_graph(Graph):

    def create_arcs_list(self):
        G = nx.petersen_graph()
        if self.config['input_graph']["bidirectional"]:
            G = G.to_directed()
        self.E = np.asarray(G.edges)
        self.num_nodes = G.number_of_nodes()
        self.G = G
        # Edges are directional in DGL; Make them bi-directional.
        if self.config["graph_drawer"]["stress"]:
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az

class Tutte_graph(Graph):

    def create_arcs_list(self):
        G = nx.tutte_graph()
        if self.config['input_graph']["bidirectional"]:
            G = G.to_directed()
        self.E = np.asarray(G.edges)
        self.num_nodes = G.number_of_nodes()
        self.G = G
        # Edges are directional in DGL; Make them bi-directional.
        if self.config["graph_drawer"]["stress"]:
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az





class Star(Graph):
    def create_arcs_list(self):
        G = nx.star_graph(n=20)
        if self.config['input_graph']["bidirectional"]:
            G = G.to_directed()
        self.E = np.asarray(G.edges)
        self.num_nodes = G.number_of_nodes()
        self.G = G
        # Edges are directional in DGL; Make them bi-directional.
        if self.config["graph_drawer"]["stress"]:
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az


class Grid(Graph):
    def create_arcs_list(self):

        N = 7
        G = nx.grid_2d_graph(N, N)  # 2D regular graph of 10000 nodes
        if self.config['input_graph']["bidirectional"]:
            G = G.to_directed()
        pos = dict((n, n) for n in G.nodes())  # Dict of positions
        labels = dict(((i, j), i + (N - 1 - j) * N) for i, j in G.nodes())
        nx.relabel_nodes(G, labels, False)
        self.E = np.asarray(G.edges)
        self.num_nodes = G.number_of_nodes()
        self.G = G
        # Edges are directional in DGL; Make them bi-directional.
        if self.config["graph_drawer"]["stress"]:
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az


class Random(Graph):

    def create_arcs_list(self,):
        folder = os.path.join("data", "random_graph")
        if self.config['input_graph']['graph_id'] is None:
            graph_id = random.randint(0, 10000)
        else:
            graph_id = self.config['input_graph']['graph_id']
        path = os.path.join(folder, f"random_graph_{graph_id}.gpickle")
        if os.path.exists(path):
            G = nx.read_gpickle(path)
        else:
            n, p = self.config["input_graph"]["nodes"],  self.config["input_graph"]["p"]
            G = nx.generators.fast_gnp_random_graph(n, p)
            connected_component = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)][0]
            G = G.subgraph(connected_component).copy()
            # nx.draw_spring(G)
            # plt.show()
            G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')

        if self.config['input_graph']["bidirectional"]:
            G = G.to_directed()
        G = nx.convert_node_labels_to_integers(G)
        self.G = G
        self.E = np.asarray(G.edges)
        if self.config["graph_drawer"]["stress"]:
            self.num_nodes = G.number_of_nodes()
            shortest_paths = dict(nx.shortest_path_length(G))

            az = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    az[i][j] = shortest_paths[i][j]

            self.shortest_paths = az

