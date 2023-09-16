import torch
import torch.nn as nn
import numpy as np
from typing import List
import networkx as nx
from scipy import sparse as sp
from scipy.sparse.linalg import norm
import dgl


def minimize_edge_variance(edge_parameters: torch.Tensor, E: np.array, norm: int = 2) -> torch.float:
    """
    Function which returns the variance of the length of the edges

    :param edge_parameters: list of vertices positions
    :param E: list of edges specified as tuple of connected nodes indices
    :param norm: norm to be used to calculate the distances (default Euclidean)
    """
    # edge ([x1, y1], [x2, y2])

    # edges = [edge_parameters[arc] for arc in E]
    #
    # edge_lengths = torch.stack([torch.dist(edge[0], edge[1], p=norm) for edge in edges])
    #
    # variance = torch.var(edge_lengths)

    # tensorial form - maybe faster
    pdist = nn.PairwiseDistance(p=2, eps=1e-16)
    tensor_edges_coordinates = edge_parameters[E[None, :]]
    edge_length = pdist(tensor_edges_coordinates[:, 0, :], tensor_edges_coordinates[:, 1, :])
    variance = torch.var(edge_length)

    return variance


def maximize_node_distances(edge_parameters: torch.Tensor) -> torch.float:
    """
    Function which return the sum of the inverse of the distances between graph nodes
    :param edge_parameters: list of vertices positions
    :return:
    """
    # distances = 0.
    # for i, edge1 in enumerate(edge_parameters):
    #     for j, edge2 in enumerate(edge_parameters):
    #         if j > i:
    #             # distances += 1 / torch.dist(edge1, edge2)
    #             distances += 1 / torch.dist(edge1, edge2)
    # return distances
    # tensorial form
    matrix_distances = torch.triu(
        torch.cdist(edge_parameters, edge_parameters))  # get upper triangle of the distance matrix
    sum_dist = torch.sum(1 / (matrix_distances[matrix_distances.nonzero(as_tuple=True)]))
    # sum_dist = 1/torch.sum(matrix_distances)
    return sum_dist


def shortest_path_computation(G):
    num_nodes = G.number_of_nodes()
    shortest_paths = dict(nx.shortest_path_length(G))

    shortest_p = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            shortest_p[i][j] = shortest_paths[i][j]

    range_graph = torch.arange(num_nodes)
    couples_indices = torch.combinations(range_graph, r=2)

    return shortest_p, couples_indices


class Stress_loss:
    def __init__(self, reduction="mean", normalize=True):
        self.loss_c = nn.MSELoss(reduction=reduction)
        self.normalize = normalize
        self.pdist = torch.nn.PairwiseDistance()

    def stress_loss(self, logits, targets, full_graph=True):
        shortest_paths, couples_indices = targets

        loss = 0.0

        if full_graph:
            sources = couples_indices[:, 0]
            dest = couples_indices[:, 1]
            coordinates_sources = logits[sources]
            coordinates_dest = logits[dest]

            targets = shortest_paths[sources, dest]
            if self.normalize:
                distances = self.pdist(coordinates_sources, coordinates_dest) * 1 / targets
            else:
                distances = self.pdist(coordinates_sources, coordinates_dest)
            loss = self.loss_c(distances, targets)

        return loss


class StressCorrected:
    def __init__(self, ):
        self.loss_c = nn.MSELoss()
        self.pdist = torch.nn.PairwiseDistance()

    def stress_loss(self, logits, targets, full_graph=True):
        shortest_paths, couples_indices = targets

        loss = 0.0

        if full_graph:
            sources = couples_indices[:, 0]
            dest = couples_indices[:, 1]
            coordinates_sources = logits[sources]
            coordinates_dest = logits[dest]

            delta = shortest_paths[sources, dest]

            distance = self.pdist(coordinates_sources, coordinates_dest)
            weight = 1 / (delta + 1e-7)
            loss = weight * self.loss_c(distance, delta)

        return loss.mean()


def loss_lspe(g, p, pos_enc_dim=None, lambda_loss=1.):
    # Loss B: Laplacian Eigenvector Loss --------------------------------------------

    n = g.number_of_nodes()

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt="csr")
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N

    pT = torch.transpose(p, 1, 0)
    loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(g.device)), p))

    # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
    # bg = dgl.unbatch(g)
    # batch_size = len(bg)
    # P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
    P = sp.block_diag([p.detach().cpu()])
    PTP_In = P.T * P - sp.eye(P.shape[1])
    loss_b_2 = torch.tensor(norm(PTP_In, 'fro') ** 2).float().to(g.device)

    # loss_b = (loss_b_1 + lambda_loss * loss_b_2) / (pos_enc_dim * batch_size * n)
    loss_b = (loss_b_1 + lambda_loss * loss_b_2) / (pos_enc_dim * n)

    return loss_b
