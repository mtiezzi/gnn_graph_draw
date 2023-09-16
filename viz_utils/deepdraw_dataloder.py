import torch
import os
import pickle as pkl
import dgl
from torch.utils.data import Dataset, DataLoader
from dgl.dataloading import GraphDataLoader
import networkx as nx

import viz_utils.utils
from viz_utils.utils import plot_nxgraph_pos
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg
from viz_utils.aesthetic_losses import shortest_path_computation


# dataset_file = "deepdrawing-dataset/grid_v1_test_dataset_folder_preprocess"
# graphlist = os.listdir(dataset_file)
#
#
# pathname = graphlist[0]
# object1 = {}
# with open(os.path.join(dataset_file, pathname), "rb") as f:
#     object1 = pkl.load(f)
#
# nodenum = object1["len"]
# graph = object1["graph"]
# print("ciaone")
#
# adj = object1["adj"]
# pos= object1["pos"]


def binary(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


########## Graph_sequence_from_file
class Graph_sequence_from_file_dgl(Dataset):
    def __init__(self, dataset_file, encoding="binary", enc_digits=None, stress=None):
        self.dataset_file = dataset_file
        self.graphlist = os.listdir(self.dataset_file)
        self.encoding = encoding
        self.enc_digits = enc_digits
        self.stress_flag = stress

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, idx):
        pathname = self.graphlist[idx]
        object1 = {}
        with open(os.path.join(self.dataset_file, pathname), "rb") as f:
            object1 = pkl.load(f)
        nodenum = object1["len"]
        graph = object1["graph"]
        g1 = dgl.DGLGraph()

        # g2 = dgl.DGLGraph()
        # add nodes into the graph; nodes are labeled from 0 to (nodenum - 1)
        g1.add_nodes(nodenum)
        # g2.add_nodes(nodenum)
        # real edges
        for i in range(nodenum):
            for j in range(len(graph[i])):
                tgt = graph[i][j]
                src = i
                # if src < tgt:
                g1.add_edges(src, tgt)
                # if src > tgt:
                #     g2.add_edges(src, tgt)
        real_edge_num = g1.number_of_edges()

        # # fake edges due to BFS order
        # for i in range(nodenum - 1):
        #     g1.add_edges(i, i + 1)
        #     g2.add_edges(nodenum - 1 - i, nodenum - 2 - i)
        # all_edge_num = g1.number_of_edges()

        # initialize all the node and edge features
        g1.set_n_initializer(dgl.init.zero_initializer)
        # g2.set_n_initializer(dgl.init.zero_initializer)
        g1.set_e_initializer(dgl.init.zero_initializer)
        # g2.set_e_initializer(dgl.init.zero_initializer)

        # g1.ndata["feat"] = torch.eye(n=nodenum)
        if self.encoding == "one_hot":
            features = torch.eye(n=self.enc_digits)
            g1.ndata["feat"] = features[:nodenum]
        elif self.encoding == "binary":
            g1.ndata["feat"] = binary(torch.arange(nodenum), self.enc_digits)
        elif self.encoding == "laplacian_eigenvectors":
            g1.ndata["feat"] = viz_utils.utils.positional_encoding(g1, pos_enc_dim=self.enc_digits)
        elif self.encoding == "ones":
            g1.ndata["feat"] = torch.ones((nodenum, self.enc_digits))
        elif self.encoding == "random":
            g1.ndata["feat"] = torch.rand(size=(nodenum, self.enc_digits))
        elif self.encoding == "original":
            g1.ndata["feat"] = torch.as_tensor(object1["x"][:nodenum], dtype=torch.float32)
        elif self.encoding == "void":
            g1.ndata["feat"] = torch.zeros((nodenum, 1), dtype=torch.float32)
        else:
            raise NotImplementedError
        # g1.ndata["feat"] = torch.randint(low=0, high=,size=(nodenum, 10))

        if self.stress_flag:  # TODO what happens with batches?!?
            H = g1.to_networkx().to_directed()
            shortest_p, couples_indices = shortest_path_computation(H)
            return g1, torch.tensor(object1["pos"][:nodenum], dtype=torch.float), shortest_p, couples_indices

        # # add label to edges
        # g1.edata['edge_label'] = torch.ones(all_edge_num, 1)
        # g1.edata['edge_label'][0:real_edge_num] = 0
        # g2.edata['edge_label'] = torch.ones(all_edge_num, 1)
        # g2.edata['edge_label'][0:real_edge_num] = 0
        #
        # object1["g1"] = g1
        g1.labels = torch.tensor(object1["pos"][:nodenum])
        # return g1, object1["pos"][:nodenum]
        # g1.labels = torch.tensor(object1["pos"])
        return g1, torch.tensor(object1["pos"][:nodenum], dtype=torch.float)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(labels)


def collate_sp(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels, sh_path, coup_idx = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(labels), torch.cat(sh_path), torch.cat(coup_idx),

def collate_stress_deepdraw(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, _, sh_path, coup_idx = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(sh_path), torch.cat(coup_idx),

if __name__ == '__main__':
    dataset_file = "../deepdrawing-dataset/grid_v1_test_dataset_folder_preprocess"

    graph_dataset = Graph_sequence_from_file_dgl(dataset_file=dataset_file)
    dataloader = GraphDataLoader(graph_dataset, batch_size=1, collate_fn=collate)

    it = iter(dataloader)
    first_graph, targets = next(it)
    num_nodes = first_graph.num_nodes()
    real_targ = targets[0][:num_nodes]
    graph = first_graph.to_networkx().to_directed()
    H = nx.Graph(graph)

    # draw this graph with positions
    my_dpi = 96
    retina_dims = 800
    retina, ax = plt.subplots(1, 1, figsize=(retina_dims / my_dpi, retina_dims / my_dpi), dpi=my_dpi)
    dict_positions = {i: real_targ[i].detach().numpy() for i in range(len(real_targ))}
    plot_nxgraph_pos(H, dict_positions, axis=ax, retina_dims=retina_dims)
    plt.show()


def orthogonal_procrustes_torch(A, B):
    # Be clever with transposes, with the intention to save memory.
    A_device = A.device
    B_copy = B.clone().to(A_device)

    input = torch.transpose(torch.matmul(torch.transpose(B_copy, 0, 1), A), 0, 1)
    u, w, vt = torch.svd(input)
    # u, w, vt = torch.svd(torch.transpose(torch.matmul(torch.transpose(B,0,1),A),0,1))
    R = torch.matmul(u, torch.transpose(vt, 0, 1))
    scale = torch.sum(w)
    return R, scale


def criterion_procrustes(data1, data2):
    device = data1.device
    mtx1 = data1
    mtx2 = data2.clone().to(device)

    # translate all the data to the origin
    mtx3 = mtx1 - torch.mean(mtx1, 0)
    mtx4 = mtx2 - torch.mean(mtx2, 0)

    norm1 = torch.norm(mtx3)
    norm2 = torch.norm(mtx4)

    if norm1 == 0:
        norm1 = 1e-16
    if norm2 == 0:
        norm2 = 1e-16
    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx3 = mtx3 / norm1
    mtx4 = mtx4 / norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes_torch(mtx3, mtx4)
    mtx4 = torch.matmul(mtx4, torch.transpose(R, 0, 1)) * s

    # measure the dissimilarity between the two datasets
    disparity = torch.sum((mtx3 - mtx4) ** 2)

    return disparity


def manual_procrustes(data1, data2):
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)
    # print("manual norm")
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    # print(norm1,norm2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity


def orthogonal_procrustes(A, B, check_finite=True):
    """
    Compute the matrix solution of the orthogonal Procrustes problem.
    Given matrices A and B of equal shape, find an orthogonal matrix R
    that most closely maps A to B [1]_.
    Note that unlike higher level Procrustes analyses of spatial data,
    this function only uses orthogonal transformations like rotations
    and reflections, and it does not use scaling or translation.
    Parameters
    """
    if check_finite:
        A = np.asarray_chkfinite(A)
        B = np.asarray_chkfinite(B)
    else:
        A = np.asanyarray(A)
        B = np.asanyarray(B)
    if A.ndim != 2:
        raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
    if A.shape != B.shape:
        raise ValueError('the shapes of A and B differ (%s vs %s)' % (
            A.shape, B.shape))
    # Be clever with transposes, with the intention to save memory.
    input = B.T.dot(A).T
    u, w, vt = linalg.svd(input)
    R = u.dot(vt)
    scale = w.sum()
    return R, scale
