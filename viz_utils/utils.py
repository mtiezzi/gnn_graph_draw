import random
import dgl
import torch
import torch.nn as nn
import typing
import numpy as np
import networkx as nx
# from deepdraw import binary
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from scipy import sparse as sp
from scipy.sparse.linalg import norm

import pathlib
import platform

plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
elif plt == 'Windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_device(n_gpu_use, gpu_id=None):
    """
    setup specific GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
              "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:{}'.format(gpu_id) if n_gpu_use > 0 else 'cpu')
    print("Executing on device: ", device)
    return device


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes: typing.Iterable[int], out_dim, activation_function=nn.Sigmoid(),
                 activation_out=None):
        super(MLP, self).__init__()

        i_h_sizes = [input_dim] + hidden_sizes  # add input dim to the iterable
        self.mlp = nn.Sequential()
        for idx in range(len(i_h_sizes) - 1):
            self.mlp.add_module(f"layer_{idx}",
                                nn.Linear(in_features=i_h_sizes[idx], out_features=i_h_sizes[idx + 1]))
            self.mlp.add_module(f"act_{idx}", activation_function)
        self.mlp.add_module("out_layer", nn.Linear(i_h_sizes[-1], out_dim))
        if activation_out is not None:
            self.mlp.add_module("out_layer_activation", activation_out)

    def init(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x).squeeze()


def build_karate_club_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
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
    return u


# graph plotting method
def plot_graph_pos(E, edges, pos, axis, retina_dims):
    G = nx.Graph()
    G.add_nodes_from(range(edges))
    G.add_edges_from(E)
    pos_mult = {el: id * retina_dims for el, id in pos.items()}
    nx.draw(G, pos=pos_mult, with_labels=True, ax=axis)


def plot_nxgraph_pos(G, pos, axis, retina_dims):
    pos_mult = {el: id * retina_dims for el, id in pos.items()}
    nx.draw(G, pos=pos_mult, with_labels=True, ax=axis)


def laplacian_eigenvectors(g, pos_enc_dim):
    import dgl
    from scipy import sparse as sp

    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    n = g.number_of_nodes()
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N
    # Eigenvectors
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()


def positional_encoding(g, pos_enc_dim):
    import dgl
    from scipy import sparse as sp

    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    n = g.number_of_nodes()
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N
    # Eigenvectors
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()


def random_walk_positional_encoding(g, pos_enc_dim):
    """
        Initializing positional encoding with RWPE
    """

    # Geometric diffusion features with Random Walk
    A = g.adjacency_matrix(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
    RW = A * Dinv
    M = RW

    # Iterate
    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE, dim=-1)

    return PE


def get_encoding(g: dgl.graph, encoding, enc_digits):
    nodenum = g.number_of_nodes()
    if encoding == "one_hot":
        features = torch.eye(n=enc_digits)
        return features[:nodenum]
    # elif encoding == "binary":
    #     return binary(torch.arange(nodenum), enc_digits)
    elif encoding == "laplacian_eigenvectors":
        return positional_encoding(g, pos_enc_dim=enc_digits)
    elif encoding == "ones":
        return torch.ones((nodenum, enc_digits))
    elif encoding == "random":
        return torch.rand(size=(nodenum, enc_digits))
    elif encoding == "void":
        return torch.zeros((nodenum, 1), dtype=torch.float32)
    else:
        raise NotImplementedError


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


def collate_stress(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, sh_path, coup_idx = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(sh_path), torch.cat(coup_idx),


########## Rome_sequence_from_file
class Rome_from_file_dgl(Dataset):
    def __init__(self, dataset_folder, set, encoding="binary", enc_digits=None, target_type=None):
        self.dataset_folder = dataset_folder
        with open(os.path.join(dataset_folder, set), 'rb') as f:
            self.graphlist = pkl.load(f)
        self.encoding = encoding
        self.enc_digits = enc_digits
        self.target_type = target_type

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, idx):
        pathname = self.graphlist[idx]
        # print(f"Graph: {pathname}")
        g = nx.read_graphml(path=os.path.join(self.dataset_folder, pathname))
        nodenum = g.number_of_nodes()
        if nodenum < 10:
            print(pathname)
            exit()
        # convert the labels to integer - otherwise there are errors
        map_dict = {f"n{i}": i for i in range(nodenum)}
        g = nx.relabel_nodes(g, mapping=map_dict)
        g1 = dgl.from_networkx(g)

        g1.set_n_initializer(dgl.init.zero_initializer)
        g1.set_e_initializer(dgl.init.zero_initializer)

        if self.encoding == "one_hot":
            features = torch.eye(n=self.enc_digits)
            g1.ndata["feat"] = features[:nodenum]

        elif self.encoding == "laplacian_eigenvectors":
            g1.ndata["feat"] = positional_encoding(g1, pos_enc_dim=self.enc_digits)
        elif self.encoding == "ones":
            g1.ndata["feat"] = torch.ones((nodenum, self.enc_digits))
        elif self.encoding == "random":
            g1.ndata["feat"] = torch.rand(size=(nodenum, self.enc_digits))
        elif self.encoding == "void":
            g1.ndata["feat"] = torch.zeros((nodenum, 1), dtype=torch.float32)
        elif self.encoding == "rwpe":
            g1.ndata["feat"] = positional_encoding(g1, pos_enc_dim=self.enc_digits)
            g1.ndata["rwe"] = random_walk_positional_encoding(g1, pos_enc_dim=self.enc_digits)
        else:
            raise NotImplementedError
        # g1.ndata["feat"] = torch.randint(low=0, high=,size=(nodenum, 10))

        # if self.stress_flag:  # TODO what happens with batches?!?
        #     H = g1.to_networkx().to_directed()
        #     shortest_p, couples_indices = shortest_path_computation(H)
        #     return g1, torch.tensor(object1["pos"][:nodenum], dtype=torch.float), shortest_p, couples_indices
        layout_seed = 1234567
        if self.target_type == "circular":
            pos = nx.circular_layout(g)
        elif self.target_type == "spring":
            pos = nx.spring_layout(g, seed=layout_seed)
        elif self.target_type == "kamada":
            pos = nx.kamada_kawai_layout(g)
        elif self.target_type == "spectral":
            pos = nx.spectral_layout(g)
        elif self.target_type == "graphviz":
            pos = nx.nx_agraph.graphviz_layout(g)
        elif self.target_type == "stress":
            shortest_p, couples_indices = shortest_path_computation(g)
            return g1, shortest_p, couples_indices
        elif self.target_type == "void":
            pos = np.zeros((nodenum, 1))
            return g1, torch.zeros((nodenum, 1), dtype=torch.float32)
        else:
            raise NotImplementedError
        # convert into numpy
        pos = np.array(list(pos.values()))
        # # add label to edges
        # g1.edata['edge_label'] = torch.ones(all_edge_num, 1)
        # g1.edata['edge_label'][0:real_edge_num] = 0
        # g2.edata['edge_label'] = torch.ones(all_edge_num, 1)
        # g2.edata['edge_label'][0:real_edge_num] = 0
        #
        # object1["g1"] = g1

        # return g1, object1["pos"][:nodenum]
        # g1.labels = torch.tensor(object1["pos"])
        return g1, torch.tensor(pos, dtype=torch.float32)


class Random_from_file_dgl(Dataset):
    def __init__(self, dataset_folder, set, encoding="binary", enc_digits=None, target_type=None):
        self.dataset_folder = dataset_folder
        with open(os.path.join(dataset_folder, set), 'rb') as f:
            self.graphlist = pkl.load(f)
        self.encoding = encoding
        self.enc_digits = enc_digits
        self.target_type = target_type

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, idx):
        pathname = self.graphlist[idx]
        # print(f"Graph: {pathname}")
        g = nx.read_gpickle(path=os.path.join(self.dataset_folder, pathname))
        nodenum = g.number_of_nodes()
        if nodenum < 10:
            print(pathname)
            exit()
        # convert the labels to integer - otherwise there are errors
        # maybe not necessary
        # map_dict = {f"n{i}": i for i in range(nodenum)}
        # g = nx.relabel_nodes(g, mapping=map_dict)
        g1 = dgl.from_networkx(g)

        g1.set_n_initializer(dgl.init.zero_initializer)
        g1.set_e_initializer(dgl.init.zero_initializer)

        if self.encoding == "one_hot":
            features = torch.eye(n=self.enc_digits)
            g1.ndata["feat"] = features[:nodenum]

        elif self.encoding == "laplacian_eigenvectors":
            g1.ndata["feat"] = positional_encoding(g1, pos_enc_dim=self.enc_digits)
        elif self.encoding == "ones":
            g1.ndata["feat"] = torch.ones((nodenum, self.enc_digits))
        elif self.encoding == "random":
            g1.ndata["feat"] = torch.rand(size=(nodenum, self.enc_digits))
        elif self.encoding == "void":
            g1.ndata["feat"] = torch.zeros((nodenum, 1), dtype=torch.float32)
        elif self.encoding == "rwpe":
            g1.ndata["feat"] = positional_encoding(g1, pos_enc_dim=self.enc_digits)
            g1.ndata["rwe"] = random_walk_positional_encoding(g1, pos_enc_dim=self.enc_digits)
        else:
            raise NotImplementedError
        # g1.ndata["feat"] = torch.randint(low=0, high=,size=(nodenum, 10))

        # if self.stress_flag:  # TODO what happens with batches?!?
        #     H = g1.to_networkx().to_directed()
        #     shortest_p, couples_indices = shortest_path_computation(H)
        #     return g1, torch.tensor(object1["pos"][:nodenum], dtype=torch.float), shortest_p, couples_indices

        if self.target_type == "circular":
            pos = nx.circular_layout(g)
        elif self.target_type == "kamada":
            pos = nx.kamada_kawai_layout(g)
        elif self.target_type == "spectral":
            pos = nx.spectral_layout(g)
        elif self.target_type == "stress":
            shortest_p, couples_indices = shortest_path_computation(g)
            return g1, shortest_p, couples_indices
        else:
            raise NotImplementedError
        # convert into numpy
        pos = np.array(list(pos.values()))
        # # add label to edges
        # g1.edata['edge_label'] = torch.ones(all_edge_num, 1)
        # g1.edata['edge_label'][0:real_edge_num] = 0
        # g2.edata['edge_label'] = torch.ones(all_edge_num, 1)
        # g2.edata['edge_label'][0:real_edge_num] = 0
        #
        # object1["g1"] = g1

        # return g1, object1["pos"][:nodenum]
        # g1.labels = torch.tensor(object1["pos"])
        return g1, torch.tensor(pos, dtype=torch.float32)


class Big_from_file_dgl(Dataset):
    def __init__(self, dataset_folder, set, encoding="binary", enc_digits=None, target_type=None, index=None,
                 limits: list = None):
        self.dataset_folder = dataset_folder
        with open(os.path.join(dataset_folder, set), 'rb') as f:
            self.graphlist = pkl.load(f)
            if index is not None:
                self.graphlist = self.graphlist[index - 1:index]
            elif index is None and limits is not None:
                self.graphlist = self.graphlist[limits[0]:limits[1]]
        self.encoding = encoding
        self.enc_digits = enc_digits
        self.target_type = target_type

    def __len__(self):
        return len(self.graphlist)

    def __getitem__(self, idx):
        pathname = self.graphlist[idx]
        g = nx.read_gpickle(path=pathname)
        nodenum = g.number_of_nodes()

        g1 = dgl.from_networkx(g)

        g1.set_n_initializer(dgl.init.zero_initializer)
        g1.set_e_initializer(dgl.init.zero_initializer)

        if self.encoding == "one_hot":
            features = torch.eye(n=self.enc_digits)
            g1.ndata["feat"] = features[:nodenum]

        elif self.encoding == "laplacian_eigenvectors":
            g1.ndata["feat"] = positional_encoding(g1, pos_enc_dim=self.enc_digits)
        elif self.encoding == "ones":
            g1.ndata["feat"] = torch.ones((nodenum, self.enc_digits))
        elif self.encoding == "random":
            g1.ndata["feat"] = torch.rand(size=(nodenum, self.enc_digits))
        elif self.encoding == "void":
            g1.ndata["feat"] = torch.zeros((nodenum, 1), dtype=torch.float32)
        elif self.encoding == "rwpe":
            g1.ndata["feat"] = positional_encoding(g1, pos_enc_dim=self.enc_digits)
            g1.ndata["rwe"] = random_walk_positional_encoding(g1, pos_enc_dim=self.enc_digits)
        else:
            raise NotImplementedError

        if self.target_type == "neato":
            pos = nx.get_node_attributes(g, "neato")[0]  # TODO
        elif self.target_type == "sfdp":
            pos = nx.get_node_attributes(g, "sfdp")[0]
        elif self.target_type == "stress":
            shortest_p, couples_indices = shortest_path_computation(g)
            return g1, shortest_p, couples_indices
        else:
            raise NotImplementedError
        # convert into numpy
        # pos = np.array(list(pos.values()))
        return g1, torch.tensor(pos, dtype=torch.float32)


def plot_tsne(converged_states, show, plot_dir=None, epoch=None):
    tsne_states = TSNE(n_components=2, early_exaggeration=1.0).fit_transform(converged_states)
    plt.scatter(tsne_states[:, 0], tsne_states[:, 1])
    if show == "screen":
        plt.show()
    elif show == "disk":
        plt.savefig(os.path.join(plot_dir, f'tsne_{epoch}.png'), bbox_inches='tight')
        # print(dict_positions)
        plt.close()


def plot_pca(converged_states, show, plot_dir=None, epoch=None):
    pca = PCA(n_components=2).fit_transform(converged_states)
    fig, ax = plt.subplots()
    idx = [i for i in range(converged_states.shape[0])]
    plt.scatter(pca[:, 0], pca[:, 1])
    for i in range(converged_states.shape[0]):
        ax.annotate(str(i), (pca[i, 0], pca[i, 1]))
    if show == "screen":
        plt.show()
    elif show == "disk":

        plt.savefig(os.path.join(plot_dir, f'pca_{epoch}.png'), bbox_inches='tight')
        # print(dict_positions)
        plt.close()


def batch_loss_computation(g, loss_comp, logits, targets):
    batch_nodes = g.batch_num_nodes()
    # logits = model(first_graph, first_graph.ndata["feat"])
    batch_loss = 0
    for i in range(g.batch_size):
        num_nodes = batch_nodes[i]
        num_nodes_pred = 0 if i == 0 else torch.sum(batch_nodes[0:i])
        batch_loss = batch_loss + loss_comp(logits[num_nodes_pred:num_nodes_pred + num_nodes],
                                            targets[num_nodes_pred:num_nodes_pred + num_nodes])

    return batch_loss


class CrossModel:

    def __init__(self, config):
        self.path = os.path.join(config["trained_model"]["path"], config["trained_model"]["name"])
        self.hidden_sizes = config["trained_model"]["hidden_dims"]
        self.device = config["device"]
        self.model = None
        self.load_model(self.hidden_sizes, config["trained_model"]["activation"], self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def load_model(self, dims, activation, device):
        self.model = MLP(input_dim=8, hidden_sizes=dims, out_dim=2, activation_function=activation,
                         activation_out=activation).to(device)
        # loading weights
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()

    def __call__(self, x):
        return self.model(x)

    def cross_loss(self, pred, edges, number_pairs=50):
        # random_indices = np.random.randint(low=0, high=self.number_arcs, size=(number_pairs, 2))
        edges_tensor = torch.hstack((torch.unsqueeze(edges[0], 1), torch.unsqueeze(edges[1], 1)))

        # all edges combinations
        range_graph = torch.arange(edges_tensor.shape[0])
        couples_indices = torch.combinations(range_graph, r=2)
        targets = torch.zeros(couples_indices.shape[0], dtype=torch.long).to(self.device)

        # random edges  combinations
        # random_indices = np.random.randint(low=0, high=len(edges), size=(number_pairs, 2)) # TODO substitute without repetiton
        # arcs_chosen = edges_tensor[random_indices]  # [number_pairs, 2, 2] - for each pair of arc, we have the 4 nodes the two arcs are involving
        # targets = torch.zeros(number_pairs, dtype=torch.long).to(self.device)

        arcs_chosen = edges_tensor[
            couples_indices]  # [number_pairs, 2, 2] - for each pair of arc, we have the 4 nodes the two arcs are involving
        first_arc_tensor = arcs_chosen[:, 0,
                           :]  # [number_pairs, 2] - select the first arc in each couple => we get its nodes
        second_arc_tensor = arcs_chosen[:, 1, :]  # [number_pairs, 2] - select the second arc
        # create the batch of data for the MLP expert
        # create model inputs [batch, ax, ay, bx, by]
        node_coordinates_first_arc = pred[first_arc_tensor].flatten(1)  # get the node coordinates
        node_coordinates_second_arc = pred[second_arc_tensor].flatten(1)  # get the node coordinates

        # if node_coordinates_first_arc.max() > 1:
        max_scale = torch.max(torch.cat([node_coordinates_first_arc, node_coordinates_second_arc]))
        scaled_node_coordinates_first_arc = node_coordinates_first_arc / max_scale
        # if node_coordinates_second_arc.max() > 1:
        scaled_node_coordinates_second_arc = node_coordinates_second_arc / max_scale

        model_input = torch.cat((scaled_node_coordinates_first_arc, scaled_node_coordinates_second_arc), dim=1).to(
            self.device)
        out = self.model(model_input)

        crossing_loss = self.loss_fn(out, targets)  # gradient towards not cross

        return crossing_loss
