import torch
import os
from dgl.dataloading import GraphDataLoader
import networkx as nx
from viz_utils.deepdraw_dataloder import Graph_sequence_from_file_dgl, collate
import argparse
import matplotlib.pyplot as plt
import wandb
from dgl_gnn_models.models import GNN_factory
from viz_utils.deepdraw_dataloder import criterion_procrustes
from viz_utils.utils import collate_stress
from viz_utils.aesthetic_losses import Stress_loss, StressCorrected
from viz_utils.utils import Random_from_file_dgl, Rome_from_file_dgl
import viz_utils.utils
import numpy as np
from viz_utils.utils import collate_stress

# importing GD packages
import fa2
from fa2.forceatlas2 import ForceAtlas2
import networkit as nk


class GD_plotter:
    def __init__(self, gd_alg):
        self.gd_alg = gd_alg

        if self.gd_alg == "forceatlas2":
            self.fadrawer = self.GDrawer()

    def GDrawer(self):
        if self.gd_alg == "forceatlas2":
            forceatlas2 = ForceAtlas2(
                # Behavior alternatives
                outboundAttractionDistribution=True,  # Dissuade hubs
                linLogMode=False,  # NOT IMPLEMENTED
                adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                edgeWeightInfluence=1.0,

                # Performance
                jitterTolerance=1.0,  # Tolerance
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                multiThreaded=False,  # NOT IMPLEMENTED

                # Tuning
                scalingRatio=0.5,
                strongGravityMode=True,
                gravity=1.0,

                # Log
                verbose=False)
            return forceatlas2

    def draw(self, g):
        if self.gd_alg == "neato":
            pos = nx.nx_pydot.pydot_layout(g)
            pos_neato = nx.rescale_layout(np.asarray(list(pos.values())))
            return pos_neato
        elif self.gd_alg == "pivotmds":
            G_nk = nk.nxadapter.nx2nk(g)
            az = nk.viz.PivotMDS(G=G_nk, dim=2, numberOfPivots=50)
            return az.run().getCoordinates()
        elif self.gd_alg == "forceatlas2":
            return self.fadrawer.forceatlas2_networkx_layout(g, pos=None, iterations=100)


def forceatlas2_draw(g):
    pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)


def forward_plot(g, model, retina_dims, epoch, show="disk", plot_dir="gnn_plots", name="first", ):
    out = model.draw(g)
    dict_positions = {i: out[i] for i in range(len(out))}
    retina, ax = plt.subplots(1, 1, figsize=(retina_dims / my_dpi, retina_dims / my_dpi),
                              dpi=my_dpi)
    nx.draw(g.to_undirected(), pos=dict_positions, with_labels=False, ax=ax)
    # limits = plt.axis('on')
    # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # plt.title(f"Evolution at Epoch {epoch}")

    if show == "screen":
        plt.show()
    elif show == "disk":
        plt.savefig(os.path.join(plot_dir, f'{name}_{epoch}.png'), bbox_inches='tight')
        # print(dict_positions)
        plt.close()
    elif show == "wandb":
        # Log plot object
        # wandb.log({f"plot_{name}": plt}, step=epoch)
        wandb.log({f"plot_{name}": wandb.Image(plt)}, step=epoch)
        plt.close()
    else:
        raise NotImplementedError


def plot_dataloader_stress(dataset):
    # plotting dataloaders test
    dataloader = GraphDataLoader(dataset, batch_size=1, collate_fn=collate_stress)
    it = iter(dataloader)
    first_graph_test, _, _ = next(it)
    second, _, _ = next(it)
    third, _, _ = next(it)
    f, _, _ = next(it)
    g, _, _ = next(it)
    return first_graph_test, second, third, f, g


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default="rome", choices=["rome", "random", ],
                        help='dataset')
    parser.add_argument('--loss_f', type=str, default="stress",
                        choices=["l1", "mse", "huber", "procrustes", "stress", "stress_corr"],
                        help='loss function')
    parser.add_argument('--encoding', type=str, default="laplacian_eigenvectors",
                        choices=["one_hot", "laplacian_eigenvectors", "binary", "ones", "random", "original"],
                        help='node initial encoding')
    parser.add_argument('--enc_dims', type=int, default=8, metavar='N',
                        help='input encoding size for training (default: 64)')
    parser.add_argument('--GD_alg', type=str, default="neato",
                        choices=["neato", "pivotmds", "forceatlas2"],
                        help='GD package for plotting')
    parser.add_argument('--wandb', type=str, default="true",
                        help='activate wandb')

    args = parser.parse_args()

    args.wandb = args.wandb in {'True', 'true'}

    device = "cpu"
    dataset_folder = os.path.join("data", "rome") if args.dataset == "rome" else os.path.join("data", "random_graph")
    viz_utils.utils.set_seed(args.seed)

    tr_set = "training_list"
    test_set = "test_list"
    vl_set = "validation_list"

    enc_digits = args.enc_dims

    exp_config = {"dataset": dataset_folder,
                  "device": device,
                  "encoding": args.encoding,
                  "enc_dim": enc_digits,
                  "loss_f": args.loss_f,
                  "gd_alg": args.GD_alg}

    # wandb init

    # WANDB_PROJ = "viz_gnn_rome"
    if args.wandb:
        WANDB_PROJ = "viz_gd"
        WANDB_ENTITY = "test"
        wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, config=exp_config)

    plot_dir = "gnn_plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    encoding = args.encoding  # "one_hot", "laplacian_eigenvectors", "binary", "ones", "random", "original"

    # dataset loading
    target_type = "stress"
    if args.dataset == "rome":
        graph_dataset = Rome_from_file_dgl(dataset_folder=dataset_folder, set=tr_set, encoding=encoding,
                                           enc_digits=enc_digits,
                                           target_type=target_type)

        # validation

        graph_dataset_val = Rome_from_file_dgl(dataset_folder=dataset_folder, set=vl_set, encoding=encoding,
                                               enc_digits=enc_digits,
                                               target_type=target_type)

        graph_dataset_test = Rome_from_file_dgl(dataset_folder=dataset_folder, set=test_set, encoding=encoding,
                                                enc_digits=enc_digits,
                                                target_type=target_type)

    elif args.dataset == "random":
        graph_dataset = Random_from_file_dgl(dataset_folder=dataset_folder, set=tr_set, encoding=encoding,
                                             enc_digits=enc_digits,
                                             target_type=target_type)
        graph_dataset_val = Random_from_file_dgl(dataset_folder=dataset_folder, set=vl_set, encoding=encoding,
                                                 enc_digits=enc_digits,
                                                 target_type=target_type)
        graph_dataset_test = Random_from_file_dgl(dataset_folder=dataset_folder, set=test_set, encoding=encoding,
                                                  enc_digits=enc_digits,
                                                  target_type=target_type)

    dataloader = GraphDataLoader(graph_dataset, batch_size=1, collate_fn=collate_stress, num_workers=4)
    print("Training set loaded...")
    dataloader_val = GraphDataLoader(graph_dataset_val, batch_size=1, collate_fn=collate_stress, num_workers=4)
    print("Validation set loaded...")

    dataloader_test = GraphDataLoader(graph_dataset_test, batch_size=1, collate_fn=collate_stress, num_workers=4)
    print("Test set loaded...")
    # plot graph

    g_train_plot = plot_dataloader_stress(graph_dataset)
    g_val_plot = plot_dataloader_stress(graph_dataset_val)
    g_test_plot = plot_dataloader_stress(graph_dataset_test)

    # plot config
    retina_dims = 800
    my_dpi = 96

    if args.loss_f == "l1":
        loss_comp = torch.nn.L1Loss()
    elif args.loss_f == "mse":
        loss_comp = torch.nn.MSELoss()
    elif args.loss_f == "huber":
        loss_comp = torch.nn.SmoothL1Loss()
    elif args.loss_f == "procrustes":
        loss_comp = criterion_procrustes
    elif args.loss_f == "stress":
        loss_c = Stress_loss()
        loss_comp = loss_c.stress_loss
    elif args.loss_f == "stress_corr":
        loss_c = StressCorrected()
        loss_comp = loss_c.stress_loss
    else:
        raise NotImplementedError

    show = "wandb" if args.wandb else "disk"

    gd_drawer = GD_plotter(args.GD_alg)

    for i, el in enumerate(g_train_plot):
        el = el.to_networkx()
        forward_plot(el, gd_drawer, retina_dims, 0, show=show, plot_dir=plot_dir,
                     name=f"{i}_train", )
    for i, el in enumerate(g_val_plot):
        el = el.to_networkx()
        forward_plot(el, gd_drawer, retina_dims, 0, show=show, plot_dir=plot_dir,
                     name=f"{i}_val", )
    for i, el in enumerate(g_test_plot):
        el = el.to_networkx()
        forward_plot(el, gd_drawer, retina_dims, 0, show=show, plot_dir=plot_dir,
                     name=f"{i}_test", )

    counter = 0
    # used for print
    global_train_loss = 0.0
    # used for backprop
    batch_train_loss = torch.zeros(1).to(device)
    for g, short_p, couple_idx in dataloader:
        # forward propagation by using all nodes
        targets = [short_p, couple_idx]

        logits = gd_drawer.draw(g.to_networkx())

        if args.GD_alg == "forceatlas2":
            logits = torch.tensor(list(logits.values()))
        else:
            logits = torch.tensor(logits)
        # compute loss
        loss = loss_comp(logits, targets)
        batch_train_loss += loss
        global_train_loss += loss
        # hack to obtain batched loss computation
        counter += 1
        if (counter % args.batch_size == 0) or (counter == len(dataloader)):

            batch_train_loss = torch.zeros(1).to(device)
            if counter == len(dataloader):
                counter = 0

    epoch_train_loss = global_train_loss / len(dataloader)

    print(f"Epoch 0 \t Global Train Loss: \t  {epoch_train_loss}")

    if args.wandb:
        wandb.log({"epoch": 1, "train_loss": epoch_train_loss}, step=1)

    test_loss = 0.0
    for g_test, short_p_test, couple_idx_test in dataloader_test:

        targets_test = [short_p_test, couple_idx_test]
        logits_test = gd_drawer.draw(g_test.to_networkx())

        if args.GD_alg == "forceatlas2":
            logits_test = torch.tensor(list(logits_test.values()))
        else:
            logits_test = torch.tensor(logits_test)

        # test_loss += loss_comp(logits_test, targets_test)
        test_loss += loss_comp(logits_test, targets_test)

    epoch_test_loss = test_loss / len(dataloader_test)

    print(f"Epoch 0 \t Global Test Loss: \t  {epoch_test_loss}")
    # where the magic happens
    if args.wandb:
        wandb.log({"epoch": 1, "test_loss": epoch_test_loss}, step=1)
