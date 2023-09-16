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
from viz_utils.utils import Random_from_file_dgl, Rome_from_file_dgl, Big_from_file_dgl
import viz_utils.utils
from viz_utils.utils import collate_stress, positional_encoding
from copy import deepcopy
from tqdm import tqdm
from time import sleep
import scipy as sp
import scipy.io  # for mmread() and mmwrite()
import io  # Use BytesIO as a stand-in for a Python file object
from scipy.io import mmread
import dgl


def forward_plot(g, model, retina_dims, epoch, show="disk", plot_dir="gnn_plots", name="first", ):
    out = model(g, g.ndata["feat"])
    dict_positions = {i: out[i].detach().cpu().numpy() for i in range(len(out))}
    retina, ax = plt.subplots(1, 1, figsize=(retina_dims / my_dpi, retina_dims / my_dpi),
                              dpi=my_dpi)
    graph = g.cpu().to_networkx()
    H = nx.Graph(graph)
    nx.draw(H, pos=dict_positions, ax=ax, edge_color='black', width=1, linewidths=1, node_size=10, node_color='blue',
            alpha=0.9)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--drop', type=float, default=0., metavar='LR',
                        help='dropout (default: 0.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--dataset', type=str, default="qh882",
                        help='dataset')
    parser.add_argument('--encoding', type=str, default="laplacian_eigenvectors",
                        choices=["one_hot", "laplacian_eigenvectors", "binary", "ones", "random", "original"],
                        help='node initial encoding')
    parser.add_argument('--enc_dims', type=int, default=8, metavar='N',
                        help='input encoding size for training (default: 64)')
    parser.add_argument('--loss_f', type=str, default="stress_corr",
                        choices=["l1", "mse", "huber", "procrustes", "stress", "stress_corr"],
                        help='loss function')
    parser.add_argument('--layers', type=int, default=5,
                        help='number of gnn layers')
    parser.add_argument('--activation', type=str, default="relu",
                        choices=["relu", "sigm", "tanh"],
                        help='activation function')
    parser.add_argument('--gnn_model', type=str, default="gat",
                        choices=["gcn", "gat", "gin"],
                        help='activation function')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='hidden size')
    parser.add_argument('--target_type', type=str, default="stress",
                        choices=["stress"],
                        help='hidden size')
    parser.add_argument('--wandb', type=str, default="false",
                        help='activate wandb')

    args = parser.parse_args()
    args.wandb = args.wandb in {'True', 'true'}

    device = "cuda"
    dataset_folder = os.path.join("data", args.dataset)
    viz_utils.utils.set_seed(args.seed)

    tr_set = "training_list"
    test_set = "test_list"
    vl_set = "validation_list"

    enc_digits = args.enc_dims

    exp_config = {"dataset": dataset_folder,
                  "device": device,
                  "lr": args.lr,
                  "batch_size": args.batch_size,
                  "encoding": args.encoding,
                  "enc_dim": enc_digits,
                  "loss_f": args.loss_f,
                  "activation": args.activation,
                  "gnn_model": args.gnn_model,
                  "layers": args.layers,
                  "hidden_size": args.hidden_size,
                  "target_type": args.target_type,
                  "dropout": args.drop}

    show = "wandb" if args.wandb else "disk"
    # wandb init
    if args.wandb:
        # WANDB_PROJ = "viz_gnn_rome"
        WANDB_PROJ = "viz_reproduce"
        WANDB_ENTITY = "mtiezzi"
        wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, config=exp_config)

    model_name = "start_model.pth"
    plot_dir = f"gnn_plots_saved_{args.dataset}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    save_dir = "saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    encoding = args.encoding  # "one_hot", "laplacian_eigenvectors", "binary", "ones", "random", "original"

    # dataset loading
    target_type = args.target_type  # , "circular", "spring", "spectral", "stress"

    file = os.path.join(dataset_folder, f"{args.dataset}.mtx")
    G = nx.from_scipy_sparse_matrix(sp.io.mmread(file))
    g1 = dgl.from_networkx(G)

    g1.set_n_initializer(dgl.init.zero_initializer)
    g1.set_e_initializer(dgl.init.zero_initializer)

    g1.ndata["feat"] = positional_encoding(g1, pos_enc_dim=args.enc_dims)

    final_state_dim = 2  # directly regression

    # model definition 
    model_type = args.gnn_model

    model = GNN_factory.createModel(name=args.gnn_model, config=exp_config).to(device)

    # plot config
    retina_dims = 800
    my_dpi = 96

    # optimizer and loss    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    best_acc_val = None
    patience = 30
    patience_counter = 0

    # load model
    loaded = torch.load(os.path.join(save_dir, model_name))
    # for i in range(1, 5):
    #     loaded.pop(f'gat_layers.{i}.res_fc.weight')
    model.load_state_dict(loaded)
    model.eval()

    forward_plot(g1, model, retina_dims, 0, show=show, plot_dir=plot_dir,
                 name="plot", )

    plt.close('all')
    print("All plots completed!")
