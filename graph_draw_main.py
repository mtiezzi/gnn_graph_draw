import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import lr_scheduler
from viz_utils.aesthetic_losses import minimize_edge_variance, maximize_node_distances
from viz_utils.utils import MLP, set_seed
import os
from viz_utils.graph_factory import GraphFactory, Graph


class CrossModel:

    def __init__(self, config):
        self.path = os.path.join(config["trained_model"]["path"], config["trained_model"]["name"])
        self.hidden_sizes = config["trained_model"]["hidden_dims"]
        self.device = config["device"]
        self.model = None
        self.load_model(self.hidden_sizes, config["trained_model"]["activation"], self.device)

    def load_model(self, dims, activation, device):
        self.model = MLP(input_dim=8, hidden_sizes=dims, out_dim=2, activation_function=activation,
                         activation_out=activation).to(device).eval()
        # loading weights
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()

    def __call__(self, x):
        return self.model(x)


class GraphDrawer_Model:
    def __init__(self, model: CrossModel, graph: Graph, config, number_pairs):
        self.cross_predictor = model
        self.config = config
        self.input_graph = graph
        self.output_folder = config["graph_drawer"]["output_folder"]
        self.device = config["device"]

        self.node_distance_weight = config["graph_drawer"]["node_distance_weight"]
        self.arc_variance_weight = config["graph_drawer"]["arc_variance_weight"]

        self.edges_params = None
        self.optimizer = None
        self.loss_fn = None
        self.learning_rate = config["graph_drawer"]["lr"]

        self.targets = torch.zeros(number_pairs, dtype=torch.long).to(self.device)
        self.number_pairs = number_pairs
        self.graph_edges = graph.num_nodes
        self.E = graph.get_arcs_list()
        self.number_arcs = len(self.E)
        self.plotter_counter = 0

        # other params
        self.my_dpi = 96
        self.plot_dir = config["graph_drawer"]["output_folder"]
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.build()
        self.optimizer_build()

    def build(self):
        if self.config["graph_drawer"]["pgd_flag"]:
            self.edges_params = torch.FloatTensor(self.graph_edges, 2).uniform_(0, 1).to(self.device)
        else:
            self.edges_params = nn.Parameter(torch.FloatTensor(self.graph_edges, 2).uniform_(0, 1).to(self.device))

    def plot_current_graph(self, show, counter=None):
        # dictionary for visualization
        num_edges = self.input_graph.num_nodes
        dict_positions = {i: self.edges_params[i].detach().cpu().numpy() for i in range(num_edges)}
        retina, ax = plt.subplots(1, 1, figsize=(retina_dims / self.my_dpi, retina_dims / self.my_dpi), dpi=self.my_dpi)

        G = nx.Graph()
        G.add_nodes_from(range(num_edges))
        G.add_edges_from(self.E)
        pos_mult = {el: id * retina_dims for el, id in dict_positions.items()}
        nx.draw(G, pos=pos_mult, with_labels=True, ax=ax)
        limits = plt.axis('off')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if show == "screen":
            # plot the initial state of the graph
            print("Starting random node locations...")
            print(dict_positions)
            plt.show()
            plt.close()
        elif show == "disk":
            plt.savefig(os.path.join(self.plot_dir, f'g_{counter}.png'), bbox_inches='tight')
            print(dict_positions)
            plt.close()
        else:
            raise NotImplementedError

    def optimizer_build(self):
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.ASGD([self.edges_params], lr=self.learning_rate)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def avoid_over_limits(self, nodes):
        loss = torch.linalg.norm(nodes - 0.5, ord=1)
        return loss

    def couple_arcs_predict(self, number_pairs=2):

        random_indices = np.random.randint(low=0, high=self.number_arcs, size=(number_pairs, 2))
        arcs_chosen = self.E[
            random_indices]  # [number_pairs, 2, 2] - for each pair of arc, we have the 4 nodes the two arcs are involving
        first_arc_tensor = arcs_chosen[None, :, 0,
                           :]  # [number_pairs, 2] - select the first arc in each couple => we get its nodes
        second_arc_tensor = arcs_chosen[None, :, 1, :]  # [number_pairs, 2] - select the second arc
        # create the batch of data for the MLP expert
        # create model inputs [batch, ax, ay, bx, by]
        node_coordinates_first_arc = self.edges_params[first_arc_tensor].flatten(1)  # get the node coordinates
        node_coordinates_second_arc = self.edges_params[second_arc_tensor].flatten(1)  # get the node coordinates

        # if node_coordinates_first_arc.max() > 1:
        scaled_node_coordinates_first_arc = node_coordinates_first_arc / node_coordinates_first_arc.max()
        # if node_coordinates_second_arc.max() > 1:
        scaled_node_coordinates_second_arc = node_coordinates_second_arc / node_coordinates_second_arc.max()

        additional_losses = 0.0
        if self.config["graph_drawer"]["avoid_limits"]:
            selected_nodes = np.unique(arcs_chosen)
            additional_losses += self.avoid_over_limits(self.edges_params[selected_nodes])

        model_input = torch.cat((scaled_node_coordinates_first_arc, scaled_node_coordinates_second_arc), dim=1).to(
            self.device)
        out = self.cross_predictor(model_input)
        # if out.max() > 0:
        #     print(out)
        return out, additional_losses

    def moving_arcs(self, iterations=10000, min_edge_variance: bool = True,
                    max_node_distance: bool = True):
        counter = 0
        torch.autograd.set_detect_anomaly(True)
        while counter < iterations:

            # wrap the following into a function -> repeat for more arc couples
            loss = torch.tensor(0.0).to(self.device)
            # crossing_loss = torch.tensor(0.0, requires_grad=True).to(self.device)

            crossing_out, loss_moved_nodes = self.couple_arcs_predict(self.number_pairs)
            # if crossing_flag_out.argmax(1) == 1:  # if crossing
            crossing_loss = self.loss_fn(crossing_out,
                                         self.targets)  # gradient towards not cross

            loss += crossing_loss + loss_moved_nodes
            # aesthetic loss minimization
            variance_loss = torch.tensor(0.)
            if min_edge_variance:
                variance_loss = self.arc_variance_weight * minimize_edge_variance(self.edges_params, self.E)
                loss += variance_loss

            distance_loss = torch.tensor(0.)
            if max_node_distance:
                distance_loss = self.node_distance_weight * maximize_node_distances(self.edges_params)
                loss += distance_loss

            # Backpropagation
            if loss != 0:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.edges_params.data = torch.clamp(self.edges_params, min=0., max=1.0)

            if counter % 100 == 0:
                self.plot_current_graph(show="disk", counter=counter)
                print(f"Iterations {counter}/{iterations}, l: {loss.item():.2f}, cr_loss: {crossing_loss.item():.2f}, "
                      f"var_l: {variance_loss.item():.2f}", f"d_l: {distance_loss.item():.2f}")
            counter += 1

    def move_arcs_pgd_like(self, iterations=10000, min_edge_variance: bool = True, max_node_distance: bool = True):
        counter = 0
        while counter < iterations:

            delta = torch.zeros_like(self.edges_params, requires_grad=True)
            loss = torch.tensor(0.0, requires_grad=True).to(self.device)

            random_indices = np.random.randint(low=0, high=self.number_arcs, size=(self.number_pairs, 2))
            arcs_chosen = self.E[
                random_indices]  # [number_pairs, 2, 2] - for each pair of arc, we have the 4 nodes the two arcs are involving
            first_arc_tensor = arcs_chosen[None, :, 0,
                               :]  # [number_pairs, 2] - select the first arc in each couple => we get its nodes
            second_arc_tensor = arcs_chosen[None, :, 1, :]  # [number_pairs, 2] - select the second arc
            # create the batch of data for the MLP expert
            # create model inputs [batch, ax, ay, bx, by]
            node_coordinates_first_arc = self.edges_params[first_arc_tensor].flatten(1)  # get the node coordinates
            node_coordinates_second_arc = self.edges_params[second_arc_tensor].flatten(
                1)  # get the node coordinates

            num_iter = config["graph_drawer"]["num_iter"]
            alpha = config["graph_drawer"]["alpha"]
            eps = config["graph_drawer"]["eps"]
            for t in range(num_iter):
                delta_fist = delta[first_arc_tensor].flatten(1)
                delta_second = delta[second_arc_tensor].flatten(1)

                model_input = torch.cat((node_coordinates_first_arc + delta_fist, node_coordinates_second_arc +
                                         delta_second), dim=1).to(self.device)

                crossing_out = self.cross_predictor(model_input)
                crossing_loss = self.loss_fn(crossing_out,
                                             self.targets)  # gradient towards not cross

                crossing_loss.backward()
                delta.data = (delta - alpha * delta.grad.detach().sign()).clamp(-eps, eps)
                delta.grad.zero_()
            self.edges_params.data = (self.edges_params.data + delta.data).clamp(0., 1.)

            if counter % 2000 == 0:
                self.plot_current_graph(show="disk", counter=counter)

            counter += 1

    def stress(self, iterations=10000, ):
        counter = 0
        loss_c = nn.MSELoss()
        pdist = torch.nn.PairwiseDistance()
        while counter < iterations:
            # TODO replace this, can be sampled the same node number!!
            number_nodes = input_graph.num_nodes
            batch_size = number_nodes // 2
            random_indices = np.random.choice(np.arange(0, number_nodes),
                                              size=(2, batch_size), replace=False)
            sources, dest = random_indices
            coordinates_sources = self.edges_params[sources]
            coordinates_dest = self.edges_params[dest]

            targets = torch.tensor(self.input_graph.shortest_paths[sources, dest], dtype=torch.float)

            distances = pdist(coordinates_sources, coordinates_dest)
            weight = 1 / (targets + 1e-7)
            # loss = weight * loss_c(distances, targets)
            loss = weight * (distances - targets) ** 2

            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # loss = nn.MSELoss(coordinates[:, 0] - )
            if counter % 50 == 0:
                self.scheduler.step()
                self.plot_current_graph(show="disk", counter=counter)
                print(f"Iterations {counter}/{iterations}, l: {loss.item():.2f}")
            counter += 1

            if counter % 100 == 0:
                self.scheduler.step()


if __name__ == '__main__':

    device = "cpu" if torch.cuda.is_available() else "cpu"
    set_seed(3010701)
    # input graph selection
    retina_dims = 800

    # optimization parameters
    learning_rate = 0.2
    activation = nn.ReLU()
    hidden_sizes = [100, 300, 10]
    # path = "model_weights_m.pth"
    path = "model_weights_m_starting_node.pth"
    # path = "model_weights_thick_starting_node.pth"


    config = {
        "device": device,
        "retina_dims": retina_dims,
        "trained_model": {
            "path": "trained_models",
            "name": path,
            "hidden_dims": hidden_sizes,
            "activation": activation
        },
        "graph_drawer": {
            "lr": learning_rate,
            "output_folder": "plots_gif",
            "node_distance_weight": 0.,
            "arc_variance_weight": 1.0,
            "pgd_flag": False,
            "num_iter": 5,
            "alpha": 1e-3,
            "eps": 0.05,
            "avoid_limits": False,
            "stress": True,
        },
        "input_graph": {
            "name": "cycle",  # "simple", "karate", "cycle", "cube",  "dodecahedral", "star", "grid", "random"
            "nodes": 20, "p": 0.1,
            "bidirectional": True,
            "graph_id": 7897987
        }

    }

    cross_model = CrossModel(config)
    input_graph = GraphFactory.create_graph(config)
    try:
        input_graph()
    except ValueError:
        raise

    drawer = GraphDrawer_Model(model=cross_model, graph=input_graph, config=config, number_pairs=500)
    drawer.plot_current_graph(show="screen")
    import time

    start_time = time.time()
    # drawer.moving_arcs(iterations=2000)
    drawer.stress(iterations=5000)
    end_time = time.time() - start_time
    print(f"Execution time: {end_time:.4f}")
    # drawer.move_arcs_pgd_like()
