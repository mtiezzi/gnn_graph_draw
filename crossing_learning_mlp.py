import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torch import nn
import argparse
from viz_utils.utils import prepare_device, MLP
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class CrossingDatasetLoader(Dataset):

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data_inp = self.dataframe.iloc[idx, 1:9]  # get coordinates
        target = self.dataframe.iloc[idx, 9]  # get target
        return torch.FloatTensor(data_inp), torch.tensor(target)

    def get_dataset(self) -> [torch.Tensor, torch.Tensor]:
        data = self.dataframe.iloc[:, 1:9].to_numpy()
        target = self.dataframe.iloc[:, 9].to_numpy()
        return torch.tensor(data, dtype=torch.float), torch.tensor(target, dtype=torch.long)


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size if dataloader.batch_size <= size else size
    if batch_size == size:
        dataloader = [dataloader.dataset.get_dataset()]

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # if True:
        #     data = X.to("cpu").numpy()[0]
        #     plt.plot([data[0], data[2]], [data[1], data[3]], color='b', linestyle='-', linewidth=2)
        #     plt.plot([data[4], data[6]], [data[5], data[7]], color='k', linestyle='-', linewidth=2)
        #     plt.show()
        #     print(y[0])
        #     exit()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size if dataloader.batch_size <= size else size
    if batch_size == size:
        dataloader = [dataloader.dataset.get_dataset()]
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= (size / batch_size)
    correct /= size
    return 100 * correct, test_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=100000, metavar='N',
                        help='number of epochs to train (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--hidden_dims', nargs="+", type=int, default=[100, 300, 10],
                        help='list of hidden dimensions (example: 20 15)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_dev', type=int, default=0,
                        help='select specific CUDA device for training')
    parser.add_argument('--n_gpu_use', type=int, default=1,
                        help='select number of CUDA device for training')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='logging training status cadency')
    # parser.add_argument('--tensorboard', action='store_true', default=True,
    #                     help='For logging the model in tensorboard')

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    args = parser.parse_args()
    print(args)

    # device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not use_cuda:
        args.n_gpu_use = 0

    device = prepare_device(n_gpu_use=args.n_gpu_use, gpu_id=args.cuda_dev)

    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    hidden_sizes = args.hidden_dims
    # dataset
    folder_dataset_path = "data"
    training_set = CrossingDatasetLoader(csv_file=os.path.join(folder_dataset_path, "starting_node_training_set.csv"))
    test_set = CrossingDatasetLoader(csv_file=os.path.join(folder_dataset_path, "starting_node_test_set.csv"))
    validation_set = CrossingDatasetLoader(csv_file=os.path.join(folder_dataset_path, "starting_node_validation_set.csv"))

    train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)

    x, y = training_set.get_dataset()
    x, y = x.numpy(), y.numpy()
    x_test, y_test = test_set.get_dataset()
    x_test, y_test = x_test.numpy(), y_test.numpy()

    tree = DecisionTreeClassifier()
    tree.fit(x, y)
    pred_test = tree.predict(x_test)
    print("Decision tree accuracy", accuracy_score(y_test, pred_test))

    model = MLP(input_dim=8, hidden_sizes=hidden_sizes, out_dim=2, activation_function=nn.ReLU(),
                activation_out=nn.ReLU()).to(device)

    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_valid = -1.0
    best_test = -1.
    patience = epochs // 10
    count_patience = 0
    for t in range(epochs):
        print(f"\nEpoch {t + 1}/{epochs}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        valid_acc, valid_loss = validation_loop(validation_dataloader, model, loss_fn, device)
        test_acc, test_loss = validation_loop(test_dataloader, model, loss_fn, device)
        print(f"Validation Error: \n Accuracy: {(valid_acc):>0.1f}%, Avg loss: {valid_loss:>8f}")
        print(f"Test Error: \n Accuracy: {(test_acc):>0.1f}%, Avg loss: {test_loss:>8f}")

        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test = test_acc
            count_patience = 0
            # best model saving
            torch.save(model.state_dict(), 'trained_models/model_weights_m_starting_node.pth')
        else:
            count_patience += 1
            print(f"Validation accuracy not increasing for {count_patience} epochs!")

        if count_patience > patience:
            print("Early stopping!")
            break
    print("Done!")
    print(f"Best val, test accuracy obtained is: {best_valid, best_test}")


if __name__ == '__main__':
    main()
