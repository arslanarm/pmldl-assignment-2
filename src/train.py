import pandas as pd
import torch
from torch.nn import MSELoss
from torch.nn.functional import mse_loss
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm

from src.dataset import load_dataset
from src.model.gnn import Model
import argparse as ap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = "data/"
model_folder = "models/"


def main():
    users = pd.read_csv(data_folder + "interm/users.csv")
    items = pd.read_csv(data_folder + "interm/items.csv")
    ratings = pd.read_csv(data_folder + "interm/ratings.csv")
    genres = pd.read_csv(data_folder + "raw/ml-100k/u.genre", delimiter="|", names=["name", "index"])

    data = load_dataset(device, users, items, ratings, genres)

    train_data, val_data, test_data = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rates', 'item')],
        rev_edge_types=[('item', 'rev_rates', 'user')],
    )(data)

    model = Model(n_factors=150, hidden_channels=200).to(device)
    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_f = MSELoss()

    def train():
        model.train()
        optimizer.zero_grad()
        pred, mask = model(train_data.x_dict, train_data.edge_index_dict,
                           train_data['user', 'rates', 'item'].edge_label_index)
        target = train_data['user', 'rates', 'item'].edge_label
        loss = loss_f(pred, target[mask].float()).pow(2)
        loss.backward()
        optimizer.step()
        return float(loss)

    def test(data):
        model.eval()
        pred, _ = model(data.x_dict, data.edge_index_dict,
                        data['user', 'rates', 'item'].edge_label_index)
        target = data['user', 'rates', 'item'].edge_label.float()
        rmse = mse_loss(pred, target).sqrt()
        return float(rmse)

    t = tqdm(range(1, 10_000))
    for _ in t:
        loss = train()
        train_rmse = test(train_data)
        val_rmse = test(val_data)
        test_rmse = test(test_data)
        t.set_postfix_str(f"Loss: {loss:.4f}, Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, Test: {test_rmse:.4f}")

    torch.save(model, model_folder + "gnn_model.pt")
    torch.save(data.metadata(), model_folder + "hetero_data_metadata")


if __name__ == "__main__":
    main()
