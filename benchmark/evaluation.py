import numpy as np
import pandas as pd
import torch
from torch.nn import Dropout
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero, Linear, SAGEConv
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import dropout_edge

from benchmark.metrics import metrics
from data_preprocess import preprocess_data
from sentence_transformers import SentenceTransformer

device = "cuda:0"

metadata = torch.load("../models/hetero_data_metadata")


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # these convolutions have been replicated to match the number of edge types\
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, n_factors, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * n_factors, hidden_channels)
        self.dropout1 = Dropout(p=0.5)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.dropout2 = Dropout(p=0.5)
        self.lin3 = Linear(hidden_channels, hidden_channels)
        self.dropout3 = Dropout(p=0.25)
        self.lin4 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        # concat user and movie embeddings
        z = torch.cat([z_dict['user'][row], z_dict['item'][col]], dim=-1)
        # concatenated embeddings passed to linear layer
        z = self.lin1(z).relu()
        z = self.dropout1(z)
        z = self.lin2(z).relu()
        z = self.dropout2(z)
        z = self.lin3(z).relu()
        z = self.dropout3(z)
        z = self.lin4(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, n_factors, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(n_factors)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.decoder = EdgeDecoder(n_factors, hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # z_dict contains dictionary of movie and user embeddings returned from GraphSage
        edge_label_index, mask = dropout_edge(edge_label_index, p=0.02, training=self.training)
        z_dict = self.encoder(x_dict, edge_index_dict)
        output = self.decoder(z_dict, edge_label_index)
        output = torch.sigmoid(output)
        output = output * 4 + 1
        return output, mask


model = torch.load("../models/gnn_model.pt")
model.eval()

preprocess_data()

data_folder = "data/"
users = pd.read_csv(data_folder + "users.csv")
items = pd.read_csv(data_folder + "items.csv")
ratings = pd.read_csv(data_folder + "ratings-base.csv")
test_ratings = pd.read_csv(data_folder + "ratings-test.csv")
genres = pd.read_csv("../data/raw/ml-100k/u.genre", delimiter="|", names=["name","index"])


def create_torch_edges(ratings):
    src = ratings["user_id"] - 1
    dst = ratings["item_id"] - 1
    attrs = ratings["rating"]

    edge_index = torch.tensor([src, dst], dtype=torch.int64)
    edge_attr = torch.tensor(attrs)

    return edge_index, edge_attr


edge_index, edge_attr = create_torch_edges(ratings)


def SequenceEncoder(movie_titles, model_name=None):
    model = SentenceTransformer(model_name, device=device)
    title_embeddings = model.encode(movie_titles, show_progress_bar=True,
                                    convert_to_tensor=True, device=device)

    return title_embeddings.to("cpu")


item_title = SequenceEncoder(items["movie_title"], model_name='all-MiniLM-L6-v2')
item_genres = torch.tensor(items[genres.name].to_numpy(), dtype=torch.bool)
item_release_year = torch.tensor(items["release_year"].to_numpy()[:, np.newaxis], dtype=torch.int32)

item_x = torch.cat((item_title, item_genres), dim=-1).float()

user_ages = torch.tensor(users["age"].to_numpy()[:,np.newaxis], dtype=torch.uint8)
user_sex = torch.tensor(users[["male", "female"]].to_numpy(), dtype=torch.bool)
occupations = [i for i in users.keys() if i.startswith("occupation_")]
user_occupation = torch.tensor(users[occupations].to_numpy(), dtype=torch.bool)
user_x = torch.cat((user_ages, user_sex, user_occupation), dim=-1).float()

data = HeteroData()

data['user'].x = user_x
data['item'].x = item_x
data['user', 'rates', 'item'].edge_index = edge_index
data['user', 'rates', 'item'].edge_label = edge_attr

data = ToUndirected()(data)
del data['item', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.
data = data.to(device)


test_ratings = test_ratings.to_numpy()
users = test_ratings[:, 0] - 1
movies = test_ratings[:, 1] - 1
true_labels = torch.tensor(test_ratings[:, 2]).to(device)

x = torch.from_numpy(np.stack([users, movies], axis=0)).to(device)
pred_labels, _ = model(data.x_dict, data.edge_index_dict, x)

for name, metric in metrics:
    print(f"{name}: {metric(true_labels, pred_labels)}")
