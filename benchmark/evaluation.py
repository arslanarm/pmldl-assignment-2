import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from benchmark.metrics import metrics
from data_preprocess import preprocess_data
from sentence_transformers import SentenceTransformer

from src.dataset import load_dataset
from src.model.gnn import Model, EdgeDecoder, GNNEncoder

device = "cuda:0"

model = torch.load("models/gnn_model.pt")
model.eval()

preprocess_data("data/", "benchmark/data/")

data_folder = "benchmark/data/"
users = pd.read_csv(data_folder + "users.csv")
items = pd.read_csv(data_folder + "items.csv")
ratings = pd.read_csv(data_folder + "ratings-base.csv")
test_ratings = pd.read_csv(data_folder + "ratings-test.csv")
genres = pd.read_csv("data/raw/ml-100k/u.genre", delimiter="|", names=["name","index"])

data = load_dataset(device, users, items, ratings, genres)

test_ratings = test_ratings.to_numpy()
users = test_ratings[:, 0] - 1
movies = test_ratings[:, 1] - 1
true_labels = torch.tensor(test_ratings[:, 2]).to(device)

x = torch.from_numpy(np.stack([users, movies], axis=0)).to(device)
pred_labels, _ = model(data.x_dict, data.edge_index_dict, x)

for name, metric in metrics:
    print(f"{name}: {metric(true_labels, pred_labels)}")
