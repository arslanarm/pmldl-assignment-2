import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

data_folder = "data/"


def generate_intermediate_dataset():
    ratings = pd.read_csv(data_folder + "raw/ml-100k/u.data", delimiter="\t",
                          names=["user_id", "item_id", "rating", "timestamp"])
    users = pd.read_csv(data_folder + "raw/ml-100k/u.user", delimiter="|", names=["user_id", "age", "gender", "occupation", "zip_code"])
    genres = pd.read_csv(data_folder + "raw/ml-100k/u.genre", delimiter="|", names=["name", "genre_id"])
    items = []
    with open("../data/raw/ml-100k/u.item", "rb") as f:
        for line in f.readlines():
            try:
                line = line.decode()
            except:
                line = line.decode('iso-8859-1')
            items.append(line[:-1].split("|"))
    items = pd.DataFrame(items,
                         columns=["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url"] + list(
                             genres.name))
    users["male"] = users.gender == "M"
    users["female"] = users.gender == "F"
    del users["gender"]
    occupations = users.occupation.unique()
    for i in occupations:
        users["occupation_" + i] = users.occupation == i
    del users["occupation"]

    users.to_csv(data_folder + "interm/users.csv", index=False)

    def date_to_year(date):
        if not date:
            return None
        return int(date.split("-")[-1])

    items["release_year"] = items["release_date"].map(date_to_year)

    del items["release_date"]
    del items["video_release_date"]
    del items["imdb_url"]
    items.to_csv(data_folder + "interm/items.csv", index=False)
    del ratings["timestamp"]
    ratings.to_csv(data_folder + "interm/ratings.csv", index=False)


def load_dataset(device, users, items, ratings, genres):
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

    item_x = torch.cat((item_title, item_genres), dim=-1).float()

    user_ages = torch.tensor(users["age"].to_numpy()[:, np.newaxis], dtype=torch.uint8)
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

    return data


def main():
    generate_intermediate_dataset()


if __name__ == "__main__":
    main()