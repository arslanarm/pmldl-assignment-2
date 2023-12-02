import pandas as pd


def preprocess_data():
    items = pd.read_csv("../data/interm/items.csv")
    items.to_csv("data/items.csv", index=False)
    users = pd.read_csv("../data/interm/users.csv")
    users.to_csv("data/users.csv", index=False)

    ratings_base = pd.read_csv("../data/raw/ml-100k/ua.base", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    del ratings_base['timestamp']
    ratings_base.to_csv("data/ratings-base.csv", index=False)
    ratings_base = pd.read_csv("../data/raw/ml-100k/ua.test", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    del ratings_base['timestamp']
    ratings_base.to_csv("data/ratings-test.csv", index=False)
