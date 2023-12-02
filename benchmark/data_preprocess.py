import pandas as pd


def preprocess_data(data_folder, output_folder):
    items = pd.read_csv(data_folder + "interm/items.csv")
    items.to_csv(output_folder + "items.csv", index=False)
    users = pd.read_csv(data_folder + "interm/users.csv")
    users.to_csv(output_folder + "users.csv", index=False)

    ratings_base = pd.read_csv(data_folder + "raw/ml-100k/ua.base", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    del ratings_base['timestamp']
    ratings_base.to_csv(output_folder + "ratings-base.csv", index=False)
    ratings_base = pd.read_csv(data_folder + "raw/ml-100k/ua.test", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    del ratings_base['timestamp']
    ratings_base.to_csv(output_folder + "ratings-test.csv", index=False)
