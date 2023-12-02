# Movie Recommender System

Movie recommender system using Graph Neural Network.

Libraries used for GNN:
* PyTorch
* torch_geometric

## Usage

Generate intermediate dataset
```bash
python ./src/dataset.py
```

Training the model
```bash
python ./src/train.py
```

Trained models are located in `models`

Evaluating the model
```bash
python ./benchmark/evaluation.py
```