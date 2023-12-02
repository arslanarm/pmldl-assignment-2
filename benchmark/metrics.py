import torch


def rmse(true: torch.Tensor, pred: torch.Tensor):
    return (true - pred).pow(2).mean().sqrt()


def mae(true: torch.Tensor, pred: torch.Tensor):
    return torch.abs(true - pred).mean()


metrics = [
    ("RMSE", rmse),
    ("MAE", mae)
]
