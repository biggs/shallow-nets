#!/usr/bin/env python3

from collections import OrderedDict
from main import run, MAX_EPOCHS

GRID = (0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2)


def get_hparams_gelu_fashion(key, lr, width, dataset, model_type, ce_stop):
    hparams = OrderedDict({
        "Beta": 5.,
        "Step Size": lr,
        "Layer Width": width,
        "Batch Size": 200,
        "Momentum": 0.9,
        "CE Stopping Value": ce_stop,
        "Prior Examples": 36000,
        "Dataset": dataset,
        "Train Size": 60000,
        "Model Type": model_type,
        "Sigma 2 Grid": GRID,
        "Sigma 1 Grid": GRID if model_type == "GELU" else (0.,),
        "Init Key": key,
    })

    KL_EXTRA = MAX_EPOCHS * len(hparams["Sigma 1 Grid"]) * len(hparams["Sigma 2 Grid"])
    hparams.update({"KL Extra": KL_EXTRA})
    return hparams


def without_momentum(key, lr, width, dataset, model_type, ce_stop):
    hparams = OrderedDict({
        "Beta": 5.,
        "Step Size": lr,
        "Layer Width": width,
        "Batch Size": 1000,
        "Momentum": 0.0,
        "CE Stopping Value": ce_stop,
        "Prior Examples": 36000,
        "Dataset": dataset,
        "Train Size": 60000,
        "Model Type": model_type,
        "Sigma 2 Grid": GRID,
        "Sigma 1 Grid": GRID if model_type == "GELU" else (0.,),
        "Init Key": key,
    })

    KL_EXTRA = MAX_EPOCHS * len(hparams["Sigma 1 Grid"]) * len(hparams["Sigma 2 Grid"])
    hparams.update({"KL Extra": KL_EXTRA})
    return hparams






keys = [2, 3, 4, 5]
widths = [50, 100, 200, 400, 800, 1600]
lrs = [0.1, 0.03, 0.01]

outfile = "results/grid-vanilla-sgd.csv"
# for m in ("GELU", "SHEL"):
#     for k in keys:
#         for l in lrs:
#             for w in widths:
#                 run(without_momentum(k, l, w, "fashion", m, 0.3), outfile)

for k in keys:
    for l in lrs:
        for w in widths:
            run(without_momentum(k, l, w, "mnist", "SHEL", 0.1), outfile)


# outfile = "results/grid-gelu-cifar.csv"
# for k in keys:
#     for l in lrs:
#         for w in widths:
#             run(get_hparams_gelu_fashion(k, l, w, "cifar", "GELU", 0.5), outfile)




def train_size_test(key, size, model_type, ce_stop, dataset):
    hparams = OrderedDict({
        "Beta": 5.,
        "Step Size": 0.01,
        "Layer Width": 200,
        "Batch Size": 200,
        "Momentum": 0.9,
        "CE Stopping Value": ce_stop,
        "Prior Examples": int(0.6 * size),
        "Dataset": dataset,
        "Train Size": size,
        "Model Type": model_type,
        "Sigma 2 Grid": GRID,
        "Sigma 1 Grid": GRID if model_type == "GELU" else (0.,),
        "Init Key": key,
    })

    KL_EXTRA = MAX_EPOCHS * len(hparams["Sigma 1 Grid"]) * len(hparams["Sigma 2 Grid"])
    hparams.update({"KL Extra": KL_EXTRA})
    return hparams


