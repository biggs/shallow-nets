#!/usr/bin/env python3
import argparse
import os
import csv
from pprint import pprint
from collections import OrderedDict
from collections import deque
from itertools import product

import jax
from jax import random, jit
import jax.numpy as jnp

import models
from utils import pretty_ordered_dict, batch_generator, load_data
from evaluation import stochastic_loss_by_sigma, seeger_bound, best_bound, true_bound, accuracy, overall_stochastic_loss
from evaluation import train_step_eval, prior_step_eval, loss, grid_sigma_final_evaluation



MAX_EPOCHS = 4000
OPENML = False      # Load data using openml if tensorflow not available.


@jit
def update(model, x, y, step_size, velocity, momentum=0.):
    "Takes a model and a batch and returns new model."
    ce_loss, grads = jax.value_and_grad(loss)(model, x, y)
    u_velocity, v_velocity = velocity
    u_velocity = momentum * u_velocity - step_size * grads.u
    v_velocity = momentum * v_velocity - step_size * grads.v
    u = model.u + u_velocity
    v = model.v + v_velocity
    return ce_loss, type(model)(u, v, model.beta), (u_velocity, v_velocity)


def train_model(key, init_params, data, hparams):
    x_train, y_train, x_test, y_test = data
    model = init_params

    velocity = (0., 0.)
    for epoch in range(1, MAX_EPOCHS + 1):
        for xs, ys in batch_generator(x_train, y_train, batch_size=hparams["Batch Size"]):
            ce_loss, model, velocity = update(
                model, xs, ys, hparams["Step Size"], velocity, hparams["Momentum"])

        key, subkey = random.split(key)
        evaluation = train_step_eval(subkey, model, init_params, data, ce_loss)
        print(f"Epoch: {epoch}  ---  ", pretty_ordered_dict(evaluation))

        if ce_loss < hparams["CE Stopping Value"]:
            break
    return model


def train_prefix_prior(key, init_params, model, prior_examples, data, hparams, sigma_grid):
    x_train, y_train, x_test, y_test = data
    x_prior, y_prior = x_train[:prior_examples], y_train[:prior_examples]
    x_bnd, y_bnd = x_train[prior_examples:], y_train[prior_examples:]
    bnd_set_size = x_bnd.shape[0]
    assert len(x_prior) == prior_examples and (x_prior[0] == x_train[0]).all()

    train_bnd_losses = stochastic_loss_by_sigma(key, model, x_bnd, y_bnd, sigma_grid)
    print("Model Losses on BND set:\n", train_bnd_losses)

    prior = init_params
    priors = deque([init_params], 3)    # Store the last 3 priors.
    bounds = [2.,]
    velocity = (0., 0.)
    for epoch in range(1, 1 + MAX_EPOCHS):
        for xs, ys in batch_generator(x_prior, y_prior, batch_size=hparams["Batch Size"]):
            ce_loss, prior, velocity = update(
                prior, xs, ys, hparams["Step Size"], velocity, hparams["Momentum"])

        key, subkey = random.split(key)
        evaluation = prior_step_eval(model, prior, train_bnd_losses, bnd_set_size)
        print(f"Epoch: {epoch}  ---  ", pretty_ordered_dict(evaluation))

        # Stop training prior when go up twice.
        priors.append(prior)
        bounds.append(evaluation["Seeger Bound"])
        if bounds[-1] > bounds[-2] and bounds[-2] > bounds[-3]:
            best_prior = priors[-3]
            break
    else:
        best_prior = priors[-1]


    _, best_sig1, best_sig2 = best_bound(train_bnd_losses, model, best_prior, bnd_set_size)
    best_prior_bound = true_bound(key, model, prior, x_bnd, y_bnd, hparams["KL Extra"], (best_sig1, best_sig2))
    return OrderedDict({
        "Best Prior Bound": best_prior_bound,
        "Best Prior Sigma U": best_sig1,
        "Best Prior Sigma V": best_sig2,
        "Stochastic Bnd Loss": overall_stochastic_loss(
            key, model, x_bnd, y_bnd, sigma_1=best_sig1, sigma_2=best_sig2),
        "Stochastic Test Loss": overall_stochastic_loss(
            key, model, x_test, y_test, sigma_1=best_sig1, sigma_2=best_sig2),
    })


def run(hparams):
    key_1, key_2, key_3, key_4 = random.split(random.PRNGKey(hparams["Init Key"]), 4)
    sigma_grid = list(product(hparams["Sigma 1 Grid"], hparams["Sigma 2 Grid"]))

    data = load_data(hparams["Dataset"], binary=False, openml=OPENML)
    x_train, y_train, x_test, y_test = data

    model_type = models.GELUNet if hparams["Model Type"] == "GELU" else models.SHELNet
    init_params = models.initiate_params(
        key=key_1,
        model_type=model_type,
        width=hparams["Layer Width"],
        beta=hparams["Beta"],
        out_size=10,
        in_size=data[0].shape[1])

    print("\n\n  Training model  \n\n")
    model = train_model(key_2, init_params, data, hparams)

    print("\n\n  Full Dataset Evaluation  \n\n")
    best_full_bound = grid_sigma_final_evaluation(
        key_3, model, data, init_params, sigma_grid, hparams["KL Extra"])

    print("\n\n  Training Data-Dependent Prior  \n\n")
    coupled_prior_eval = train_prefix_prior(
        key_4, init_params, model, hparams["Prior Examples"], data, hparams, sigma_grid)

    print("\n\n  Final Evaluation Info  \n\n")
    n1, n2 = models.layer_norms(model, init_params)
    out_info = OrderedDict({
        "Best Full Bound": best_full_bound,
        "Test Error": 1 - accuracy(model, x_test, y_test),
        "Train Error": 1 - accuracy(model, x_train, y_train),
        "Norm Squared U": n1,
        "Norm Squared V": n2,
        "Number Parameters": models.number_of_parameters(model),
    })
    out_info.update(coupled_prior_eval)
    out_info.update(hparams)
    pprint(out_info)

    with open(outfile, "a+", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_info.keys())
        if not os.path.getsize(outfile):
            writer.writeheader()
        writer.writerow(out_info)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", default=5., type=float)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--width", default=100, type=int)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--momentum", default=0., type=float)
    parser.add_argument("--ce_stop", default=0.3, type=float)
    parser.add_argument("--prior_ex", default=36000, type=int)
    parser.add_argument("--dataset", default="fashion", type=str)
    parser.add_argument("--model", default="GELU", type=str)
    parser.add_argument("--key", default=0, type=int)
    args = parser.parse_args()

    GRID = (0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2)

    hparams = OrderedDict({
        "Beta": args.beta,
        "Step Size": args.lr,
        "Layer Width": args.width,
        "Batch Size": args.batch_size,
        "Momentum": args.momentum,
        "CE Stopping Value": args.ce_stop,
        "Prior Examples": args.prior_ex,
        "Dataset": args.dataset,
        "Model Type": args.model,
        "Sigma 2 Grid": GRID,
        "Sigma 1 Grid": GRID if args.model == "GELU" else (0.,),
        "Init Key": args.key
    })
    KL_EXTRA = MAX_EPOCHS * len(hparams["Sigma 1 Grid"]) * len(hparams["Sigma 2 Grid"])
    hparams.update({"KL Extra": KL_EXTRA})
    run(hparams)
