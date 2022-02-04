#!/usr/bin/env python3
from collections import OrderedDict

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import numpy as np

from jax.scipy.special import logsumexp

from utils import invert_small_kl
from models import SHELNet, initiate_params, layer_norms


# Standard NN CE and training.

def loss(model, xs, ys):
    def single_loss(x, y):
        logits = model.predict(x)
        return -logits[y] + logsumexp(logits)
    losses = vmap(single_loss, [0, 0])(xs, ys)
    return jnp.mean(losses)


@jit
def accuracy(model, images, targets):
    predicted_class = jnp.argmax(model.batched_predict(images), axis=1)
    return jnp.mean(predicted_class == targets)



# Stochastic version.

def stochastic_losses(key, model, images, targets, sigma_1=0.1, sigma_2=0.1):
    preds = model.batch_stochastic_predict(key, images, sigma_1, sigma_2)
    return (jnp.argmax(preds, axis=1) != targets).astype(float)


@jit
def overall_stochastic_loss(key, model, images, targets, sigma_1=0.1, sigma_2=0.1):
    return jnp.mean(stochastic_losses(key, model, images, targets, sigma_1, sigma_2))


def stochastic_loss_by_sigma(key, model, images, targets, sigma_grid):
    l = []
    for sig1, sig2 in sigma_grid:
        key, subkey = random.split(key)
        l += [[sig1, sig2, overall_stochastic_loss(subkey, model, images, targets, sig1, sig2)]]
    return jnp.array(l)


def hoeffding_stochastic_loss(key, no_passes, model, images, targets, sigma_1=0.1, sigma_2=0.1, d_prime=0.01):
    sloss = lambda key: overall_stochastic_loss(
            key, model, images, targets, sigma_1=sigma_1, sigma_2=sigma_2)
    avg = sum([sloss(k) for k in random.split(key, no_passes)]) / no_passes
    return avg + jnp.sqrt(jnp.log(1/d_prime) / 2 / no_passes / images.shape[0])



# PAC-Bayes things.

@jit
def seeger_bound(loss, kl_complexity, m, delta=0.025):
    "Given Loss and KL (complexity), calculate Seeger bound."
    b = kl_complexity + jnp.log(2 * jnp.sqrt(m) / delta)
    return invert_small_kl(loss, b / m)


@jit
def best_bound(train_bnd_losses, model, prior, bnd_set_size, printout=False):
    "Return array (bound, sigma1, sigma2) for best bound from a set of bnd losses."
    def bound_vmap_util(x, model, prior, bnd_set_size):
        sigma1, sigma2, sloss = x[0], x[1], x[2]
        l1, l2 = model.complexity_layers(prior, sigmas=(sigma1, sigma2))
        bnd =  2 * seeger_bound(sloss, l1 + l2, bnd_set_size)
        return jnp.array([bnd, sigma1, sigma2])
    bounds = vmap(bound_vmap_util, [0, None, None, None])(
        train_bnd_losses, model, prior, bnd_set_size)
    return bounds[jnp.argmin(bounds, axis=0)[0]]


def grid_sigma_final_evaluation(key, model, data, init_params, sigma_grid, kl_extra):
    x_train, y_train, _, _ = data
    m = x_train.shape[0]
    slosses = stochastic_loss_by_sigma(key, model, x_train, y_train, sigma_grid)
    best_bound_result = best_bound(slosses, model, init_params, m, printout=True)
    best_sigmas = (best_bound_result[1], best_bound_result[2])
    best_full = true_bound(key, model, init_params, x_train, y_train, kl_extra, best_sigmas)
    print(f"\nBest Full Dataset Bound = {best_full:g}  at Sigmas = {best_sigmas}")
    return best_full


def true_bound(key, model, prior, images, labels, union_num, sigmas, delta=0.025):
    loss_ub = hoeffding_stochastic_loss(
        key, 20, model, images, labels, sigma_1=sigmas[0], sigma_2=sigmas[1])
    l1, l2 = model.complexity_layers(prior, sigmas)
    bnd = 2 * seeger_bound(
        loss_ub, l1 + l2 + np.log(union_num), images.shape[0], delta=delta)
    return bnd



# Evaluation during training.

def train_step_eval(key, model, prior, data, ce_loss, full_eval=False, sigmas=(0.1, 0.1)):

    # Use subset of examples for faster evaluation.
    eval_size = None if full_eval else 5000
    x_train, y_train, x_test, y_test = data
    x_train, y_train = x_train[:eval_size], y_train[:eval_size]
    x_test, y_test =  x_test[:eval_size], y_test[:eval_size]
    m = x_train.shape[0]

    sloss = overall_stochastic_loss(
        key, model, x_train, y_train, sigma_1=sigmas[0], sigma_2=sigmas[1])
    n1, n2 = layer_norms(model, prior)
    l1, l2 = model.complexity_layers(prior, sigmas=sigmas)
    seeger = 2 * seeger_bound(sloss, l1 + l2, m)

    return OrderedDict({
        "Test Error": 1 - accuracy(model, x_test, y_test),
        "CE": ce_loss,
        "Norm U": n1,
        "Norm V": n2,
        "Stochastic Loss": sloss,
        "Seeger Bound": seeger,    # Naive as using fixed sigmas.
        "Train Deterministic Loss": 1 - accuracy(model, x_train, y_train),
        })


def prior_step_eval(model, prior, train_bnd_losses, bnd_set_size):
    n1, n2 = layer_norms(model, prior)
    seeger, min_sigma1, min_sigma2 = best_bound(train_bnd_losses, model, prior, bnd_set_size)
    return OrderedDict({
        "Norm U": n1,
        "Norm V": n2,
        "Min Sigma U": min_sigma1,
        "Min Sigma V": min_sigma2,
        "Seeger Bound": seeger,
        })
