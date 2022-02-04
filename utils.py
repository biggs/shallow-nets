#!/usr/bin/env python3

import numpy as np
import jax
import jax.lax
import jax.numpy as jnp
from jax import random, grad, jit, vmap

from scipy.optimize import root_scalar


def pretty_ordered_dict(d):
    return " -- ".join([f"{k}: {v:g}" for k, v in d.items()])


def camp_paulson(k, n, p):
    """Camp Paulson approximation to Binomial CDF from John D. Cook blog.

    Error < 0.007 / sqrt(n p (1-p))
    """
    k = jnp.floor(k)
    b = 1 / (9*k + 9)
    r = (k + 1) * (1 - p) / (n * p - k * p)
    a = 1 / (9 * n - 9 * k)
    s = (b * (r ** (2/3)) + a) ** 0.5
    m = 1 - a
    c = (1 - b) * (r ** (1/3))
    return jax.scipy.stats.norm.cdf((c - m) / s)


# @jit
# def invert_small_kl(train_loss, rhs):
#     "Get the inverted small-kl, largest p such that kl(train_loss : p) \le rhs"
#     start = train_loss + jnp.sqrt(rhs / 2.)    # start at McAllester
#     try:
#         res = root_scalar(lambda r: bernoulli_small_kl(train_loss, r) - rhs,
#                         x0=start, bracket=[train_loss, 1. - 1e-10])
#     except ValueError:
#         return 1.
#     return res.root

def invert_small_kl(train_loss, rhs):
    "Get the inverted small-kl, largest p such that kl(train_loss : p) \le rhs"
    initial_guess = train_loss + jnp.sqrt(rhs / 2.)    # start at McAllester
    initial_guess = jnp.minimum(initial_guess, 0.999)
    res = newton_method(lambda r: bernoulli_small_kl(train_loss, r) - rhs, initial_guess)
    return jnp.nan_to_num(res, nan=1.)


def newton_method(f, guess, epsilon=1e-8):
    grad_f = grad(f)
    step = lambda x: x - f(x) / grad_f(x)
    for epoch in range(6):
        contin = abs(f(guess)) > epsilon
        guess = jax.lax.cond(True, step, lambda x: x, guess)
    return guess





def reverse_invert_small_kl(train_loss, rhs):
    "Get the reverse inverted small-kl, smallest p such that kl(train_loss : p) \le rhs"
    try:
        res = root_scalar(lambda r: bernoulli_small_kl(train_loss, r) - rhs,
                        x0=train_loss / 2, bracket=[1e-10, train_loss])
    except ValueError:
        return 1.
    return res.root



def bernoulli_small_kl(q, p):
    return q * jnp.log(q/p) + (1-q) * jnp.log((1-q)/(1-p))



# Data Loading

def load_data(dataset_name, binary=True, openml=False):
    "Load datasets. Use (slow) openml option when tensorflow is not available."

    if openml:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        datasets = { "mnist": 554, "fashion": 40996, "cifar": 40927 }
        X, y = fetch_openml(data_id=datasets[dataset_name], return_X_y=True, as_frame=False)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=10000)
    else:
        import tensorflow.keras as keras
        if dataset_name == "mnist":
            d = keras.datasets.mnist.load_data()
        elif dataset_name == "fashion":
            d = keras.datasets.fashion_mnist.load_data()
        elif dataset_name == "cifar":
            d = keras.datasets.cifar10.load_data()
        else:
            print("dataset not found!")
        (x_train, y_train), (x_test, y_test) = d

        # Fix weird cifar labels.
        if dataset_name == "cifar":
            y_train = np.squeeze(y_train)
            y_test = np.squeeze(y_test)
            x_train = np.reshape(x_train, [-1, 32 * 32 * 3])
            x_test = np.reshape(x_test, [-1, 32 * 32 * 3])
        else:
            x_train = np.reshape(x_train, [-1, 784])
            x_test = np.reshape(x_test, [-1, 784])

    x_train = x_train / 255.
    x_test = x_test / 255.
    if binary:
        y_train = np.array([-1 if int(y) < 5 else 1 for y in y_train])
        y_test = np.array([-1 if int(y) < 5 else 1 for y in y_test])
    else:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    return x_train, y_train, x_test, y_test


def batch_generator(xs, ys, batch_size=256):
    "Yield batches for a single epoch."
    for i in range(0, len(xs), batch_size):
        yield xs[i:i+batch_size], ys[i:i+batch_size]
