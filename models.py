#!/usr/bin/env python3

from math import prod

from typing import NamedTuple

import jax
from jax import jit, random, vmap
import jax.numpy as jnp



class Model(NamedTuple):
    # Params here.
    u: 'typing.Any'
    v: 'typing.Any'
    beta: float

    def predict(self, image):
        "Predict on image for SHEL with params."
        norm_image = image / jnp.linalg.norm(image, ord=2)
        hidden = self.activation(jnp.dot(self.u, norm_image))
        return jnp.dot(self.v, hidden)

    @jit
    def batched_predict(self, images):
        "Make a batch of predictions on a set of images."
        return vmap(self.predict, in_axes=(0,))(images)

    def stochastic_predict_one(self, key, image, sigma_1=0.1, sigma_2=0.1):
        key1, key2, key3 = random.split(key, 3)
        hidden = self.stochastic_first_layer(image, key1, sigma_1=sigma_1)
        noise_f = sigma_2 * jnp.linalg.norm(hidden) * random.normal(key3, (self.v.shape[0],))
        return jnp.dot(self.v, hidden) + noise_f

    @jit
    def batch_stochastic_predict(self, key, images, sigma_1=0.1, sigma_2=0.1):
        keys = random.split(key, images.shape[0])
        return vmap(self.stochastic_predict_one, [0, 0, None, None])(
            keys, images, sigma_1, sigma_2)



class SHELNet(Model):

    def activation(self, preactiv):
        return jax.lax.erf(self.beta * preactiv)

    def stochastic_first_layer(self, image, key, sigma_1=None):
        noise_h_scale = jnp.linalg.norm(image) / (jnp.sqrt(2) * self.beta)
        noise_h =  noise_h_scale * random.normal(key, (self.u.shape[0],))
        return jnp.sign(jnp.dot(self.u, image) + noise_h)

    @jit
    def complexity_layers(self, reference_params, sigmas, zero_prior_final=False):
        _, sigma_2 = sigmas
        norm1_sq, norm2_sq = layer_norms(self, reference_params, zero_prior_final)
        l1 = (self.beta ** 2) * norm1_sq
        l2 = norm2_sq / (2 * sigma_2 ** 2)
        return l1, l2



class SHELDoubleNet(SHELNet):

    def stochastic_first_layer(self, image, key):
        key1, key2 = random.split(key)
        noise_h_scale = jnp.linalg.norm(image) / (jnp.sqrt(2) * self.beta)
        noise_h_1 =  noise_h_scale * random.normal(key1, (self.u.shape[0],))
        noise_h_2 =  noise_h_scale * random.normal(key2, (self.u.shape[0],))
        h1 = jnp.sign(jnp.dot(self.u, image) + noise_h_1)
        h2 = jnp.sign(jnp.dot(self.u, image) + noise_h_2)
        return (h1 + h2) / 2.

    def complexity_layers(self, reference_params, sigma_2=0.01, zero_prior_final=False):
        l1, l2 = super().complexity_layers(reference_params, sigma_2, zero_prior_final)
        return 2 * l1, l2



class GELUNet(Model):

    def activation(self, preactiv):
        x = preactiv * self.beta
        return jax.scipy.stats.norm.cdf(x) * x

    def stochastic_first_layer(self, image, key, sigma_1=0.1):
        key1, key2 = random.split(key)
        noise_1_scale = jnp.linalg.norm(image) / self.beta
        noise_2_scale = jnp.linalg.norm(image) * sigma_1
        noise_h_1 =  noise_1_scale * random.normal(key1, (self.u.shape[0],))
        noise_h_2 =  noise_2_scale * random.normal(key2, (self.u.shape[0],))
        dot = jnp.dot(self.u, image)
        return ((dot + noise_h_1) >= 0.).astype(float) * (dot + noise_h_2)

    @jit
    def complexity_layers(self, reference_params, sigmas, zero_prior_final=False):
        sigma_1, sigma_2 = sigmas
        norm1_sq, norm2_sq = layer_norms(self, reference_params, zero_prior_final)
        l1 = 0.5 * (self.beta ** 2 + (1./sigma_1) ** 2) * norm1_sq
        l2 = norm2_sq / (2 * sigma_2 ** 2)
        return l1, l2



def layer_norms(model, reference_params, zero_prior_final=False):
    "Return the squared frobenius norms from model to reference."
    norm1 = jnp.linalg.norm(model.u - reference_params.u, ord='fro')
    v0 = 0 if zero_prior_final else reference_params.v
    norm2 = jnp.linalg.norm(model.v - v0, ord='fro')
    return norm1 ** 2, norm2 ** 2


def single_hidden_params_init(key, width, in_size, out_size, scale=1e-1):
    "Create random Normal(0, scale) params for a single hidden layer network."
    # Also create biases alongside.
    key_1, key_2 = random.split(key)
    l2_shape = (width, ) if out_size == 1 else (out_size, width)
    weights1 = scale * random.normal(key_1, (width, in_size))
    weights2 = scale * random.normal(key_2, l2_shape)
    return weights1, weights2


def initiate_params(key, model_type, width, beta, out_size=10, in_size=784):
    "Randomly intialise parameters for a single layer NN."
    assert type(beta) == float    # Beta must be a float or we get error in grad.
    u, v = single_hidden_params_init(key, width, in_size, out_size)
    return model_type(u, v, beta)


def number_of_parameters(model):
    "Number of paramters in a shallow NN."
    return prod(model.u.shape) + prod(model.v.shape)
