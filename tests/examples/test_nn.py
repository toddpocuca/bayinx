"""
Test inference for a linear model:

    Y_i ~ Normal(x_i * beta, sigma)
"""

from functools import partial

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Normal
from bayinx.flows import DiagAffine
from bayinx.nodes import Continuous, Observed


class MyNeuralNetwork(eqx.Module):
    layers: list

    def __init__(self):
        self.layers = [
            eqx.nn.MLP('scalar', 'scalar', 10, 3, key=jr.key(0))
        ]

    @partial(eqx.filter_vmap, in_axes = (None, 0))
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



# Define model
class NeuralNetworkModel(byx.Model):
    nn: Continuous[Scalar] = define(
        init = MyNeuralNetwork()
    )
    sigma: Continuous[Scalar] = define(shape = (), lower = 0.0)

    x: Observed[Array] = define(shape = 'n_obs')
    y: Observed[Array] = define(shape = 'n_obs')

    def model(self, target):
        # Set prior
        self.nn << Normal(0.0, 3.0)

        # Compute expected response
        mu = self.nn(self.x)

        # Accumulate likelihood
        self.y << Normal(mu, self.sigma)

        return target

# Simulate sample
n_obs = 100
x: Array = jr.normal(jr.key(0), (n_obs, ))

y = x

def test_inference():
    # Define posterior
    posterior = byx.Posterior(NeuralNetworkModel,
        n_obs = n_obs,
        x = x,
        y = y
    )

    # Configure and fit
    posterior.configure(flowspecs = [DiagAffine()])
    posterior.fit(max_iters = int(1e5), learning_rate = 1e-4, grad_draws = 4, batch_size = 4)

    predictions = posterior.predictive(lambda model, key: model.nn(x), 10000)
