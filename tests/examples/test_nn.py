from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
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
            eqx.nn.Linear('scalar', 20, key=jr.key(0)),
            jax.nn.leaky_relu,
            eqx.nn.Linear(20, 20, key=jr.key(0)),
            jax.nn.leaky_relu,
            eqx.nn.Linear(20, 20, key=jr.key(0)),
            jax.nn.leaky_relu,
            eqx.nn.Linear(20, 20, key=jr.key(0)),
            jax.nn.leaky_relu,
            eqx.nn.Linear(20, 'scalar', key=jr.key(0))
        ]

    @partial(eqx.filter_vmap, in_axes = (None, 0))
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



# Define model
class NeuralNetworkModel(byx.Model):
    nn: Continuous[MyNeuralNetwork] = define(
        init = MyNeuralNetwork()
    )
    sigma: Continuous[Scalar] = define(shape = (), lower = 0.0)

    x: Observed[Array] = define(shape = 'n_obs')
    y: Observed[Array] = define(shape = 'n_obs')

    def model(self, target):
        # Set prior
        self.nn << Normal(0.0, 2.0)

        # Compute expected response
        mu = self.nn(self.x)

        # Accumulate likelihood
        self.y << Normal(mu, self.sigma)

        return target

# Simulate sample
n_obs = 5000
x: Array = jr.uniform(jr.key(0), (n_obs, ), minval = -4.0, maxval = 4.0)
def f(x):
    return 1 + 3*x**2 + x*jnp.sin(x)

y = f(x)

def test_inference():
    # Define posterior
    posterior = byx.Posterior(NeuralNetworkModel,
        n_obs = n_obs,
        x = x,
        y = y
    )

    # Configure and fit
    posterior.configure(flowspecs = [DiagAffine()])
    posterior.fit(max_iters = int(2e5), grad_draws = 4, batch_size = 4)

    # Test for good fit
    assert posterior.sample('sigma', 1000).mean() < 0.1

    # Test on new data
    x_new = jnp.array([0.0, 1.0, 2.0, 3.0])
    y_new = f(x_new)
    predictions = posterior.predictive(
        lambda model, key: model.nn(x_new),
        10000
    )

    assert all(abs(y_new - predictions).mean(0) < 0.1)
