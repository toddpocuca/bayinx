import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Scalar
from plotnine import *

import bayinx as byx
from bayinx import define
from bayinx.dists import Exponential, Normal
from bayinx.flows import DiagAffine
from bayinx.nodes import Continuous, Observed


class MyNeuralNetwork(eqx.Module):
    layers: list

    def __init__(self):
        self.layers = [
            eqx.nn.Linear('scalar', 6, key=jr.key(0)),
            jax.nn.leaky_relu,
            eqx.nn.Linear(6, 3, key=jr.key(0)),
            jax.nn.leaky_relu,
            eqx.nn.Linear(3, 3, key=jr.key(0)),
            jax.nn.leaky_relu,
            eqx.nn.Linear(3, 'scalar', key=jr.key(1))
        ]

    @eqx.filter_vmap(in_axes = (None, 0))
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
        self.nn << Normal(0, 3)
        self.sigma << Exponential(1)

        # Compute expected response
        mu = self.nn(self.x)

        # Accumulate likelihood
        self.y << Normal(mu, self.sigma)

        return target


# Simulate sample
n_obs = 1000
x: Array = jr.uniform(jr.key(0), (n_obs, ), minval = -4.0, maxval = 4.0)
def f(x):
    return jnp.sin(x)

y = f(x) + jr.normal(jr.key(1), (n_obs, )) * 0.1

def test_inference():
    # Define posterior
    posterior = byx.Posterior(NeuralNetworkModel,
        n_obs = n_obs,
        x = x,
        y = y
    )

    # Configure and fit
    posterior.configure(flowspecs = [DiagAffine()])
    posterior.fit(100_000, learning_rate = 0.1, grad_draws = 2, batch_size = 2, stl = True)

    # Test for good fit
    assert posterior.sample('sigma', 1000).mean() < 0.1

    # Test on new data
    x_new = jnp.linspace(-4, 4, 30)
    y_new = f(x_new)
    predictions = posterior.predictive(
        lambda model, key: Normal(model.nn(x_new), model.sigma).sample(x_new.shape, key),
        10000
    )

    plot = (
        ggplot() +
            geom_ribbon(aes(
                x = np.array(x_new),
                ymin = np.array(jnp.quantile(predictions, 0.1, 0)),
                ymax = np.array(jnp.quantile(predictions, 0.9, 0))
            ))
        )
    plot.show()
