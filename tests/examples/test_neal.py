import jax.nn
import jax.random as jr
import numpy as np
import polars as pl
from jaxtyping import Array, Scalar
from plotnine import *

import bayinx as byx
import bayinx.ops as byo
from bayinx import define
from bayinx.dists import Normal
from bayinx.flows import CNeuralAffine
from bayinx.nodes import Continuous


# Define model
class NealsFunnelModel(byx.Model):
    y: Continuous[Scalar] = define(shape = ())
    x: Continuous[Array] = define(shape = ())

    def model(self, target):
        self.y << Normal(0, 2)
        self.x << Normal(0, byo.exp(self.y / 2))


def test_inference():
    # Define posterior
    posterior = byx.Posterior(NealsFunnelModel)
    posterior.configure(
        [CNeuralAffine(
            flip = i % 2 == 0,
            activation = lambda x: 0.1 * jax.nn.relu(x),
            key = jr.key(i)
        ) for i in range(4)]
    )
    posterior.fit(50_000, grad_draws = 4, batch_size = 4, stl = True)
    print(posterior.prop_ess(), posterior.pareto_k(), posterior.vari.elbo(10_000, 1), posterior.sample('x', 10_000).mean())
    posterior.fit(900_000, grad_draws = 1, batch_size = 1, learning_rate = 1e-3, stl = True)
    print(posterior.prop_ess(), posterior.pareto_k(), posterior.vari.elbo(10_000, 1), posterior.sample('x', 10_000).mean())


    # Get posterior samples
    y_draws, x_draws = posterior.predictive(lambda model, key: (model.y, model.x), 100000)

    plot = ggplot(pl.DataFrame({'y': np.array(y_draws), 'x': np.array(x_draws)}), aes(x = 'x', y = 'y')) + \
        geom_point()

    plot.show()
