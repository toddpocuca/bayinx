import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Scalar
from plotnine import *

import bayinx as byx
from bayinx import define
from bayinx.dists import Cauchy, Normal
from bayinx.flows import CNeuralAffine
from bayinx.nodes import Continuous


# Define model
class HorseshoeModel(byx.Model):
    x: Continuous[Scalar] = define(shape = ())
    lam: Continuous = define(shape = (), lower = 0)

    def model(self, target):
        self.lam << Cauchy(0.0, 1.0)
        self.x << Normal(0.0, self.lam)


def test_inference():
    # Define posterior
    posterior = byx.Posterior(HorseshoeModel)
    posterior.configure([CNeuralAffine(flip = i % 2 == 0, key = jr.key(i)) for i in range(4)])
    posterior.fit(200_000, learning_rate = 1e-3, stl = True)

    # Get posterior samples
    x_draws = posterior.sample('x', int(3e4), sir = True)

    # Get ground-truth draws
    true_draws = jr.normal(jr.key(0), (int(1e5),)) * jnp.abs(jr.cauchy(jr.key(1), (int(1e5), )))

    plot = ggplot() + \
        geom_density(aes(x = np.array(x_draws)), color = 'green', linetype = 'dashed', size = 1) + \
        geom_density(aes(x = np.array(true_draws)), color = 'black', linetype = 'solid', size = 1) + \
        xlim(-5, 5)


    plot.show()
    # Compare distributions
    q = jnp.linspace(0.1, 0.9, 9)
    x_qs = jnp.quantile(x_draws, q)
    true_qs = jnp.quantile(true_draws, q)

    assert all(jnp.abs(( x_qs - true_qs ) / true_qs) < 1.0)
