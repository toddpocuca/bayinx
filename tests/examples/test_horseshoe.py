import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Cauchy, Normal
from bayinx.flows import FullAffine, Sylvester
from bayinx.nodes import Continuous


# Define model
class HorseshoeModel(byx.Model):
    x: Continuous[Scalar] = define(shape = ())
    lam: Continuous[Scalar] = define(shape = (), lower = 0)

    def model(self, target):
        self.lam << Cauchy(0, 1)
        self.x << Normal(0, self.lam)


def test_inference():
    # Define posterior
    posterior = byx.Posterior(HorseshoeModel)
    posterior.configure([FullAffine()] + [Sylvester(2)] * 16)
    posterior.fit(learning_rate = 2e-4, max_iters = 500_000, grad_draws = 8, max_batch_size = 8)

    # Get posterior samples
    x_draws = posterior.sample('x', int(1e5))

    # Get ground-truth draws
    true_draws = jr.normal(jr.key(0), (int(1e5),)) * jnp.abs(jr.cauchy(jr.key(0), (int(1e5), )))

    # Compare distributions
    q = jnp.linspace(0.1, 0.9, 9)
    x_qs = jnp.quantile(x_draws, q)
    true_qs = jnp.quantile(true_draws, q)

    assert jnp.abs(( x_qs - true_qs ) / true_qs) < 1.0
