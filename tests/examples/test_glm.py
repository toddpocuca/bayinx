"""
Test inference for a generalized linear model:

    Y_i ~ Poisson(e^{x_i * beta}, sigma)
"""

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Poisson
from bayinx.flows import LowRankAffine
from bayinx.nodes import Continuous, Observed


# Define model
class PoissonModel(byx.Model):
    beta: Continuous[Scalar] = define(shape = 'n_predictors')

    X: Observed[Array] = define(shape = ('n_obs', 'n_predictors'))
    y: Observed[Array] = define(shape = 'n_obs', lower = 0)

    def model(self, target):
        # Accumulate likelihood
        self.y << Poisson(log_rate = self.X @ self.beta)

        return target

# Simulate sample
n_obs = 1000
n_predictors = 5
X: Array = jr.normal(jr.key(0), (n_obs, n_predictors - 1))
X = jnp.column_stack((jnp.ones((n_obs,)), X))
beta = jnp.array(range(n_predictors))

y = jr.poisson(jr.key(0), jnp.exp(X @ beta), (n_obs, ))

def test_inference():
    # Define posterior
    posterior = byx.Posterior(PoissonModel,
        n_obs = n_obs,
        n_predictors = n_predictors,
        X = X,
        y = y
    )

    # Configure and fit
    posterior.configure(flowspecs = [LowRankAffine(2)])
    posterior.fit(max_iters = int(1e5), learning_rate = 1e-2)

    # Check fit
    assert jnp.linalg.norm(posterior.sample('beta', int(1e6)).mean(0) - beta) < 0.1
