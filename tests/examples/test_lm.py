"""
Test inference for a linear model:

    Y_i ~ Normal(x_i * beta, sigma)
"""

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Normal
from bayinx.flows import LowRankAffine
from bayinx.nodes import Continuous, Observed


# Define model
class LinearModel(byx.Model):
    beta: Continuous[Scalar] = define(shape = 'n_predictors')
    sigma: Continuous[Scalar] = define(shape = (), lower = 0.0)

    X: Observed[Array] = define(shape = ('n_obs', 'n_predictors'))
    y: Observed[Array] = define(shape = 'n_obs')

    def model(self, target):
        # Compute expected response
        mu = self.X @ self.beta

        # Accumulate likelihood
        self.y << Normal(mu, self.sigma)

        return target

# Simulate sample
n_obs = 2500
n_predictors = 5
X: Array = jr.normal(jr.key(0), (n_obs, n_predictors - 1))
X = jnp.column_stack((jnp.ones((n_obs,)), X))
beta = jnp.array(range(n_predictors))

y = jr.normal(jr.key(0), (n_obs, )) * 0.5 + X @ beta

def test_inference():
    # Define posterior
    posterior = byx.Posterior(LinearModel,
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
