"""
Test inference for a generalized linear model:

    Y_i ~ Poisson(e^{x_i * beta}, sigma)
"""

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
import bayinx.ops as byo
from bayinx import define
from bayinx.dists import Poisson
from bayinx.flows import FullAffine
from bayinx.nodes import Continuous, Observed


# Define model
class PoissonModel(byx.Model):
    n_obs: Observed[int] = define()
    beta: Continuous[Scalar] = define(shape = 'n_pred')

    X: Observed[Array] = define(shape = ('n_obs', 'n_pred'))
    y: Observed[Array] = define(shape = 'n_obs', lower = 0)

    def model(self, target):
        # Accumulate likelihood
        byo.map(
            lambda y_i, x_i: y_i << Poisson(log_rate = x_i @ self.beta),
            self.y, self.X
        )
        #byo.fori_loop(
        #    0, self.n_obs,
        #    lambda i: self.y[i] << Poisson(log_rate = self.X[i] @ self.beta)
        #)
        #self.y << Poisson(log_rate = self.X @ self.beta)

        return target

# Simulate sample
n_obs = 2000
n_pred = 10
X: Array = jr.normal(jr.key(0), (n_obs, n_pred - 1)) * 0.1
X = jnp.column_stack((jnp.ones((n_obs,)), X))
beta = jnp.array(range(n_pred)) + 1

y = jr.poisson(jr.key(0), jnp.exp(X @ beta), (n_obs, ))

def test_inference():
    # Define posterior
    posterior = byx.Posterior(PoissonModel,
        n_obs = n_obs,
        n_pred = n_pred,
        X = X,
        y = y
    )

    # Configure and fit
    posterior.configure(flowspecs = [FullAffine()])
    posterior.fit()

    # Check fit
    assert jnp.linalg.norm(posterior.sample('beta', int(1e5)).mean(0) - beta) < 0.1
