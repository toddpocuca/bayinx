import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Normal, Poisson
from bayinx.flows import LowRankAffine
from bayinx.nodes import Continuous, Observed
from bayinx.ops import exp


# Define model
class PoissonGLM(byx.Model):
    beta: Continuous[Scalar] = define(shape = 'n_predictors')

    X: Observed[Array] = define(shape = ('n_obs', 'n_predictors'))
    y: Observed[Array] = define(shape = 'n_obs')

    def model(self, target):
        # Priors
        self.beta << Normal(0.0, 10.0)

        # Compute expected response
        mu = exp(self.X @ self.beta)

        # Accumulate likelihood
        self.y << Poisson(mu)

        return target

# Simulate example
n_obs = 100
n_predictors = 5
X: Array = jr.normal(jr.key(0), (n_obs, n_predictors - 1))
X = jnp.column_stack((jnp.ones((n_obs,)), X))
beta = jnp.array(range(n_predictors))

y = jr.poisson(jr.key(0), jnp.exp(X @ beta), (n_obs, ))

# Define posterior
posterior = byx.Posterior(PoissonGLM,
    n_obs = n_obs,
    n_predictors = n_predictors,
    X = X,
    y = y
)

# Configure and fit
posterior.configure(flowspecs = [LowRankAffine(2)])
posterior.fit(max_iters = int(1e5), learning_rate = 1e-2)

# Compute posterior mean estimate
mean_est = posterior.sample('beta', 10000).mean(0)
print(mean_est)
