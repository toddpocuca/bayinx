"""
Test inference for a simple data-generating model for data derived from a Normal distribution:

    X_1, ..., X_n ~ Normal(mu, sigma)

The marginal posteriors should be:

    mu | X = x ~ T-Dist(mean(x), std(x) / sqrt(n))
    sigma^2 | X = x ~ Inv-Gamma((n-1) / 2, (n-1)s^2 / 2)

With expectation:

    E[mu | X = x] = mean(x)
    E[sigma^2 | X = x]

and variance:

    Var(mu | X = x) =
"""

import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Normal
from bayinx.flows import DiagAffine
from bayinx.nodes import Continuous, Observed


# Define model
class SimpleNormalModel(byx.Model):
    mu: Continuous[Scalar] = define(shape = ())
    sigma: Continuous[Scalar] = define(shape = (), lower = 0.0)

    x: Observed[Array] = define(shape = 'n_obs')

    def model(self, target):
        self.x << Normal(self.mu, self.sigma)

        return target

# Simulate sample
n_obs = 100
x: Array = jr.normal(jr.key(0), (n_obs, )) * 5 + 10

def test_inference():
    # Define posterior
    posterior = byx.Posterior(SimpleNormalModel, n_obs = n_obs, x = x)
    posterior.configure([DiagAffine()])
    posterior.fit()

    # Get posterior
    mu_draws = posterior.sample('mu', int(1e6))
    #sigma_draws = posterior.sample('sigma', int(1e6))

    # Confirm approximation is accurate
    # mu | X ~ t_n-1 (mu = xbar, sigma = s / sqrt(n)) ==> E[mu | X] = xbar
    # sigma^2 | X ~ inv-gamma (alpha = (n-1) / 2, beta = (n-1)s^2 / 2) ==> E[sigma^2 | X] =
    assert abs(mu_draws.mean() - x.mean()) < 0.01
    assert abs(mu_draws.var(ddof = 1) - x.var(ddof = 1) / n_obs * (n_obs - 1) / (n_obs - 3)) < 0.1 # this is not passing for low 'n_obs'
