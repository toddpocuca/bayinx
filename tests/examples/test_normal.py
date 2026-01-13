"""
Test inference for a simple data-generating model for data derived from a Normal distribution:

    X_1, ..., X_n ~ Normal(mu, sigma^2)

With expectation:

    E[mu | X = x] = mean(x)
    E[sigma^2 | X = x]

and variance:

    Var(mu | X = x) =
"""
from jaxtyping import Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Normal
from bayinx.flows import FullAffine, Sylvester
from bayinx.nodes import Continuous


# Define model
class ConjugateNormal(byx.Model):
    x: Continuous[Scalar] = define(())

    def model(self, target):
        self.x << Normal(0.0, 1.0)

def test_inference():
    # Define posterior
    posterior = byx.Posterior(ConjugateNormal, n_obs = n_obs, x = x)
    posterior.configure([FullAffine()] + [Sylvester(2)] * 20)
    posterior.fit(learning_rate = 1e-3, max_iters = 1_500_000, grad_draws = 8, batch_size = 8)

    mu_draws = posterior.sample('mu', int(5e6))

    # Confirm approximation is accurate
    assert abs(mu_draws.mean() - x.mean()) < 0.01
    assert abs(mu_draws.var(ddof = 1) - x.var(ddof = 1) / n_obs * (n_obs - 1) / (n_obs - 3)) < 0.1
