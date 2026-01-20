import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Gamma
from bayinx.flows import DiagAffine
from bayinx.nodes import Continuous, Observed


# Define model
class SimpleBinomialModel(byx.Model):
    mean: Continuous[Scalar] = define(shape = (), lower = 0)
    shape: Continuous[Scalar] = define(shape = (), lower = 0)

    x: Observed[Array] = define(lower = 0)

    def model(self, target):
        self.x << Gamma(mean = self.mean, shape = self.shape)

# Simulate sample
n_obs = 1000
x: Array = jr.gamma(jr.key(0), 2, (n_obs, )) * 5

def test_inference():
    # Define posterior
    posterior = byx.Posterior(
        SimpleBinomialModel,
        x = x
    )
    posterior.configure([DiagAffine()])
    posterior.fit()

    # Get posterior
    rate_draws = posterior.sample('rate', int(1e6))

    # Confirm approximation is accurate
    # p | X ~ beta(alpha = x + 1, beta = n + 1 - x) ==> E[p | X] = mean(x)
    assert abs(rate_draws.mean() - x.mean()) < 0.01
