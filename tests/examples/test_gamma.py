import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Gamma
from bayinx.flows import DiagAffine
from bayinx.nodes import Continuous, Observed


# Define model
class SimpleGammaModel(byx.Model):
    rate: Continuous[Scalar] = define(shape = (), lower = 0)
    shape: Continuous[Scalar] = define(shape = (), lower = 0)

    x: Observed[Array] = define(lower = 0)

    def model(self, target):
        self.x << Gamma(rate = self.rate, shape = self.shape)


# Simulate sample
n_obs = 100
x: Array = jr.gamma(jr.key(0), 2, (n_obs, ))

def test_inference():
    # Define posterior
    posterior = byx.Posterior(
        SimpleGammaModel,
        x = x
    )
    posterior.configure([DiagAffine()])
    posterior.fit()

    # Get posterior
    rate_draws = posterior.sample('mean', int(1e6))

    # Confirm approximation is accurate
    # p | X ~ beta(alpha = x + 1, beta = n + 1 - x) ==> E[p | X] = mean(x)
    assert abs(rate_draws.mean() - x.mean()) < 0.01
