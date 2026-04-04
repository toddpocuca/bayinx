import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import NegBinom
from bayinx.flows import DiagAffine, LinearRationalSpline
from bayinx.nodes import Continuous, Observed


# Define model
class SimpleBinomialModel(byx.Model, init = False):
    mu: Continuous[Scalar] = define(shape = (), lower = 0)
    theta: Continuous[Scalar] = define(shape = (), lower = 0)

    x: Observed[Array] = define(shape = 'n_obs', lower = 0)

    def model(self, target):
        self.x << NegBinom(self.mu, self.theta)

# Simulate sample
n_obs = 100
n = 1
x: Array = jr.poisson(jr.key(0), 5, (n_obs, ))

model = SimpleBinomialModel(n_obs = n_obs,n = n,x = x)


def test_inference():
    # Define posterior
    posterior = byx.Posterior(
        SimpleBinomialModel,
        n_obs = n_obs,
        n = n,
        x = x
    )
    posterior.configure([DiagAffine()])
    posterior.fit(stl = True)
    posterior.configure([LinearRationalSpline()] * 3, insert = 'prepend')
    posterior.fit(500_000, stl = True)

    # Get posterior
    p_draws = posterior.sample('p', int(1e6))

    # Confirm approximation is accurate
    # p | X ~ beta(alpha = x + 1, beta = n + 1 - x) ==> E[p | X] = mean(x)
    assert abs(p_draws.mean() - x.mean()) < 0.01
