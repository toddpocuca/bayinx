import jax.random as jr
from jaxtyping import Array, Scalar

import bayinx as byx
from bayinx import define
from bayinx.dists import Binomial
from bayinx.flows import DiagAffine
from bayinx.nodes import Continuous, Observed


# Define model
class SimpleBinomialModel(byx.Model, init=False):
    p: Continuous[Scalar] = define(shape = (), lower = 0, upper = 1)

    x: Observed[Array] = define(shape = 'n_obs', lower = 0)
    n: Observed[Array] = define(shape = (), lower = 1)

    def model(self, target):
        self.x << Binomial(self.n, self.p)

        return target

# Simulate sample
n_obs = 100
n = 1
x: Array = jr.binomial(jr.key(0), n, 0.5, (n_obs, ))

def test_inference():
    # Define posterior
    posterior = byx.Posterior(
        SimpleBinomialModel,
        n_obs = n_obs,
        n = n,
        x = x
    )
    posterior.configure([DiagAffine()])
    posterior.fit(learning_rate = 1e-4)

    # Get posterior
    p_draws = posterior.sample('p', int(1e6))

    # Confirm approximation is accurate
    # p | X ~ beta(alpha = x + 1, beta = n + 1 - x) ==> E[p | X] = p
    assert abs(p_draws.mean() - x.mean()) < 0.01
