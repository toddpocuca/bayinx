import jax.numpy as jnp
from jaxtyping import Array, Scalar

from bayinx import Model, Posterior, define
from bayinx.dists import Normal
from bayinx.flows import DiagAffine, FullAffine
from bayinx.nodes import Continuous, Observed


class ShiftedNormal(Model, init=False):
    mu: Continuous[Scalar] = define(shape = ())
    y: Observed[Array] = define()

    def model(self, target):
        self.y << Normal(self.mu, 1.0)

        return target

class CenteredNormal(Model):
    sigma: Continuous[Scalar] = define(shape = (), lower = 0.0)
    y: Observed[Array] = define()

    def model(self, target):
        self.y << Normal(0.0, self.sigma)

        return target

class ArbitraryNormal(Model):
    mu: Continuous[Scalar] = define(shape = ())
    sigma: Continuous[Scalar] = define(shape = (), lower = 0.0)
    y: Observed[Array] = define()

    def model(self, target):
        self.y << Normal(self.mu, self.sigma)

        return target

y_data: Array = jnp.array([-1.0, 0.0, 1.0])

def test_shifted():
    # Construct and configure posterior
    posterior = Posterior(ShiftedNormal, y = y_data)
    posterior.configure(flowspecs = [DiagAffine()])

    # Fit variational approximation
    posterior.fit(max_iters = 1000)
    posterior.fit(max_iters = 1000, grad_draws = 4)
    posterior.fit(max_iters = 10000, grad_draws = 6, batch_size = 3, learning_rate=1e-3)

    # Generate posterior samples
    mu_post = posterior.sample(node = 'mu', n_draws = 10000)

    assert mu_post.shape == (10000, )



def test_scaled():
    y_data: Array = jnp.array([-1.0, 0.0, 1.0])

    posterior = Posterior(CenteredNormal, y = y_data)
    posterior.configure(flowspecs = [DiagAffine()])
    posterior.fit(max_iters = 1000)
    posterior.fit(max_iters = 1000, grad_draws = 4)
    posterior.fit(max_iters = 1000, grad_draws = 6, batch_size = 3)

    # Generate posterior samples
    sigma_post = posterior.sample(node = 'sigma', n_draws = 10000)

    assert sigma_post.shape == (10000, )

def test_arbitrary():
    y_data: Array = jnp.array([-1.0, 0.0, 1.0])

    posterior = Posterior(ArbitraryNormal, y = y_data)
    posterior.configure(flowspecs = [FullAffine()])
    posterior.fit(max_iters = 1000)
    posterior.fit(max_iters = 1000, grad_draws = 4)
    posterior.fit(max_iters = 1000, grad_draws = 6, batch_size = 3)

    # Generate posterior samples
    mu_post = posterior.sample(node = 'mu', n_draws = 10000)
    sigma_post = posterior.sample(node = 'sigma', n_draws = 10000)

    assert mu_post.shape == (10000, )
    assert sigma_post.shape == (10000, )
