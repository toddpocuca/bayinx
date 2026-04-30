import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    mu: ArrayLike,
    theta: ArrayLike,
) -> Array:
    # Cast to Array
    x, mu, theta = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(theta)

    # Compute p
    p = theta / (theta + mu)

    return jnp.exp(jsp.gammaln(theta + x) - jsp.gammaln(theta) - jsp.gammaln(x + 1)) * jnp.power(p, theta) * jnp.power(1.0 - p, x)


def _logprob(
    x: ArrayLike,
    mu: ArrayLike,
    theta: ArrayLike,
) -> Array:
    # Cast to Array
    x, mu, theta = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(theta)

    # Compute log p and log(1-p) directly to avoid catastrophic cancellation
    log_p = jnp.log(theta) - jnp.log(theta + mu)
    log1m_p = jnp.log(mu) - jnp.log(theta + mu)

    return jsp.gammaln(theta + x) - jsp.gammaln(theta) - jsp.gammaln(x + 1) + theta * log_p + x * log1m_p


def _cdf(
    x: ArrayLike,
    mu: ArrayLike,
    theta: ArrayLike,
) -> Array:
    # Cast to Array
    x, mu, theta = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(theta)

    # Compute p
    p = theta / (theta + mu)

    return jsp.betainc(theta, jnp.floor(x) + 1, p)


def _logcdf(
    x: ArrayLike,
    mu: ArrayLike,
    theta: ArrayLike,
) -> Array:
    # Cast to Array
    x, mu, theta = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(theta)

    return jnp.log(_cdf(x, mu, theta))


def _ccdf(
    x: ArrayLike,
    mu: ArrayLike,
    theta: ArrayLike,
) -> Array:
    # Cast to Array
    x, mu, theta = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(theta)

    return 1.0 - _cdf(x, mu, theta)


def _logccdf(
    x: ArrayLike,
    mu: ArrayLike,
    theta: ArrayLike,
) -> Array:
    # Cast to Array
    x, mu, theta = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(theta)

    return jnp.log(_ccdf(x, mu, theta))


class MeanInvOverdispNegBinom(Parameterization):
    """
    A mean-inverse overdispersion parameterization of the negative binomial distribution.
    """

    mu: Array
    theta: Array

    def __init__(
        self,
        mu: ArrayLike,
        theta: ArrayLike
    ):
        for name, val in [("mu", mu), ("theta", theta)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        mu = self.mu
        theta = self.theta

        return _logprob(x, mu, theta)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        mu = self.mu
        theta = self.theta

        key1, key2 = jr.split(key)
        rate = jr.gamma(key1, theta, shape) * mu / theta
        return jr.poisson(key2, rate)
