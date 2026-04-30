
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    log_rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    return jnp.exp(_logprob(x, log_rate))


def _logprob(
    x: ArrayLike,
    log_rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    return x * log_rate - jnp.exp(log_rate) - jsp.gammaln(x + 1)


def _cdf(
    x: ArrayLike,
    log_rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    result = jsp.gammaincc(x + 1.0, jnp.exp(log_rate))
    result = lax.select(x < 0.0, 0.0, result)

    return result


def _logcdf(
    x: ArrayLike,
    log_rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    return lax.log(_cdf(x, log_rate))


def _ccdf(
    x: ArrayLike,
    log_rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    result = jsp.gammainc(x + 1.0, jnp.exp(log_rate))
    result = lax.select(x < 0.0, 1.0, result)

    return result


def logccdf(
    x: ArrayLike,
    log_rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    return lax.log(_ccdf(x, log_rate))


class LogRatePoisson(Parameterization):
    """
    The log-rate parameterization of the Poisson distribution.

    # Attributes
    - `log_rate`: The log of the rate parameter.
    """

    log_rate: Array

    def __init__(
        self,
        log_rate: ArrayLike
    ):
        # Initialize parameters
        for name, val in [("log_rate", log_rate)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameter
        log_rate = self.log_rate

        return _logprob(x, log_rate)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameter
        log_rate = self.log_rate

        # Transform to rate
        rate = jnp.exp(log_rate)

        return jr.poisson(key, rate, shape)
