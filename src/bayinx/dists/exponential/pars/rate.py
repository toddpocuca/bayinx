import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return rate * jnp.exp(-rate * x)


def _logprob(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return jnp.log(rate) - rate * x

def _cdf(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = 1.0 - jnp.exp(-rate * x)
    result = lax.select(x >= 0, result, jnp.array(0.0))

    return result

def _logcdf(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = jnp.log1p(-jnp.exp(-rate * x))
    result = lax.select(x >= 0.0, result, -jnp.inf)

    return result


def _ccdf(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = jnp.exp(-rate * x)
    result = lax.select(x >= 0.0, result, 1.0)

    return result


def _logccdf(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = -rate * x
    result = lax.select(x >= 0.0, result, 0.0)

    return result


class RateExponential(Parameterization):
    """
    The rate parameterization of the Exponential distribution.

    # Attributes
    - `rate`: The rate parameter.
    """

    rate: Array


    def __init__(
        self,
        rate: ArrayLike,
    ):
        for name, val in [("rate", rate)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)


    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameter
        rate = self.rate

        return _logprob(x, rate)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameter
        rate = self.rate

        return jr.exponential(key, shape) / rate
