from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return lax.exp(_logprob(x, rate))


def _logprob(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return x * lax.log(rate) - rate - jsp.gammaln(x + 1)


def _cdf(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = jsp.gammaincc(x + 1.0, rate)
    result = lax.select(x < 0.0, 0.0, result)

    return result


def _logcdf(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return lax.log(_cdf(x,rate)) # TODO


def _ccdf(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = jsp.gammainc(x + 1.0, rate)
    result = lax.select(x < 0.0, 1.0, result)

    return result


def logccdf(
    x: ArrayLike,
    rate: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return lax.log(_ccdf(x,rate))


class RatePoisson(Parameterization):
    """
    The rate parameterization of the Poisson distribution.

    # Attributes
    - `rate`: The rate parameter.
    """

    rate: Array

    def __init__(
        self,
        rate: ArrayLike
    ):
        # Initialize parameters
        for name, val in [("rate", rate)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameter
        rate = self.rate

        return _logprob(x, rate)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameter
        rate = self.rate

        return jr.poisson(key, rate, shape)
