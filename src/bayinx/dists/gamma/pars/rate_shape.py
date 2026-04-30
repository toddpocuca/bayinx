from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jssp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    rate: ArrayLike,
    shape: ArrayLike
) -> Array:
    # Cast to Array
    x, rate, shape = jnp.asarray(x), jnp.asarray(rate), jnp.asarray(shape)

    return rate**shape * x**(shape - 1) * jnp.exp(-rate * x) / jssp.gamma(shape)


def _logprob(
    x: ArrayLike,
    rate: ArrayLike,
    shape: ArrayLike
) -> Array:
    # Cast to Array
    x, rate, shape = jnp.asarray(x), jnp.asarray(rate), jnp.asarray(shape)

    return shape * jnp.log(rate) + (shape - 1) * jnp.log(x) - rate * x - jssp.gammaln(shape)


def _cdf(
    x: ArrayLike,
    rate: ArrayLike,
    shape: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate, shape = jnp.asarray(x), jnp.asarray(rate), jnp.asarray(shape)

    result = jssp.gammainc(shape, rate * x)
    result = lax.select(x >= 0.0, result, 0.0)

    return result


def _logcdf(
    x: ArrayLike,
    rate: ArrayLike,
    shape: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate, shape = jnp.asarray(x), jnp.asarray(rate), jnp.asarray(shape)

    result = jnp.log(jssp.gammainc(shape, rate * x))
    result = lax.select(x >= 0.0, result, -jnp.inf)

    return result


def _ccdf(
    x: ArrayLike,
    rate: ArrayLike,
    shape: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate, shape = jnp.asarray(x), jnp.asarray(rate), jnp.asarray(shape)

    # Regularized upper incomplete gamma function
    result = jssp.gammaincc(shape, rate * x)
    result = lax.select(x >= 0.0, result, 1.0)

    return result


def _logccdf(
    x: ArrayLike,
    rate: ArrayLike,
    shape: ArrayLike,
) -> Array:
    # Cast to Array
    x, rate, shape = jnp.asarray(x), jnp.asarray(rate), jnp.asarray(shape)

    result = jnp.log(jssp.gammaincc(shape, rate * x))
    result = lax.select(x >= 0.0, result, 0.0)

    return result


class RateShapeGamma(Parameterization):
    """
    The rate-shape parameterization of the Gamma distribution.

    # Attributes
    - `rate`: The rate parameter.
    - `shape`: The shape parameter.
    """

    rate: Array
    shape: Array

    def __init__(
        self,
        rate: ArrayLike,
        shape: ArrayLike,
    ):
        for name, val in [("rate", rate), ("shape", shape)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        rate = self.rate
        shape = self.shape

        return _logprob(x, rate, shape)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        rate = self.rate
        shp = self.shape

        return jr.gamma(key, shp, shape=shape) / rate
