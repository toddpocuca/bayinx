import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jssp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    mean: ArrayLike,
    shape: ArrayLike
) -> Array:
    # Cast to Array
    x, mean, shape = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(shape)
    rate = shape / mean

    return rate**shape * x**(shape - 1) * jnp.exp(-rate * x) / jssp.gamma(shape)


def _logprob(
    x: ArrayLike,
    mean: ArrayLike,
    shape: ArrayLike
) -> Array:
    # Cast to Array
    x, mean, shape = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(shape)
    rate = shape / mean

    return shape * jnp.log(rate) + (shape - 1) * jnp.log(x) - rate * x - jssp.gammaln(shape)


def _cdf(
    x: ArrayLike,
    mean: ArrayLike,
    shape: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, shape = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(shape)
    rate = shape / mean

    result = jssp.gammainc(shape, rate * x)
    result = lax.select(x >= 0.0, result, 0.0)

    return result


def _logcdf(
    x: ArrayLike,
    mean: ArrayLike,
    shape: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, shape = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(shape)
    rate = shape / mean

    result = jnp.log(jssp.gammainc(shape, rate * x))
    result = lax.select(x >= 0.0, result, -jnp.inf)

    return result


def _ccdf(
    x: ArrayLike,
    mean: ArrayLike,
    shape: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, shape = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(shape)
    rate = shape / mean

    result = jssp.gammaincc(shape, rate * x)
    result = lax.select(x >= 0.0, result, 1.0)

    return result


def _logccdf(
    x: ArrayLike,
    mean: ArrayLike,
    shape: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, shape = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(shape)
    rate = shape / mean

    result = jnp.log(jssp.gammaincc(shape, rate * x))
    result = lax.select(x >= 0.0, result, 0.0)

    return result


class MeanShapeGamma(Parameterization):
    """
    The mean-shape parameterization of the Gamma distribution.

    # Attributes
    - `mean`: The mean of the distribution.
    - `shape`: The shape parameter.
    """

    mean: Array
    shape: Array

    def __init__(
        self,
        mean: ArrayLike,
        shape: ArrayLike,
    ):
        for name, val in [("mean", mean), ("shape", shape)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        mean = self.mean
        shape = self.shape

        return _logprob(x, mean, shape)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        mean = self.mean
        shp = self.shape

        return jr.gamma(key, shp, shape=shape) * (mean / shp)
