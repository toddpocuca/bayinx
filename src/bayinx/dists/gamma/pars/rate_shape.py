from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jssp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed


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

    rate: Node[Array]
    shape: Node[Array]

    def __init__(
        self,
        rate: ArrayObject,
        shape: ArrayObject,
    ):
        for name, val in [("rate", rate), ("shape", shape)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        rate = byo.obj(self.rate)
        shape = byo.obj(self.shape)

        return _logprob(x, rate, shape)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        rate = byo.obj(self.rate)
        shp = byo.obj(self.shape)

        return jr.gamma(key, shp, shape=shape) / rate
