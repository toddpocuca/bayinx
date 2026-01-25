
from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jssp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, Real, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed


def _prob(
    x: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike
) -> Float[Array, "..."]:
    # Cast to Array
    x, shape, scale = jnp.asarray(x), jnp.asarray(shape), jnp.asarray(scale)

    return (x**(shape - 1) * jnp.exp(-x / scale)) / (scale**shape * jssp.gamma(shape))


def _logprob(
    x: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike
) -> Real[Array, "..."]:
    # Cast to Array
    x, shape, scale = jnp.asarray(x), jnp.asarray(shape), jnp.asarray(scale)

    return (shape - 1) * jnp.log(x) - (x / scale) - (shape * jnp.log(scale)) - jssp.gammaln(shape)


def _cdf(
    x: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, shape, scale = jnp.asarray(x), jnp.asarray(shape), jnp.asarray(scale)

    result = jssp.gammainc(shape, x / scale)
    result = lax.select(x >= 0.0, result, 0.0)

    return result


def _logcdf(
    x: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, shape, scale = jnp.asarray(x), jnp.asarray(shape), jnp.asarray(scale)

    result = jnp.log(jssp.gammainc(shape, x / scale))
    result = lax.select(x >= 0.0, result, -jnp.inf)

    return result


def _ccdf(
    x: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, shape, scale = jnp.asarray(x), jnp.asarray(shape), jnp.asarray(scale)

    result = jssp.gammaincc(shape, x / scale)
    result = lax.select(x >= 0.0, result, 1.0)

    return result


def _logccdf(
    x: ArrayLike,
    shape: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, shape, scale = jnp.asarray(x), jnp.asarray(shape), jnp.asarray(scale)

    result = jnp.log(jssp.gammaincc(shape, x / scale))
    result = lax.select(x >= 0.0, result, 0.0)

    return result

class ScaleShapeGamma(Parameterization):
    """
    The shape-scale parameterization of the Gamma distribution.

    # Attributes
    - `shape`: The shape parameter (k).
    - `scale`: The scale parameter (theta).
    """

    shape: Node[Array]
    scale: Node[Array]

    def __init__(
        self,
        scale: ArrayObject,
        shape: ArrayObject
    ):
        for name, val in [("scale", scale), ("shape", shape)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        shape = byo.obj(self.shape)
        scale = byo.obj(self.scale)

        return _logprob(x, shape, scale)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        shp = byo.obj(self.shape)
        scale = byo.obj(self.scale)

        return jr.gamma(key, shp, shape=shape) * scale
