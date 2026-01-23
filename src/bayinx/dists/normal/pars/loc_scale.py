from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed

PI = 3.141592653589793

def _prob(
    x: ArrayLike,
    mean: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return 1 / (scale * jnp.sqrt(2.0 * PI)) * jnp.exp(-0.5 * jnp.square((x - mean) / scale))


def _logprob(
    x: ArrayLike,
    mean: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    # Compute variance
    var = jnp.square(scale)

    return -0.5 * (jnp.log(2.0 * PI * var) + jnp.square(x - mean) / var)


def _cdf(
    x: ArrayLike,
    mean: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return jsp.ndtr((x - mean) / scale)


def _logcdf(
    x: ArrayLike,
    mean: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return jsp.log_ndtr((x - mean) / scale)


def _ccdf(
    x: ArrayLike,
    mean: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return jsp.ndtr((mean - x) / scale)


def _logccdf(
    x: ArrayLike,
    mean: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return jsp.log_ndtr((mean - x) / scale)



class LocScaleNormal(Parameterization):
    """
    A mean-scale parameterization of the normal distribution.
    """

    mean: Node[Array]
    scale: Node[Array]

    def __init__(
        self,
        mean: ArrayObject,
        scale: ArrayObject
    ):
        # Initialize mean parameter
        if isinstance(mean, Node):
            if isinstance(mean._byx__obj, ArrayLike):
                self.mean = mean # type: ignore
        else:
            self.mean = Observed(jnp.asarray(mean))

        # Initialize scale parameter
        if isinstance(scale, Node):
            if isinstance(scale._byx__obj, ArrayLike):
                self.scale = scale # type: ignore
        else:
            self.scale = Observed(jnp.asarray(scale))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        mean = byo.obj(self.mean)
        scale = byo.obj(self.scale)

        return _logprob(x, mean, scale)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        mean = byo.obj(self.mean)
        scale = byo.obj(self.scale)

        return jr.normal(key, shape) * scale + mean
