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
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return 1 / (scale * jnp.sqrt(2.0 * PI)) * jnp.exp(-0.5 * jnp.square((x - loc) / scale))


def _logprob(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    # Compute variance
    var = jnp.square(scale)

    return -0.5 * (jnp.log(2.0 * PI * var) + jnp.square(x - loc) / var)


def _cdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jsp.ndtr((x - loc) / scale)


def _logcdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jsp.log_ndtr((x - loc) / scale)


def _ccdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jsp.ndtr((loc - x) / scale)


def _logccdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jsp.log_ndtr((loc - x) / scale)



class LocScaleNormal(Parameterization):
    """
    A loc-scale parameterization of the normal distribution.
    """

    loc: Node[Array]
    scale: Node[Array]

    def __init__(
        self,
        loc: ArrayObject,
        scale: ArrayObject
    ):
        # Initialize loc parameter
        if isinstance(loc, Node):
            if isinstance(loc._byx__obj, ArrayLike):
                self.loc = loc # type: ignore
        else:
            self.loc = Observed(jnp.asarray(loc))

        # Initialize scale parameter
        if isinstance(scale, Node):
            if isinstance(scale._byx__obj, ArrayLike):
                self.scale = scale # type: ignore
        else:
            self.scale = Observed(jnp.asarray(scale))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        loc = byo.obj(self.loc)
        scale = byo.obj(self.scale)

        return _logprob(x, loc, scale)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        loc = byo.obj(self.loc)
        scale = byo.obj(self.scale)

        return jr.normal(key, shape) * scale + loc
