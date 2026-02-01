import jax.numpy as jnp
import jax.random as jr
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

    return 1.0 / (PI * scale * (1.0 + jnp.square((x - loc) / scale)))


def _logprob(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return -jnp.log(PI) - jnp.log(scale) - jnp.log1p(jnp.square((x - loc) / scale))


def _cdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jnp.arctan((x - loc) / scale) / PI + 0.5


def _logcdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike
) -> Array:
    return jnp.log(_cdf(x, loc, scale))


def _ccdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jnp.arctan((loc - x) / scale) / PI + 0.5


def _logccdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike
) -> Array:
    return jnp.log(_ccdf(x, loc, scale))


class LocationScaleCauchy(Parameterization):
    """
    A location-scale parameterization of the Cauchy distribution.
    """

    loc: Node[Array]
    scale: Node[Array]

    def __init__(
        self,
        loc: ArrayObject,
        scale: ArrayObject
    ):
        for name, val in [("loc", loc), ("scale", scale)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        loc = byo.obj(self.loc)
        scale = byo.obj(self.scale)

        return _logprob(x, loc, scale)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        loc = byo.obj(self.loc)
        scale = byo.obj(self.scale)

        return jr.cauchy(key, shape) * scale + loc
