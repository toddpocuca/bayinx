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
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jnp.sqrt(prec) / jnp.sqrt(2.0 * PI) * jnp.exp(-0.5 * prec * jnp.square(x - loc))


def _logprob(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return 0.5 * jnp.log(prec) - jnp.log(jnp.sqrt(2.0 * PI)) - 0.5 * prec * jnp.square(x - loc)


def _cdf(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jsp.ndtr((x - loc) * jnp.sqrt(prec))


def _logcdf(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jsp.log_ndtr((x - loc) * jnp.sqrt(prec))


def _ccdf(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jsp.ndtr((loc - x) * jnp.sqrt(prec))


def _logccdf(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jsp.log_ndtr((loc - x) * jnp.sqrt(prec))


class LocPrecisionNormal(Parameterization):
    """
    A loc-precision parameterization of the normal distribution.
    """

    loc: Node[Array]
    prec: Node[Array]

    def __init__(
        self,
        loc: ArrayObject,
        prec: ArrayObject
    ):
        # Initialize loc parameter
        if isinstance(loc, Node):
            if isinstance(loc._byx__obj, ArrayLike):
                self.loc = loc # type: ignore
        else:
            self.loc = Observed(jnp.asarray(loc))

        # Initialize precision parameter
        if isinstance(prec, Node):
            if isinstance(prec._byx__obj, ArrayLike):
                self.prec = prec # type: ignore
        else:
            self.prec = Observed(jnp.asarray(prec))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        loc = byo.obj(self.loc)
        prec = byo.obj(self.prec)

        return _logprob(x, loc, prec)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        loc = byo.obj(self.loc)
        prec = byo.obj(self.prec)

        return jr.normal(key, shape) / jnp.sqrt(prec) + loc
