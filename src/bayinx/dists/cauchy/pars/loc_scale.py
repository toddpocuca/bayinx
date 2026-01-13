from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Real, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed

PI = 3.141592653589793

def _prob(
    x: Real[ArrayLike, "..."],
    loc: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return 1.0 / (PI * scale * (1.0 + jnp.square((x - loc) / scale)))


def _logprob(
    x: Real[ArrayLike, "..."],
    loc: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return -jnp.log(PI) - jnp.log(scale) - jnp.log1p(jnp.square((x - loc) / scale))


def _cdf(
    x: Real[ArrayLike, "..."],
    loc: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jnp.arctan((x - loc) / scale) / PI + 0.5


def _logcdf(
    x: Real[ArrayLike, "..."],
    loc: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    return jnp.log(_cdf(x, loc, scale))


def _ccdf(
    x: Real[ArrayLike, "..."],
    loc: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jnp.arctan((loc - x) / scale) / PI + 0.5


def _logccdf(
    x: Real[ArrayLike, "..."],
    loc: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    return jnp.log(_ccdf(x, loc, scale))


class LocationScaleCauchy(Parameterization):
    """
    A location-scale parameterization of the Cauchy distribution.
    """

    loc: Node[Real[Array, "..."]]
    scale: Node[Real[Array, "..."]]

    def __init__(
        self,
        loc: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        scale: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize location parameter
        if isinstance(loc, Node):
            if isinstance(loc.obj, ArrayLike):
                self.loc: Node[Real[Array, "..."]] = loc # type: ignore
        else:
            self.loc = Observed(jnp.asarray(loc))

        # Initialize scale parameter
        if isinstance(scale, Node):
            if isinstance(scale.obj, ArrayLike):
                self.scale: Node[Real[Array, "..."]] = scale # type: ignore
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

        return jr.cauchy(key, shape) * scale + loc
