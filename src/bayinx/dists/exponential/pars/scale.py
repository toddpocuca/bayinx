from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Real, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


def _prob(
    x: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, scale = jnp.asarray(x), jnp.asarray(scale)

    return jnp.exp(-x / scale) / scale


def _logprob(
    x: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, scale = jnp.asarray(x), jnp.asarray(scale)

    return -jnp.log(scale) - x / scale


def _cdf(
    x: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, scale = jnp.asarray(x), jnp.asarray(scale)

    result = 1.0 - jnp.exp(-x / scale)
    result = lax.select(x >= 0, result, 0.0)

    return result

def _logcdf(
    x: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, scale = jnp.asarray(x), jnp.asarray(scale)

    result = jnp.log1p(-jnp.exp(-x / scale))
    result = lax.select(x >= 0.0, result, -jnp.inf)

    return result


def _ccdf(
    x: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, scale = jnp.asarray(x), jnp.asarray(scale)

    result = jnp.exp(-x / scale)
    result = lax.select(x >= 0.0, result, 1.0)

    return result


def _logccdf(
    x: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, scale = jnp.asarray(x), jnp.asarray(scale)

    result = -x / scale
    result = lax.select(x >= 0.0, result, 0.0)

    return result


class ScaleExponential(Parameterization):
    """
    The scale parameterization of the Exponential distribution.

    # Attributes
    - `scale`: The scale parameter.
    """

    scale: Node[Real[Array, "..."]]


    def __init__(
        self,
        scale: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
    ):
        # Initialize scale parameter
        if isinstance(scale, Node):
            if isinstance(scale.obj, ArrayLike):
                self.scale = scale # type: ignore
        else:
            self.scale = Observed(jnp.asarray(scale))


    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameter
        scale = byo.obj(self.scale)

        return _logprob(x, scale)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameter
        scale = byo.obj(self.scale)

        return jr.exponential(key, shape) * scale
