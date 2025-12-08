

from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed

PI = 3.141592653589793


def _prob(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return 1 / (scale * lax.sqrt(2.0 * PI)) * lax.exp(-0.5 * lax.square((x - mean) / scale))


def _logprob(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return -lax.log(lax.sqrt(2.0 * PI)) - lax.log(scale) - 0.5 * lax.square((x - mean) / scale)


def _cdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return jsp.ndtr((x - mean) / scale)


def _logcdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return jsp.log_ndtr((x - mean) / scale)


def _ccdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return jsp.ndtr((mean - x) / scale)


def _logccdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, scale = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(scale)

    return jsp.log_ndtr((mean - x) / scale)



class MeanScaleNormal(Parameterization):
    """
    A mean-scale parameterization of the normal distribution.
    """

    mean: Node[Real[Array, "..."]]
    scale: Node[Real[Array, "..."]]

    def __init__(
        self,
        mean: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        scale: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize mean parameter
        if isinstance(mean, Node):
            if isinstance(mean.obj, ArrayLike):
                self.mean = mean # type: ignore
        else:
            self.mean = Observed(jnp.asarray(mean))

        # Initialize scale parameter
        if isinstance(scale, Node):
            if isinstance(scale.obj, ArrayLike):
                self.scale = scale # type: ignore
        else:
            self.scale = Observed(jnp.asarray(scale))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.mean.obj, self.scale.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.normal(key, shape) * self.scale.obj + self.mean.obj
