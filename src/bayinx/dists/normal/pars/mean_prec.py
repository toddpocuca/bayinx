from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Real, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed

PI = 3.141592653589793


def _prob(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return jnp.sqrt(prec) / jnp.sqrt(2.0 * PI) * jnp.exp(-0.5 * prec * jnp.square(x - mean))


def _logprob(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return 0.5 * jnp.log(prec) - jnp.log(jnp.sqrt(2.0 * PI)) - 0.5 * prec * jnp.square(x - mean)


def _cdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return jsp.ndtr((x - mean) * jnp.sqrt(prec))


def _logcdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return jsp.log_ndtr((x - mean) * jnp.sqrt(prec))


def _ccdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return jsp.ndtr((mean - x) * jnp.sqrt(prec))


def _logccdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return jsp.log_ndtr((mean - x) * jnp.sqrt(prec))


class MeanPrecisionNormal(Parameterization):
    """
    A mean-precision parameterization of the normal distribution.
    """

    mean: Node[Real[Array, "..."]]
    prec: Node[Real[Array, "..."]]

    def __init__(
        self,
        mean: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        prec: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize mean parameter
        if isinstance(mean, Node):
            if isinstance(mean._byx__obj, ArrayLike):
                self.mean = mean # type: ignore
        else:
            self.mean = Observed(jnp.asarray(mean))

        # Initialize precision parameter
        if isinstance(prec, Node):
            if isinstance(prec._byx__obj, ArrayLike):
                self.prec = prec # type: ignore
        else:
            self.prec = Observed(jnp.asarray(prec))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        mean = byo.obj(self.mean)
        prec = byo.obj(self.prec)

        return _logprob(x, mean, prec)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        mean = byo.obj(self.mean)
        prec = byo.obj(self.prec)

        return jr.normal(key, shape) / jnp.sqrt(prec) + mean
