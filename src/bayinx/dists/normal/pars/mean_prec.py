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
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return lax.sqrt(prec) / lax.sqrt(2.0 * PI) * lax.exp(-0.5 * prec * lax.square(x - mean))


def _logprob(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return 0.5 * lax.log(prec) - lax.log(lax.sqrt(2.0 * PI)) - 0.5 * prec * lax.square(x - mean)


def _cdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return jsp.ndtr((x - mean) * lax.sqrt(prec))


def _logcdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return jsp.log_ndtr((x - mean) * lax.sqrt(prec))


def _ccdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return jsp.ndtr((mean - x) * lax.sqrt(prec))


def _logccdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    prec: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, prec = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(prec)

    return jsp.log_ndtr((mean - x) * lax.sqrt(prec))


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
            if isinstance(mean.obj, ArrayLike):
                self.mean = mean # type: ignore
        else:
            self.mean = Observed(jnp.asarray(mean))

        # Initialize precision parameter
        if isinstance(prec, Node):
            if isinstance(prec.obj, ArrayLike):
                self.prec = prec # type: ignore
        else:
            self.prec = Observed(jnp.asarray(prec))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.mean.obj, self.prec.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.normal(key, shape) / lax.sqrt(self.prec.obj) + self.mean.obj
