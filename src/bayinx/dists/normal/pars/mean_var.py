

from typing import Tuple

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
    var: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, var = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(var)

    return 1 / (jnp.sqrt(var) * jnp.sqrt(2.0 * PI)) * jnp.exp(-0.5 * (x - mean)**2 / var)


def _logprob(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    var: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, var = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(var)

    return -jnp.log(jnp.sqrt(2.0 * PI)) - 0.5 * jnp.log(var) - 0.5 * (x - mean)**2 / jnp.sqrt(var)


def _cdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    var: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, var = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(var)

    return jsp.ndtr((x - mean) / jnp.sqrt(var))


def _logcdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    var: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, var = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(var)

    return jsp.log_ndtr((x - mean) / jnp.sqrt(var))


def _ccdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    var: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, var = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(var)

    return jsp.ndtr((mean - x) / jnp.sqrt(var))


def _logccdf(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    var: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, var = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(var)

    return jsp.log_ndtr((mean - x) / jnp.sqrt(var))



class MeanVarNormal(Parameterization):
    """
    A mean-variance parameterization of the normal distribution.
    """

    mean: Node[Real[Array, "..."]]
    var: Node[Real[Array, "..."]]

    def __init__(
        self,
        mean: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        var: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize mean parameter
        if isinstance(mean, Node):
            if isinstance(mean.obj, ArrayLike):
                self.mean: Node[ArrayLike] = mean # type: ignore
        else:
            self.mean = Observed(jnp.asarray(mean))

        # Initialize scale parameter
        if isinstance(var, Node):
            if isinstance(var.obj, ArrayLike):
                self.var: Node[ArrayLike] = var # type: ignore
        else:
            self.var = Observed(jnp.asarray(var))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.mean.obj, self.var.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.normal(key, shape) * jnp.sqrt(self.var.obj) + self.mean.obj
