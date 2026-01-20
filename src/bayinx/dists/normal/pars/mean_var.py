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
    var: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, var = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(var)

    return 1 / (jnp.sqrt(var) * jnp.sqrt(2.0 * PI)) * jnp.exp(-0.5 * jnp.square(x - mean) / var)


def _logprob(
    x: Real[ArrayLike, "..."],
    mean: Real[ArrayLike, "..."],
    var: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, mean, var = jnp.asarray(x), jnp.asarray(mean), jnp.asarray(var)

    return -jnp.log(jnp.sqrt(2.0 * PI)) - 0.5 * jnp.log(var) - 0.5 * jnp.square(x - mean) / jnp.sqrt(var)


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
            if isinstance(mean._byx__obj, ArrayLike):
                self.mean: Node[ArrayLike] = mean
        else:
            self.mean = Observed(jnp.asarray(mean))

        # Initialize scale parameter
        if isinstance(var, Node):
            if isinstance(var._byx__obj, ArrayLike):
                self.var: Node[ArrayLike] = var
        else:
            self.var = Observed(jnp.asarray(var))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        mean = byo.obj(self.mean)
        var = byo.obj(self.var)

        return _logprob(x, mean, var)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        mean = byo.obj(self.mean)
        var = byo.obj(self.var)

        return jr.normal(key, shape) * jnp.sqrt(var) + mean
