from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


def _prob(
    x: Real[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return rate * lax.exp(-rate * x)


def _logprob(
    x: Real[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return lax.log(rate) - rate * x

def _cdf(
    x: Real[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = 1.0 - lax.exp(-rate * x)
    result = lax.select(x >= 0, result, jnp.array(0.0))

    return result

def _logcdf(
    x: Real[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = lax.log1p(-lax.exp(-rate * x))

    # Handle values outside of support (x < 0)
    result = lax.select(x >= 0, result, -jnp.inf)

    return result


def _ccdf(
    x: Real[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = lax.exp(-rate * x)
    result = lax.select(x >= 0, result, jnp.array(1.0))

    return result


def _logccdf(
    x: Real[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = -rate * x
    result = lax.select(x >= 0, result, jnp.array(0.0))

    return result


class RateExponential(Parameterization):
    """
    The rate parameterization of the Exponential distribution.

    # Attributes
    - `rate`: The rate parameter.
    """

    rate: Node[Real[Array, "..."]]


    def __init__(
        self,
        rate: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
    ):
        # Initialize rate parameter (ratebda)
        if isinstance(rate, Node):
            if isinstance(rate.obj, ArrayLike):
                self.rate = rate # type: ignore
        else:
            self.rate = Observed(jnp.asarray(rate))


    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.rate.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.exponential(key, shape) / self.rate.obj
