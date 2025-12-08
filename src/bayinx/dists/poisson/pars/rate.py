from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, Integer, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


def _prob(
    x: Integer[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return lax.exp(_logprob(x, rate))


def _logprob(
    x: Integer[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return x * lax.log(rate) - rate - jsp.gammaln(x + 1)


def _cdf(
    x: Integer[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = jsp.gammaincc(x + 1.0, rate)
    result = lax.select(x < 0.0, 0.0, result)

    return result


def _logcdf(
    x: Integer[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return lax.log(_cdf(x,rate)) # TODO


def _ccdf(
    x: Integer[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    result = jsp.gammainc(x + 1.0, rate)
    result = lax.select(x < 0.0, 1.0, result)

    return result


def logccdf(
    x: Integer[ArrayLike, "..."],
    rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, rate = jnp.asarray(x), jnp.asarray(rate)

    return lax.log(_ccdf(x,rate))


class RatePoisson(Parameterization):
    """
    The rate parameterization of the Poisson distribution.

    # Attributes
    - `rate`: The rate parameter.
    """

    rate: Node[Real[Array, "..."]]

    def __init__(
        self,
        rate: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize rate parameter
        if isinstance(rate, Node):
            if isinstance(rate.obj, Array):
                self.rate = rate # type: ignore
        else:
            self.rate = Observed(jnp.asarray(rate))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.rate.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.poisson(key, self.rate.obj, shape)
