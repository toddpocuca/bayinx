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
    log_rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    return lax.exp(_logprob(x, log_rate))


def _logprob(
    x: Integer[ArrayLike, "..."],
    log_rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    return x * log_rate - lax.exp(log_rate) - jsp.gammaln(x + 1)


def _cdf(
    x: Integer[ArrayLike, "..."],
    log_rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    result = jsp.gammaincc(x + 1.0, lax.exp(log_rate))
    result = lax.select(x < 0.0, 0.0, result)

    return result


def _logcdf(
    x: Integer[ArrayLike, "..."],
    log_rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    return lax.log(_cdf(x, log_rate))


def _ccdf(
    x: Integer[ArrayLike, "..."],
    log_rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    result = jsp.gammainc(x + 1.0, lax.exp(log_rate))
    result = lax.select(x < 0.0, 1.0, result)

    return result


def logccdf(
    x: Integer[ArrayLike, "..."],
    log_rate: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, log_rate = jnp.asarray(x), jnp.asarray(log_rate)

    return lax.log(_ccdf(x, log_rate))


class LogRatePoisson(Parameterization):
    """
    The log-rate parameterization of the Poisson distribution.

    # Attributes
    - `log_rate`: The log of the rate parameter.
    """

    log_rate: Node[Real[Array, "..."]]

    def __init__(
        self,
        log_rate: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize log_rate parameter
        if isinstance(log_rate, Node):
            if isinstance(log_rate.obj, ArrayLike):
                self.log_rate = log_rate # type: ignore
        else:
            self.log_rate = Observed(jnp.asarray(log_rate))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.log_rate.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # The jax.random.poisson function requires the rate parameter lambda.
        rate = lax.exp(self.log_rate.obj)
        return jr.poisson(key, rate, shape)
