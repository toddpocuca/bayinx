from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed


def _log_binom_coeff(n: ArrayLike, x: ArrayLike) -> Array:
    n, x = jnp.asarray(n), jnp.asarray(x)
    return jsp.gammaln(n + 1) - jsp.gammaln(x + 1) - jsp.gammaln(n - x + 1)


def _prob(
    x: ArrayLike,
    n: ArrayLike,
    logit_p: ArrayLike,
) -> Array:
    """Probability mass function, P(X=x | n, logit_p)."""
    # Cast to Array
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jnp.exp(_logprob(x, n, logit_p))


def _logprob(
    x: ArrayLike,
    n: ArrayLike,
    logit_p: ArrayLike,
) -> Array:
    # Cast to Array
    k, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return _log_binom_coeff(n, k) + k * jnn.log_sigmoid(logit_p) + (n - k) * jnp.log(jnn.sigmoid(-logit_p))


def _cdf(
    x: ArrayLike,
    n: ArrayLike,
    logit_p: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jsp.betainc(n - x, x + 1, 1.0 - jnn.sigmoid(logit_p))


def _logcdf(
    x: ArrayLike,
    n: ArrayLike,
    logit_p: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jnp.log(_cdf(x, n, logit_p)) # TODO


def _ccdf(
    x: ArrayLike,
    n: ArrayLike,
    logit_p: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jsp.betainc(x + 1, n - x, jnn.sigmoid(logit_p))


def _logccdf(
    x: ArrayLike,
    n: ArrayLike,
    logit_p: ArrayLike,
) -> Array:
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jnp.log(_ccdf(x, n, logit_p)) # TODO


class LogitProbSuccessBinomial(Parameterization):
    """
    A logit-of-probability-of-success parameterization of the Binomial distribution.
    """

    n: Node[Array]
    logit_p: Node[Array]

    def __init__(
        self,
        n: ArrayObject,
        logit_p: ArrayObject
    ):
        for name, val in [("n", n), ("logit_p", logit_p)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        n = byo.obj(self.n)
        logit_p = byo.obj(self.logit_p)

        return _logprob(x, n, logit_p)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        n = byo.obj(self.n)
        logit_p = byo.obj(self.logit_p)

        # Transform to probability of success
        p = jnn.sigmoid(logit_p)

        return jr.binomial(key, n, p, shape)
