from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, Integer, PRNGKeyArray, Real, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


def _log_binom_coeff(n: ArrayLike, x: ArrayLike) -> Array:
    n, x = jnp.asarray(n), jnp.asarray(x)
    return jsp.gammaln(n + 1) - jsp.gammaln(x + 1) - jsp.gammaln(n - x + 1)


def _prob(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """Probability mass function, P(X=x | n, logit_q)."""
    # Cast to Array
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jnp.exp(_logprob(x, n, logit_q))


def _logprob(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    k, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return _log_binom_coeff(n, k) + k * jnn.log_sigmoid(-logit_q) + (n - k) * jnn.log_sigmoid(logit_q)


def _cdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jsp.betainc(n - x, x + 1, jnn.sigmoid(logit_q))


def _logcdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jnp.log(_cdf(x, n, logit_q)) # TODO


def _ccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jsp.betainc(x + 1, n - x, jnn.sigmoid(-logit_q))


def _logccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jnp.log(_ccdf(x, n, logit_q)) # TODO


class LogitProbFailureBinomial(Parameterization):
    """
    A logit-of-probability-of-failure parameterization of the Binomial distribution.
    """

    n: Node[Integer[Array, "..."]]
    logit_q: Node[Real[Array, "..."]]

    def __init__(
        self,
        n: Integer[ArrayLike, "..."] | Node[Integer[Array, "..."]],
        logit_q: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize number of trials
        if isinstance(n, Node):
            if isinstance(byo.obj(n), ArrayLike):
                self.n = n # type: ignore
        else:
            self.n = Observed(jnp.asarray(n))

        # Initialize logit of probability of failure
        if isinstance(logit_q, Node):
            if isinstance(byo.obj(logit_q), ArrayLike):
                self.logit_q = logit_q # type: ignore
        else:
            self.logit_q = Observed(jnp.asarray(logit_q))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        n = byo.obj(self.n)
        logit_q = byo.obj(self.logit_q)

        return _logprob(x, n, logit_q)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        n = byo.obj(self.n)
        logit_q = byo.obj(self.logit_q)

        # Transform to probability of success
        p = jnn.sigmoid(-logit_q)

        return jr.binomial(key, n, p, shape)
