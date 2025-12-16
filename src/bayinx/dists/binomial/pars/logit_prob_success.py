from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, Integer, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


def _log_binom_coeff(n: ArrayLike, x: ArrayLike) -> Array:
    n, x = jnp.asarray(n), jnp.asarray(x)
    return jsp.gammaln(n + 1) - jsp.gammaln(x + 1) - jsp.gammaln(n - x + 1)


def _prob(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """Probability mass function, P(X=x | n, logit_p)."""
    # Cast to Array
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jnp.exp(_logprob(x, n, logit_p))


def _logprob(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    k, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return _log_binom_coeff(n, k) + k * jnn.log_sigmoid(logit_p) + (n - k) * jnp.log(jnn.sigmoid(-logit_p))


def _cdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jsp.betainc(n - x, x + 1, 1.0 - jnn.sigmoid(logit_p))


def _logcdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jnp.log(_cdf(x, n, logit_p)) # TODO


def _ccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jsp.betainc(x + 1, n - x, jnn.sigmoid(logit_p))


def _logccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    x, n, logit_p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_p)

    return jnp.log(_ccdf(x, n, logit_p)) # TODO


class LogitProbSuccessBinomial(Parameterization):
    """
    A logit-of-probability-of-success parameterization of the Binomial distribution.
    """

    n: Node[Integer[Array, "..."]]
    logit_p: Node[Real[Array, "..."]]

    def __init__(
        self,
        n: Integer[ArrayLike, "..."] | Node[Integer[Array, "..."]],
        logit_p: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize number of trials
        if isinstance(n, Node):
            if isinstance(n.obj, ArrayLike):
                self.n = n # type: ignore
        else:
            self.n = Observed(jnp.asarray(n))

        # Initialize logit of probability of success
        if isinstance(logit_p, Node):
            if isinstance(logit_p.obj, ArrayLike):
                self.logit_p = logit_p # type: ignore
        else:
            self.logit_p = Observed(jnp.asarray(logit_p))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.n.obj, self.logit_p.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.binomial(key, self.n.obj, jnn.sigmoid(self.logit_p.obj), shape)
