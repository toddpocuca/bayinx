from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _log_binom_coeff(n: ArrayLike, x: ArrayLike) -> Array:
    n, x = jnp.asarray(n), jnp.asarray(x)
    return jsp.gammaln(n + 1) - jsp.gammaln(x + 1) - jsp.gammaln(n - x + 1)


def _prob(
    x: ArrayLike,
    n: ArrayLike,
    logit_q: ArrayLike,
) -> Array:
    """Probability mass function, P(X=x | n, logit_q)."""
    # Cast to Array
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jnp.exp(_logprob(x, n, logit_q))


def _logprob(
    x: ArrayLike,
    n: ArrayLike,
    logit_q: ArrayLike,
) -> Array:
    # Cast to Array
    k, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return _log_binom_coeff(n, k) + k * jnn.log_sigmoid(-logit_q) + (n - k) * jnn.log_sigmoid(logit_q)


def _cdf(
    x: ArrayLike,
    n: ArrayLike,
    logit_q: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jsp.betainc(n - x, x + 1, jnn.sigmoid(logit_q))


def _logcdf(
    x: ArrayLike,
    n: ArrayLike,
    logit_q: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jnp.log(_cdf(x, n, logit_q)) # TODO


def _ccdf(
    x: ArrayLike,
    n: ArrayLike,
    logit_q: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jsp.betainc(x + 1, n - x, jnn.sigmoid(-logit_q))


def _logccdf(
    x: ArrayLike,
    n: ArrayLike,
    logit_q: ArrayLike,
) -> Array:
    x, n, logit_q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logit_q)

    return jnp.log(_ccdf(x, n, logit_q)) # TODO


class LogitProbFailureBinomial(Parameterization):
    """
    A logit-of-probability-of-failure parameterization of the Binomial distribution.
    """

    n: Array
    logit_q: Array

    def __init__(
        self,
        n: ArrayLike,
        logit_q: ArrayLike
    ):
        for name, val in [("n", n), ("logit_q", logit_q)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        n = self.n
        logit_q = self.logit_q

        return _logprob(x, n, logit_q)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        n = self.n
        logit_q = self.logit_q

        # Transform to probability of success
        p = jnn.sigmoid(-logit_q)

        return jr.binomial(key, n, p, shape)
