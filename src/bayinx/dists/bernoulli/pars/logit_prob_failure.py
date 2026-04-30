from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    logit_q: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_q = jnp.asarray(x), jnp.asarray(logit_q)

    return jnp.exp(_logprob(x,logit_q))


def _logprob(
    x: ArrayLike,
    logit_q: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_q = jnp.asarray(x), jnp.asarray(logit_q)

    return x * jnn.log_sigmoid(-logit_q) + (1 - x) * jnn.log_sigmoid(logit_q)


def _cdf(
    x: ArrayLike,
    logit_q: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_q = jnp.asarray(x), jnp.asarray(logit_q)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, jnn.sigmoid(logit_q), 1.0)
    )


def _logcdf(
    x: ArrayLike,
    logit_q: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_q = jnp.asarray(x), jnp.asarray(logit_q)

    return jnp.where(
        x < 0.0,
        -jnp.inf,
        jnp.where(x < 1.0, jnn.log_sigmoid(logit_q), 0.0)
    )


def _ccdf(
    x: ArrayLike,
    logit_q: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_q = jnp.asarray(x), jnp.asarray(logit_q)

    return jnp.where(
        x < 0.0,
        1.0,
        jnp.where(x < 1.0, jnn.sigmoid(-logit_q), 0.0)
    )


def _logccdf(
    x: ArrayLike,
    logit_q: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_q = jnp.asarray(x), jnp.asarray(logit_q)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, jnn.log_sigmoid(-logit_q), -jnp.inf)
    )


class LogitProbFailureBernoulli(Parameterization):
    """
    A logit probability-of-failure parameterization of the Bernoulli distribution.
    """

    logit_q: Array

    def __init__(
        self,
        logit_q: ArrayLike,
    ):
        for name, val in [("logit_q", logit_q)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract logit probability of failure
        logit_q = self.logit_q

        return _logprob(x, logit_q)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract probability of success
        p = jnn.sigmoid(-self.logit_q)

        return jr.bernoulli(key, p, shape)
