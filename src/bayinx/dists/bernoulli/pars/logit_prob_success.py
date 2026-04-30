from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    logit_p: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return jnp.exp(_logprob(x, logit_p))


def _logprob(
    x: ArrayLike,
    logit_p: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return x * jnn.log_sigmoid(logit_p) + (1.0 - x) * jnn.log_sigmoid(-logit_p)


def _cdf(
    x: ArrayLike,
    logit_p: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, jnn.sigmoid(-logit_p), 1.0)
    )


def _logcdf(
    x: ArrayLike,
    logit_p: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return jnp.where(
        x < 0.0,
        -jnp.inf,
        jnp.where(x < 1.0, jnn.log_sigmoid(-logit_p), 0.0)
    )


def _ccdf(
    x: ArrayLike,
    logit_p: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return jnp.where(
        x < 0.0,
        1.0,
        jnp.where(x < 1.0, jnn.sigmoid(logit_p), 0.0)
    )


def _logccdf(
    x: ArrayLike,
    logit_p: ArrayLike
) -> Array:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, jnn.log_sigmoid(logit_p), -jnp.inf)
    )


class LogitProbSuccessBernoulli(Parameterization):
    """
    A logit probability-of-success parameterization of the Bernoulli distribution.
    """

    logit_p: Array

    def __init__(
        self,
        logit_p: ArrayLike,
    ):
        for name, val in [("logit_p", logit_p)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Array:
        # Extract logit probability of success
        logit_p = self.logit_p

        return _logprob(x, logit_p)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray) -> Array:
        # Extract probability of success
        p = jnn.sigmoid(self.logit_p)

        return jr.bernoulli(key, p, shape)
