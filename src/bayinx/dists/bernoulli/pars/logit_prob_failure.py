from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


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

    logit_q: Node[Array]

    def __init__(
        self,
        logit_q: ArrayLike | Node[Array],
    ):
        # Initialize probability of success
        if isinstance(logit_q, Node):
            if isinstance(byo.obj(logit_q), ArrayLike):
                self.logit_q = logit_q # type: ignore
        else:
            self.logit_q = Observed(jnp.asarray(logit_q))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract logit probability of failure
        logit_q = byo.obj(self.logit_q)

        return _logprob(x, logit_q)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract probability of success
        p = jnn.sigmoid(-byo.obj(self.logit_q))

        return jr.bernoulli(key, p, shape)
