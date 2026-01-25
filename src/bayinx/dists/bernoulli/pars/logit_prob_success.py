from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Integer, PRNGKeyArray, Real

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


def _prob(
    x: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return jnp.exp(_logprob(x, logit_p))


def _logprob(
    x: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return x * jnn.log_sigmoid(logit_p) + (1.0 - x) * jnn.log_sigmoid(-logit_p)


def _cdf(
    x: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, jnn.sigmoid(-logit_p), 1.0)
    )


def _logcdf(
    x: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return jnp.where(
        x < 0.0,
        -jnp.inf,
        jnp.where(x < 1.0, jnn.log_sigmoid(-logit_p), 0.0)
    )


def _ccdf(
    x: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, logit_p = jnp.asarray(x), jnp.asarray(logit_p)

    return jnp.where(
        x < 0.0,
        1.0,
        jnp.where(x < 1.0, jnn.sigmoid(logit_p), 0.0)
    )


def _logccdf(
    x: Integer[ArrayLike, "..."],
    logit_p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
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

    logit_p: Node[Real[Array, "..."]]

    def __init__(
        self,
        logit_p: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
    ):
        # Initialize probability of success
        if isinstance(logit_p, Node):
            if isinstance(byo.obj(logit_p), ArrayLike):
                self.logit_p = logit_p # type: ignore
        else:
            self.logit_p = Observed(jnp.asarray(logit_p))

    def logprob(self, x: ArrayLike) -> Array:
        # Extract logit probability of success
        logit_p = byo.obj(self.logit_p)

        return _logprob(x, logit_p)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray) -> Array:
        # Extract probability of success
        p = jnn.sigmoid(byo.obj(self.logit_p))

        return jr.bernoulli(key, p, shape)
