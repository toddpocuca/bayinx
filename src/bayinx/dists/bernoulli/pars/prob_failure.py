from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Integer, PRNGKeyArray, Real, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


def _prob(
    x: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return jnp.exp(_logprob(x,q))


def _logprob(
    x: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return x * jnp.log1p(q) + (1.0 - x) * jnp.log(q)


def _cdf(
    x: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, q, 1.0)
    )


def _logcdf(
    x: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return jnp.where(
        x < 0.0,
        -jnp.inf,
        jnp.where(x < 1.0, jnp.log(q), 0.0)
    )


def _ccdf(
    x: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return jnp.where(
        x < 0.0,
        1.0,
        jnp.where(x < 1.0, 1 - q, 0.0)
    )


def _logccdf(
    x: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, jnp.log1p(-q), -jnp.inf)
    )


class ProbFailureBernoulli(Parameterization):
    """
    A probability-of-failure parameterization of the Bernoulli distribution.
    """

    q: Node[Real[Array, "..."]]

    def __init__(
        self,
        q: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
    ):
        # Initialize probability of success
        if isinstance(q, Node):
            if isinstance(byo.obj(q), ArrayLike):
                self.q = q # type: ignore
        else:
            self.q = Observed(jnp.asarray(q))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract probability of failure
        q = byo.obj(self.q)

        return _logprob(x, q)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract probability of success
        p = 1.0 - byo.obj(self.q)

        return jr.bernoulli(key, p, shape)
