from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Integer, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


def _prob(
    x: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return lax.exp(_logprob(x,q))


def _logprob(
    x: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return x * lax.log1p(q) + (1.0 - x) * lax.log1p(q)


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
        jnp.where(x < 1.0, lax.log(q), 0.0)
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
        jnp.where(x < 1.0, lax.log1p(-q), -jnp.inf)
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
            if isinstance(q.obj, ArrayLike):
                self.q = q # type: ignore
        else:
            self.q = Observed(jnp.asarray(q))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.q.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.bernoulli(key, self.q.obj, shape)
