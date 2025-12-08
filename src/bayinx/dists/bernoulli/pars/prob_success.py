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
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return lax.exp(_logprob(x,p))


def _logprob(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return x * lax.log(p) + (1.0 - x) * lax.log1p(-p)


def _cdf(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, 1.0 - p, 1.0)
    )


def _logcdf(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        -jnp.inf,
        jnp.where(x < 1.0, lax.log1p(-p), 0.0)
    )


def _ccdf(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        1.0,
        jnp.where(x < 1.0, p, 0.0)
    )


def _logccdf(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, lax.log(p), -jnp.inf)
    )


class ProbSuccessBernoulli(Parameterization):
    """
    A probability-of-success parameterization of the Bernoulli distribution.
    """

    p: Node[Real[Array, "..."]]

    def __init__(
        self,
        p: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
    ):
        # Initialize probability of success
        if isinstance(p, Node):
            if isinstance(p.obj, ArrayLike):
                self.p = p # type: ignore
        else:
            self.p = Observed(jnp.asarray(p))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.p.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.bernoulli(key, self.p.obj, shape)
