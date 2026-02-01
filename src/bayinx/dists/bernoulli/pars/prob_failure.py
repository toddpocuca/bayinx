from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed


def _prob(
    x: ArrayLike,
    q: ArrayLike
) -> Array:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return jnp.exp(_logprob(x,q))


def _logprob(
    x: ArrayLike,
    q: ArrayLike
) -> Array:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return x * jnp.log1p(q) + (1.0 - x) * jnp.log(q)


def _cdf(
    x: ArrayLike,
    q: ArrayLike
) -> Array:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, q, 1.0)
    )


def _logcdf(
    x: ArrayLike,
    q: ArrayLike
) -> Array:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return jnp.where(
        x < 0.0,
        -jnp.inf,
        jnp.where(x < 1.0, jnp.log(q), 0.0)
    )


def _ccdf(
    x: ArrayLike,
    q: ArrayLike
) -> Array:
    # Cast to Array
    x, q = jnp.asarray(x), jnp.asarray(q)

    return jnp.where(
        x < 0.0,
        1.0,
        jnp.where(x < 1.0, 1 - q, 0.0)
    )


def _logccdf(
    x: ArrayLike,
    q: ArrayLike
) -> Array:
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

    q: Node[Array]

    def __init__(
        self,
        q: ArrayObject,
    ):
        for name, val in [("q", q)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract probability of failure
        q = byo.obj(self.q)

        return _logprob(x, q)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract probability of success
        p = 1.0 - byo.obj(self.q)

        return jr.bernoulli(key, p, shape)
