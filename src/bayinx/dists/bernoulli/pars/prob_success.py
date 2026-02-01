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
    p: ArrayLike
) -> Array:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.exp(_logprob(x,p))


def _logprob(
    x: ArrayLike,
    p: ArrayLike
) -> Array:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return x * jnp.log(p) + (1.0 - x) * jnp.log1p(-p)


def _cdf(
    x: ArrayLike,
    p: ArrayLike
) -> Array:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, 1.0 - p, 1.0)
    )


def _logcdf(
    x: ArrayLike,
    p: ArrayLike
) -> Array:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        -jnp.inf,
        jnp.where(x < 1.0, jnp.log1p(-p), 0.0)
    )


def _ccdf(
    x: ArrayLike,
    p: ArrayLike
) -> Array:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        1.0,
        jnp.where(x < 1.0, p, 0.0)
    )


def _logccdf(
    x: ArrayLike,
    p: ArrayLike
) -> Array:
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, jnp.log(p), -jnp.inf)
    )


class ProbSuccessBernoulli(Parameterization):
    """
    A probability-of-success parameterization of the Bernoulli distribution.
    """

    p: Node[Array]

    def __init__(
        self,
        p: ArrayObject,
    ):
        for name, val in [("p", p)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract probability of success
        p = byo.obj(self.p)

        return _logprob(x, p)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract probability of success
        p = byo.obj(self.p)

        return jr.bernoulli(key, p, shape)
