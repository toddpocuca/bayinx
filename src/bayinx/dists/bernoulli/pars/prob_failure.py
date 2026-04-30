from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


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

    q: Array

    def __init__(
        self,
        q: ArrayLike,
    ):
        for name, val in [("q", q)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract probability of failure
        q = self.q

        return _logprob(x, q)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract probability of success
        p = 1.0 - self.q

        return jr.bernoulli(key, p, shape)
