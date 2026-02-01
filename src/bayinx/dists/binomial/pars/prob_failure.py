from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed


def _log_binom_coeff(n: ArrayLike, x: ArrayLike) -> Array:
    n, x = jnp.asarray(n), jnp.asarray(x)
    return jsp.gammaln(n + 1) - jsp.gammaln(x + 1) - jsp.gammaln(n - x + 1)


def _prob(
    x: ArrayLike,
    n: ArrayLike,
    q: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return jnp.exp(_logprob(x, n, q))


def _logprob(
    x: ArrayLike,
    n: ArrayLike,
    q: ArrayLike,
) -> Array:
    # Cast to Array
    k, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return _log_binom_coeff(n, k) + k * jnp.log1p(- q) + (n - k) * jnp.log(q)


def _cdf(
    x: ArrayLike,
    n: ArrayLike,
    q: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return jsp.betainc(n - x, x + 1, q)


def _logcdf(
    x: ArrayLike,
    n: ArrayLike,
    q: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return jnp.log(_cdf(x, n, q))


def _ccdf(
    x: ArrayLike,
    n: ArrayLike,
    q: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return jsp.betainc(x + 1, n - x, 1 - q)


def _logccdf(
    x: ArrayLike,
    n: ArrayLike,
    q: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return jnp.log(_ccdf(x, n, q))


class ProbFailureBinomial(Parameterization):
    """
    A probability-of-failure parameterization of the Binomial distribution.
    """

    n: Node[Array]
    q: Node[Array]

    def __init__(
        self,
        n: ArrayObject,
        q: ArrayObject
    ):
        for name, val in [("n", n), ("q", q)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        n = byo.obj(self.n)
        q = byo.obj(self.q)

        return _logprob(x, n, q)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        n = byo.obj(self.n)
        q = byo.obj(self.q)

        # Transform to probability of success
        p = 1.0 - q

        return jr.binomial(key, n, p, shape)
