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
    p: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return jnp.exp(_logprob(x, n, p))


def _logprob(
    x: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
) -> Array:
    # Cast to Array
    k, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return _log_binom_coeff(n, k) + k * jnp.log(p) + (n - k) * jnp.log1p(-p)


def _cdf(
    x: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return jsp.betainc(n - x, x + 1, 1.0 - p)


def _logcdf(
    x: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return jnp.log(_cdf(x, n, p)) # TODO


def _ccdf(
    x: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return jsp.betainc(x + 1, n - x, p)


def _logccdf(
    x: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
) -> Array:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return jnp.log(_ccdf(x, n, p)) # TODO


class ProbSuccessBinomial(Parameterization):
    """
    A probability-of-success parameterization of the Binomial distribution.
    """

    n: Node[Array]
    p: Node[Array]

    def __init__(
        self,
        n: ArrayObject,
        p: ArrayObject
    ):
        for name, val in [("n", n), ("p", p)]:
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
        p = byo.obj(self.p)

        return _logprob(x, n, p)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        n = byo.obj(self.n)
        p = byo.obj(self.p)

        return jr.binomial(key, n, p, shape)
