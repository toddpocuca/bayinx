
from typing import Tuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, Integer, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed


def _log_binom_coeff(n: ArrayLike, x: ArrayLike) -> Array:
    n, x = jnp.asarray(n), jnp.asarray(x)
    return jsp.gammaln(n + 1) - jsp.gammaln(x + 1) - jsp.gammaln(n - x + 1)


def _prob(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return lax.exp(_logprob(x, n, q))


def _logprob(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    k, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return _log_binom_coeff(n, k) + k * lax.log1p(- q) + (n - k) * lax.log(q)


def _cdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return jsp.betainc(n - x, x + 1, q)


def _logcdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return lax.log(_cdf(x, n, q))


def _ccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return jsp.betainc(x + 1, n - x, 1 - q)


def _logccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    q: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, q = jnp.asarray(x), jnp.asarray(n), jnp.asarray(q)

    return lax.log(_ccdf(x, n, q))


class ProbFailureBinomial(Parameterization):
    """
    A probability-of-failure parameterization of the Binomial distribution.
    """

    n: Node[Integer[Array, "..."]]
    q: Node[Real[Array, "..."]]

    def __init__(
        self,
        n: Integer[ArrayLike, "..."] | Node[Integer[Array, "..."]],
        q: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize number of trials
        if isinstance(n, Node):
            if isinstance(n.obj, ArrayLike):
                self.n = n # type: ignore
        else:
            self.n = Observed(jnp.asarray(n))

        # Initialize probability of success
        if isinstance(q, Node):
            if isinstance(q.obj, ArrayLike):
                self.q = q # type: ignore
        else:
            self.q = Observed(jnp.asarray(q))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.n.obj, self.q.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.binomial(key, self.n.obj, self.q.obj, shape)
