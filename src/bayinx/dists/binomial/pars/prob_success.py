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
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return lax.exp(_logprob(x, n, p))


def _logprob(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    k, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return _log_binom_coeff(n, k) + k * lax.log(p) + (n - k) * lax.log1p(-p)


def _cdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return jsp.betainc(n - x, x + 1, 1.0 - p)


def _logcdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return lax.log(_cdf(x, n, p)) # TODO


def _ccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return jsp.betainc(x + 1, n - x, p)


def _logccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return lax.log(_ccdf(x, n, p)) # TODO


class ProbSuccessBinomial(Parameterization):
    """
    A probability-of-success parameterization of the Binomial distribution.
    """

    n: Node[Integer[Array, "..."]]
    p: Node[Real[Array, "..."]]

    def __init__(
        self,
        n: Integer[ArrayLike, "..."] | Node[Integer[Array, "..."]],
        p: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize number of trials
        if isinstance(n, Node):
            if isinstance(n.obj, ArrayLike):
                self.n = n # type: ignore
        else:
            self.n = Observed(jnp.asarray(n))

        # Initialize probability of success
        if isinstance(p, Node):
            if isinstance(p.obj, ArrayLike):
                self.p = p # type: ignore
        else:
            self.p = Observed(jnp.asarray(p))

    def logprob(self, x: ArrayLike) -> Scalar:
        return _logprob(x, self.n.obj, self.p.obj)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        return jr.binomial(key, self.n.obj, self.p.obj, shape)
