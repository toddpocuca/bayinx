from typing import Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
import jax.tree as jt
from jaxtyping import Array, ArrayLike, Float, Integer, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Distribution
from bayinx.core.node import Node
from bayinx.nodes import Observed


def log_binom_coeff(n: ArrayLike, x: ArrayLike) -> Array:
    n, x = jnp.asarray(n), jnp.asarray(x)
    return jsp.gammaln(n + 1) - jsp.gammaln(x + 1) - jsp.gammaln(n - x + 1)


def prob(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The probability mass function (PMF) for a Binomial distribution.

    # Parameters
    - `x`: Where to evaluate the PMF (number of successes).
    - `n`: The number of trials.
    - `p`: The probability of success.

    # Returns
    The PMF evaluated at `x`. The output will have the broadcasted shapes of `x`, `n`, and `p`.
    """
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return lax.exp(logprob(x, n, p))


def logprob(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the probability mass function (log PMF) for a Binomial distribution.

    # Parameters
    - `x`: Where to evaluate the log PMF (number of successes).
    - `n`: The number of trials.
    - `p`: The probability of success.

    # Returns
    The log PMF evaluated at `x`. The output will have the broadcasted shapes of `x`, `n`, and `p`.
    """
    # Cast to Array
    k, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return log_binom_coeff(n, k) + k * lax.log(p) + (n - k) * lax.log1p(-p)


def cdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The cumulative density function (CDF) for a Binomial distribution.

    # Parameters
    - `x`: Where to evaluate the CDF (P(X <= x)).
    - `n`: The number of trials.
    - `p`: The probability of success.

    # Returns
    The CDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `n`, and `p`.
    """
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return jsp.betainc(n - x, x + 1, 1.0 - p)


def logcdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the cumulative density function (log CDF) for a Binomial distribution.

    # Parameters
    - `x`: Where to evaluate the log CDF (ln P(X <= x)).
    - `n`: The number of trials.
    - `p`: The probability of success.

    # Returns
    The log CDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `n`, and `p`.
    """
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return lax.log(cdf(x, n, p))


def ccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The complementary cumulative density function (cCDF) for a Binomial distribution (P(X > k)).

    # Parameters
    - `k`: Where to evaluate the cCDF.
    - `n`: The number of trials.
    - `p`: The probability of success.

    # Returns
    The cCDF evaluated at `k`. The output will have the broadcasted shapes of `k`, `n`, and `p`.
    """

    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return jsp.betainc(x + 1, n - x, p)


def logccdf(
    x: Integer[ArrayLike, "..."],
    n: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the complementary cumulative density function (log cCDF) for a Binomial distribution.

    # Parameters
    - `x`: Where to evaluate the log cCDF.
    - `n`: The number of trials.
    - `p`: The probability of success.

    # Returns
    The log cCDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `n`, and `p`.
    """
    # Cast to Array
    x, n, p = jnp.asarray(x), jnp.asarray(n), jnp.asarray(p)

    return lax.log(ccdf(x, n, p))


class Binomial(Distribution):
    """
    A Binomial distribution.

    # Attributes
    - `n`: The number of trials parameter (integer).
    - `p`: The probability of success parameter (float, [0, 1]).
    """

    n: Node[Integer[Array, "..."]]
    p: Node[Real[Array, "..."]]


    def __init__(
        self,
        n: Integer[ArrayLike, "..."] | Node[Integer[Array, "..."]],
        p: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize number of trials (n)
        if isinstance(n, Node):
            if isinstance(n.obj, ArrayLike):
                self.n = n # type: ignore
        else:
            self.n = Observed(jnp.asarray(n))

        # Initialize probability of success parameter (p)
        if isinstance(p, Node):
            if isinstance(p.obj, ArrayLike):
                self.p = p # type: ignore
        else:
            self.p = Observed(jnp.asarray(p))

    def logprob(self, node: Node) -> Scalar:
        obj, n, p = (
            node.obj,
            self.n.obj,
            self.p.obj
        )

        # Filter out irrelevant values
        obj, _ = eqx.partition(obj, node._filter_spec)

        # Helper function for the single-leaf log-probability evaluation
        def leaf_logprob(k: Float[ArrayLike, "..."]) -> Scalar:
            return logprob(k, n, p).sum()

        # Compute log probabilities across leaves
        eval_obj = jt.map(leaf_logprob, obj)

        # Compute total sum
        total = jt.reduce_associative(lambda x,y: x + y, eval_obj, identity=0.0)

        return jnp.asarray(total)

    def sample(self, shape: int | Tuple[int, ...], key: PRNGKeyArray = jr.key(0)):
        # Coerce to tuple
        if isinstance(shape, int):
            shape = (shape, )

        # Use jr.binomial for sampling (returns integer array)
        # Note: jr.binomial accepts n as int, p as float, and shape
        return jr.binomial(key, self.n.obj, self.p.obj, shape=shape)
