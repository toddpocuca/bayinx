from typing import Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
import jax.tree as jt
from jaxtyping import Array, ArrayLike, Integer, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Distribution
from bayinx.core.node import Node
from bayinx.nodes import Observed


def prob(
    x: Integer[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The probability mass function (PMF) for a Poisson distribution.

    # Parameters
    - `x`: Where to evaluate the PMF.
    - `lam`: The rate parameter (lambda), representing the average number of events.

    # Returns
    The PMF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    return lax.exp(logprob(x, lam))


def logprob(
    x: Integer[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the probability mass function (log PMF) for a Poisson distribution.

    # Parameters
    - `x`: Where to evaluate the log PMF.
    - `lam`: The rate parameter (lambda), representing the average number of events.

    # Returns
    The log PMF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    return x * lax.log(lam) - lam - jsp.gammaln(x + 1)


def cdf(
    x: Integer[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The cumulative density function (CDF) for a Poisson distribution (P(X <= x)).

    # Parameters
    - `x`: Where to evaluate the CDF.
    - `lam`: The rate parameter (lambda).

    # Returns
    The CDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    result = jsp.gammainc(x + 1.0, lam)
    result = lax.select(x < 0, jnp.array(0.0), result)

    return result


def logcdf(
    x: Integer[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the cumulative density function (log CDF) for a Poisson distribution.

    # Parameters
    - `x`: Where to evaluate the log CDF (ln P(X <= x)).
    - `lam`: The rate parameter (lambda).

    # Returns
    The log CDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    result = lax.log(jsp.gammainc(x + 1.0, lam))

    # Handle values outside of support
    result = lax.select(x < 0, -jnp.inf, result)

    return result


def ccdf(
    x: Integer[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The complementary cumulative density function (cCDF) for a Poisson distribution (P(X > x)).

    # Parameters
    - `x`: Where to evaluate the cCDF.
    - `lam`: The rate parameter (lambda).

    # Returns
    The cCDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    result = jsp.gammaincc(x + 1.0, lam)

    # Handle values outside of support
    result = lax.select(x < 0, jnp.array(1.0), result)

    return result


def logccdf(
    x: Integer[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the complementary cumulative density function (log cCDF) for a Poisson distribution.

    # Parameters
    - `x`: Where to evaluate the log cCDF.
    - `lam`: The rate parameter (lambda).

    # Returns
    The log cCDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    result = lax.log(jsp.gammaincc(x + 1.0, lam))

    # Handle values outside of support
    result = lax.select(x < 0, jnp.array(0.0), result)

    return result


class Poisson(Distribution):
    """
    A Poisson distribution.

    # Attributes
    - `lam`: The rate parameter (lambda), representing the average number of events in an interval.
    """

    lam: Node[Real[Array, "..."]]


    def __init__(
        self,
        lam: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
    ):
        # Initialize rate parameter (lambda)
        if isinstance(lam, Node):
            if isinstance(lam.obj, ArrayLike):
                self.lam = lam # type: ignore
        else:
            self.lam = Observed(jnp.asarray(lam))


    def logprob(self, node: Node) -> Scalar:
        obj, lam = (
            node.obj,
            self.lam.obj,
        )

        # Filter out irrelevant values
        obj, _ = eqx.partition(obj, node._filter_spec)

        # Helper function for the single-leaf log-probability evaluation
        def leaf_logprob(x: Integer[ArrayLike, "..."]) -> Scalar:
            return logprob(x, lam).sum()

        # Compute log probabilities across leaves
        eval_obj = jt.map(leaf_logprob, obj)

        # Compute total sum
        total = jt.reduce_associative(lambda x,y: x + y, eval_obj, identity=0.0)

        return jnp.asarray(total)

    def sample(self, shape: int | Tuple[int, ...], key: PRNGKeyArray = jr.key(0)):
        # Coerce to tuple
        if isinstance(shape, int):
            shape = (shape, )

        return jr.poisson(key, self.lam.obj, shape=shape)
