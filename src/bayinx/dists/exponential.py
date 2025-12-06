from typing import Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Distribution
from bayinx.core.node import Node
from bayinx.nodes import Observed


def prob(
    x: Real[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The probability density function (PDF) for an Exponential distribution.

    # Parameters
    - `x`: Where to evaluate the PDF (must be >= 0).
    - `lam`: The rate parameter (lambda), must be > 0.

    # Returns
    The PDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    return lam * lax.exp(-lam * x)


def logprob(
    x: Real[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the probability density function (log PDF) for an Exponential distribution.

    # Parameters
    - `x`: Where to evaluate the log PDF (must be >= 0).
    - `lam`: The rate parameter (lambda), must be > 0.

    # Returns
    The log PDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    return lax.log(lam) - lam * x

def cdf(
    x: Real[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The cumulative density function (CDF) for an Exponential distribution (P(X <= x)).

    # Parameters
    - `x`: Where to evaluate the CDF.
    - `lam`: The rate parameter (lambda).

    # Returns
    The CDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    result = 1.0 - lax.exp(-lam * x)

    # Handle values outside of support
    result = lax.select(x >= 0, result, jnp.array(0.0))

    return result

def logcdf(
    x: Real[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the cumulative density function (log CDF) for an Exponential distribution.

    # Parameters
    - `x`: Where to evaluate the log CDF (ln P(X <= x)).
    - `lam`: The rate parameter (lambda).

    # Returns
    The log CDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    result = lax.log1p(-lax.exp(-lam * x))

    # Handle values outside of support (x < 0)
    result = lax.select(x >= 0, result, -jnp.inf)

    return result


def ccdf(
    x: Real[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The complementary cumulative density function (cCDF) for an Exponential distribution (P(X > x)).

    # Parameters
    - `x`: Where to evaluate the cCDF.
    - `lam`: The rate parameter (lambda).

    # Returns
    The cCDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    result = lax.exp(-lam * x)

    # Handle values outside of support
    result = lax.select(x >= 0, result, jnp.array(1.0))

    return result


def logccdf(
    x: Real[ArrayLike, "..."],
    lam: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the complementary cumulative density function (log cCDF) for an Exponential distribution.

    # Parameters
    - `x`: Where to evaluate the log cCDF.
    - `lam`: The rate parameter (lambda).

    # Returns
    The log cCDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `lam`.
    """
    # Cast to Array
    x, lam = jnp.asarray(x), jnp.asarray(lam)

    # log(cCDF(x)) = -lambda * x for x >= 0
    log_ccdf_val = -lam * x

    # Handle values outside of support (x < 0), log(P(X > x)) = log(1.0) = 0.0
    return lax.select(x >= 0, log_ccdf_val, jnp.array(0.0))


class Exponential(Distribution):
    """
    An Exponential distribution.

    # Attributes
    - `lam`: The rate parameter (lambda), must be positive.
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
        def leaf_logprob(x: Real[ArrayLike, "..."]) -> Scalar:
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

        return jr.exponential(key, shape=shape) *  1.0 / self.lam.obj
