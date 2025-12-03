from typing import Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, ArrayLike, Float, Integer, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Distribution
from bayinx.core.node import Node
from bayinx.nodes import Observed


def prob(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    """
    The probability mass function (PMF) for a Bernoulli distribution.

    # Parameters
    - `x`: Where to evaluate the PMF (pst be 0 or 1).
    - `p`: The probability of success (p).

    # Returns
    The PMF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    # Bernoulli PMF: p^x * (1-p)^(1-x)
    return lax.exp(x * lax.log(p) + (1.0 - x) * lax.log1p(-p))


def logprob(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    """
    The log of the probability mass function (log PMF) for a Bernoulli distribution.

    # Parameters
    - `x`: Where to evaluate the log PMF (pst be 0 or 1).
    - `p`: The probability of success (p).

    # Returns
    The log PMF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return x * lax.log(p) + (1.0 - x) * lax.log(1.0 - p)


def cdf(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    """
    The cuplative distribution function (CDF) for a Bernoulli distribution.

    # Parameters
    - `x`: Where to evaluate the CDF.
    - `p`: The probability of success (p).

    # Returns
    The CDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, 1.0 - p, 1.0)
    )


def logcdf(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    """
    The log of the cuplative distribution function (log CDF) for a Bernoulli distribution.

    # Parameters
    - `x`: Where to evaluate the log CDF.
    - `p`: The probability of success (p).

    # Returns
    The log CDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        -jnp.inf,
        jnp.where(x < 1.0, lax.log1p(-p), 0.0)
    )


def ccdf(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    """
    The complementary cuplative distribution function (cCDF) for a Bernoulli distribution.

    # Parameters
    - `x`: Where to evaluate the cCDF.
    - `p`: The probability of success (p).

    # Returns
    The cCDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """

    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        1.0,
        jnp.where(x < 1.0, p, 0.0)
    )


def logccdf(
    x: Integer[ArrayLike, "..."],
    p: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    """
    The log of the complementary cuplative distribution function (log cCDF) for a Bernoulli distribution.

    # Parameters
    - `x`: Where to evaluate the log cCDF.
    - `p`: The probability of success (p).

    # Returns
    The log cCDF evaluated at `x`. The output will have the broadcasted shapes of `x` and `p`.
    """
    # Cast to Array
    x, p = jnp.asarray(x), jnp.asarray(p)

    return jnp.where(
        x < 0.0,
        0.0,
        jnp.where(x < 1.0, lax.log(p), -jnp.inf)
    )


class Bernoulli(Distribution):
    """
    A Bernoulli distribution.

    # Attributes
    - `p`: The probability of success (p).
    """

    p: Node[Real[Array, "..."]]


    def __init__(
        self,
        p: Real[ArrayLike, "..."] | Node[Real[ArrayLike, "..."]],
    ):
        # Initialize probability of success parameter (p)
        if isinstance(p, Node):
            if isinstance(p.obj, ArrayLike):
                self.p = p # type: ignore
        else:
            self.p = Observed(jnp.asarray(p))


    def logprob(self, node: Node) -> Scalar:
        obj, p = (
            node.obj,
            self.p.obj,
        )

        # Filter out irrelevant values
        obj, _ = eqx.partition(obj, node._filter_spec)

        # Helper function for the single-leaf log-probability evaluation
        def leaf_logprob(x: Float[ArrayLike, ""]) -> Scalar:
            return logprob(x, p).sum()

        # Compute log probabilities across leaves
        eval_obj = jt.map(leaf_logprob, obj)

        # Compute total sum
        total = jt.reduce_associative(lambda x,y: x + y, eval_obj, identity=0.0)

        return jnp.asarray(total)

    def sample(self, shape: int | Tuple[int, ...], key: PRNGKeyArray = jr.key(0)):
        # Coerce to tuple
        if isinstance(shape, int):
            shape = (shape, )

        return jr.bernoulli(key, self.p.obj, shape)
