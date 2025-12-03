from typing import Tuple

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
import jax.tree as jt
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Real, Scalar

from bayinx.core.distribution import Distribution
from bayinx.core.node import Node
from bayinx.nodes import Observed

PI = 3.141592653589793


def prob(
    x: Real[ArrayLike, "..."],
    mu: Real[ArrayLike, "..."],
    sigma: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The probability density function (PDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return lax.exp(-0.5 * lax.square((x - mu) / sigma)) / (sigma * lax.sqrt(2.0 * PI))


def logprob(
    x: Real[ArrayLike, "..."],
    mu: Real[ArrayLike, "..."],
    sigma: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the probability density function (log PDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the log PDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The log PDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return -lax.log(lax.sqrt(2.0 * PI)) - lax.log(sigma) - 0.5 * lax.square((x - mu) / sigma)


def cdf(
    x: Real[ArrayLike, "..."],
    mu: Real[ArrayLike, "..."],
    sigma: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The cumulative density function (CDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the CDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The CDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return jsp.ndtr((x - mu) / sigma)


def logcdf(
    x: Real[ArrayLike, "..."],
    mu: Real[ArrayLike, "..."],
    sigma: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the cumulative density function (log CDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the log CDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The log CDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return jsp.log_ndtr((x - mu) / sigma)


def ccdf(
    x: Real[ArrayLike, "..."],
    mu: Real[ArrayLike, "..."],
    sigma: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The complementary cumulative density function (cCDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the cCDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The cCDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return jsp.ndtr((mu - x) / sigma)


def logccdf(
    x: Real[ArrayLike, "..."],
    mu: Real[ArrayLike, "..."],
    sigma: Real[ArrayLike, "..."],
) -> Real[Array, "..."]:
    """
    The log of the complementary cumulative density function (log cCDF) for a Normal distribution.

    # Parameters
    - `x`: Where to evaluate the log cCDF.
    - `mu`: The mean.
    - `sigma`: The standard deviation.

    # Returns
    The log cCDF evaluated at `x`. The output will have the broadcasted shapes of `x`, `mu`, and `sigma`.
    """
    # Cast to Array
    x, mu, sigma = jnp.asarray(x), jnp.asarray(mu), jnp.asarray(sigma)

    return jsp.log_ndtr((mu - x) / sigma)


class Normal(Distribution):
    """
    A normal distribution.

    # Attributes
    - `mu`: The mean/location parameter.
    - `sigma`: The standard-deviation/scale parameter.
    """

    mu: Node[Real[Array, "..."]]
    sigma: Node[Real[Array, "..."]]


    def __init__(
        self,
        mu: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        sigma: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # Initialize mean/location parameter (mu)
        if isinstance(mu, Node):
            if isinstance(mu.obj, ArrayLike):
                self.mu = mu # type: ignore
        else:
            self.mu = Observed(jnp.asarray(mu))

        # Initialize dispersion/scale parameter (sigma)
        if isinstance(sigma, Node):
            if isinstance(sigma.obj, ArrayLike):
                self.sigma = sigma # type: ignore
        else:
            self.sigma = Observed(jnp.asarray(sigma))


    def logprob(self, node: Node) -> Scalar:
        obj, mu, sigma = (
            node.obj,
            self.mu.obj,
            self.sigma.obj
        )

        # Filter out irrelevant values
        obj, _ = eqx.partition(obj, node._filter_spec)

        # Helper function for the single-leaf log-probability evaluation
        def leaf_logprob(x: Real[ArrayLike, "..."]) -> Scalar:
            return logprob(x, mu, sigma).sum()

        # Compute log probabilities across leaves
        eval_obj = jt.map(leaf_logprob, obj)

        # Compute total sum
        total = jt.reduce_associative(lambda x,y: x + y, eval_obj, identity=0.0)

        return jnp.asarray(total)

    def sample(self, shape: int | Tuple[int, ...], key: PRNGKeyArray = jr.key(0)):
        # Coerce to tuple
        if isinstance(shape, int):
            shape = (shape, )

        return jr.normal(key, shape) * self.sigma.obj + self.mu.obj
