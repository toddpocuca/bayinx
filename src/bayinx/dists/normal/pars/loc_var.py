
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization

PI = 3.141592653589793


def _prob(
    x: ArrayLike,
    loc: ArrayLike,
    var: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, var = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(var)

    # Compute scale
    scale = jnp.sqrt(var)

    return norm.pdf(x, loc, scale)


def _logprob(
    x: ArrayLike,
    loc: ArrayLike,
    var: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, var = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(var)

    # Compute scale
    scale = jnp.sqrt(var)

    return norm.logpdf(x, loc, scale)


def _cdf(
    x: ArrayLike,
    loc: ArrayLike,
    var: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, var = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(var)

    # Compute scale
    scale = jnp.sqrt(var)

    return norm.cdf(x, loc, scale)


def _logcdf(
    x: ArrayLike,
    loc: ArrayLike,
    var: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, var = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(var)

    # Compute scale
    scale = jnp.sqrt(var)

    return norm.logcdf(x, loc, scale)


def _ccdf(
    x: ArrayLike,
    loc: ArrayLike,
    var: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, var = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(var)

    # Compute scale
    scale = jnp.sqrt(var)

    return norm.sf(x, loc, scale)


def _logccdf(
    x: ArrayLike,
    loc: ArrayLike,
    var: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, var = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(var)

    # Compute scale
    scale = jnp.sqrt(var)

    return norm.logsf(x, loc, scale)



class LocVarNormal(Parameterization):
    """
    A loc-variance parameterization of the normal distribution.
    """

    loc: Array
    var: Array

    def __init__(
        self,
        loc: ArrayLike,
        var: ArrayLike
    ):
        # Initialize loc parameter
        for name, val in [("loc", loc), ("var", var)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        loc = self.loc
        var = self.var

        return _logprob(x, loc, var)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        loc = self.loc
        var = self.var

        return jr.normal(key, shape) * jnp.sqrt(var) + loc
