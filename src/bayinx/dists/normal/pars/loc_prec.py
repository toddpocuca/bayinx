
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization

PI = 3.141592653589793


def _prob(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jnp.sqrt(prec) / jnp.sqrt(2.0 * PI) * jnp.exp(-0.5 * prec * jnp.square(x - loc))


def _logprob(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return 0.5 * jnp.log(prec) - jnp.log(jnp.sqrt(2.0 * PI)) - 0.5 * prec * jnp.square(x - loc)


def _cdf(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jsp.ndtr((x - loc) * jnp.sqrt(prec))


def _logcdf(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jsp.log_ndtr((x - loc) * jnp.sqrt(prec))


def _ccdf(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jsp.ndtr((loc - x) * jnp.sqrt(prec))


def _logccdf(
    x: ArrayLike,
    loc: ArrayLike,
    prec: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, prec = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(prec)

    return jsp.log_ndtr((loc - x) * jnp.sqrt(prec))


class LocPrecisionNormal(Parameterization):
    """
    A loc-precision parameterization of the normal distribution.
    """

    loc: Array
    prec: Array

    def __init__(
        self,
        loc: ArrayLike,
        prec: ArrayLike
    ):
        # Initialize parameters
        for name, val in [("loc", loc), ("prec", prec)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        loc = self.loc
        prec = self.prec

        return _logprob(x, loc, prec)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        loc = self.loc
        prec = self.prec

        return jr.normal(key, shape) / jnp.sqrt(prec) + loc
