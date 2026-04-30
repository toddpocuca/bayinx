
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization

PI = 3.141592653589793

def _prob(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return 1 / (scale * jnp.sqrt(2.0 * PI)) * jnp.exp(-0.5 * jnp.square((x - loc) / scale))


def _logprob(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    # Compute variance
    var = jnp.square(scale)

    return -0.5 * (jnp.log(2.0 * PI * var) + jnp.square(x - loc) / var)


def _cdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jsp.ndtr((x - loc) / scale)


def _logcdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jsp.log_ndtr((x - loc) / scale)


def _ccdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jsp.ndtr((loc - x) / scale)


def _logccdf(
    x: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike,
) -> Array:
    # Cast to Array
    x, loc, scale = jnp.asarray(x), jnp.asarray(loc), jnp.asarray(scale)

    return jsp.log_ndtr((loc - x) / scale)



class LocScaleNormal(Parameterization):
    """
    A loc-scale parameterization of the normal distribution.
    """

    loc: Array
    scale: Array

    def __init__(
        self,
        loc: ArrayLike,
        scale: ArrayLike
    ):
        # Initialize parameters
        for name, val in [("loc", loc), ("scale", scale)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        loc = self.loc
        scale = self.scale

        return _logprob(x, loc, scale)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        loc = self.loc
        scale = self.scale

        return jr.normal(key, shape) * scale + loc
