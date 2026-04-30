from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import t as jsst_t
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization

PI = 3.141592653589793

def _prob(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array:
    return jnp.exp(_logprob(x, df, loc, scale))

def _logprob(
    x: ArrayLike,
    df: ArrayLike,
    loc: ArrayLike,
    scale: ArrayLike
) -> Array:
    x, df, loc, scale = jnp.asarray(x), jnp.asarray(df), jnp.asarray(loc), jnp.asarray(scale)

    return jsst_t.logpdf(x, df, loc, scale)

#def _cdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array:
#    pass

#def _logcdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array:
#    pass

#def _ccdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array:
#    pass

#def _logccdf(x: ArrayLike, df: ArrayLike, loc: ArrayLike, scale: ArrayLike) -> Array:
#    pass

class LocScaleStudentsT(Parameterization):
    """
    A location-scale parameterization of the Student's T distribution.
    """
    df: Array
    loc: Array
    scale: Array

    def __init__(
        self,
        df: ArrayLike,
        loc: ArrayLike,
        scale: ArrayLike
    ):
        # Initialize parameters
        for name, val in [("df", df), ("loc", loc), ("scale", scale)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        df = self.df
        loc = self.loc
        scale = self.scale

        return _logprob(x, df, loc, scale)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        df = self.df
        loc = self.loc
        scale = self.scale

        return jr.t(key, df, shape) * scale + loc
