from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import t as jsst_t
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed

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
    df: Node[Array]
    loc: Node[Array]
    scale: Node[Array]

    def __init__(
        self,
        df: ArrayObject,
        loc: ArrayObject,
        scale: ArrayObject
    ):
        for name, val in [("df", df), ("loc", loc), ("scale", scale)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        df = byo.obj(self.df)
        loc = byo.obj(self.loc)
        scale = byo.obj(self.scale)

        return _logprob(x, df, loc, scale)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        df = byo.obj(self.df)
        loc = byo.obj(self.loc)
        scale = byo.obj(self.scale)

        return jr.t(key, df, shape) * scale + loc
