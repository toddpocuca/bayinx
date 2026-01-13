from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import t as jsst_t
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Real, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.nodes import Observed

PI = 3.141592653589793

def _prob(x: Real[ArrayLike, "..."], df: Real[ArrayLike, "..."], loc: Real[ArrayLike, "..."], scale: Real[ArrayLike, "..."]) -> Real[Array, "..."]:
    return jnp.exp(_logprob(x, df, loc, scale))

def _logprob(
    x: Real[ArrayLike, "..."],
    df: Real[ArrayLike, "..."],
    loc: Real[ArrayLike, "..."],
    scale: Real[ArrayLike, "..."]
) -> Real[Array, "..."]:
    x, df, loc, scale = jnp.asarray(x), jnp.asarray(df), jnp.asarray(loc), jnp.asarray(scale)

    return jsst_t.logpdf(x, df, loc, scale)

#def _cdf(x: Real[ArrayLike, "..."], df: Real[ArrayLike, "..."], loc: Real[ArrayLike, "..."], scale: Real[ArrayLike, "..."]) -> Real[Array, "..."]:
#    pass

#def _logcdf(x: Real[ArrayLike, "..."], df: Real[ArrayLike, "..."], loc: Real[ArrayLike, "..."], scale: Real[ArrayLike, "..."]) -> Real[Array, "..."]:
#    pass

#def _ccdf(x: Real[ArrayLike, "..."], df: Real[ArrayLike, "..."], loc: Real[ArrayLike, "..."], scale: Real[ArrayLike, "..."]) -> Real[Array, "..."]:
#    pass

#def _logccdf(x: Real[ArrayLike, "..."], df: Real[ArrayLike, "..."], loc: Real[ArrayLike, "..."], scale: Real[ArrayLike, "..."]) -> Real[Array, "..."]:
#    pass

class LocScaleStudentT(Parameterization):
    """
    A location-scale parameterization of the Student's T distribution.
    """
    df: Node[Real[Array, "..."]]
    loc: Node[Real[Array, "..."]]
    scale: Node[Real[Array, "..."]]

    def __init__(
        self,
        df: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        loc: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        scale: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        # TODO: testing out this
        for name, val in [("df", df), ("loc", loc), ("scale", scale)]:
            if isinstance(val, Node):
                if isinstance(val.obj, ArrayLike):
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
