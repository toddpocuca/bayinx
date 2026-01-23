from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed

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

    loc: Node[Array]
    var: Node[Array]

    def __init__(
        self,
        loc: ArrayObject,
        var: ArrayObject
    ):
        # Initialize loc parameter
        if isinstance(loc, Node):
            if isinstance(loc._byx__obj, ArrayLike):
                self.loc: Node[ArrayLike] = loc
        else:
            self.loc = Observed(jnp.asarray(loc))

        # Initialize scale parameter
        if isinstance(var, Node):
            if isinstance(var._byx__obj, ArrayLike):
                self.var: Node[ArrayLike] = var
        else:
            self.var = Observed(jnp.asarray(var))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        loc = byo.obj(self.loc)
        var = byo.obj(self.var)

        return _logprob(x, loc, var)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        loc = byo.obj(self.loc)
        var = byo.obj(self.var)

        return jr.normal(key, shape) * jnp.sqrt(var) + loc
