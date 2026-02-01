from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gamma, gammaln
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed


def _prob(
    x: ArrayLike,
    n: ArrayLike,
    probs: ArrayLike
) -> Array:
    # Cast to Array
    x, probs, n = jnp.asarray(x), jnp.asarray(probs), jnp.asarray(n)

    # Calculate multinomial coefficient and probabilities
    coeff = gamma(n + 1) / jnp.prod(gamma(x + 1), -1)
    return coeff * jnp.prod(probs**x, -1)


def _logprob(
    x: ArrayLike,
    n: ArrayLike,
    probs: ArrayLike
) -> Array:
    # Cast to Array
    x, probs, n = jnp.asarray(x), jnp.asarray(probs), jnp.asarray(n)

    # Calculate log-coefficient and log-probabilities
    log_coeff = gammaln(n + 1) - jnp.sum(gammaln(x + 1), -1)
    return log_coeff + jnp.sum(x * jnp.log(probs), -1)


class ProbsMultinomial(Parameterization):
    """
    A probability parameterization of the Multinomial distribution.
    """

    n: Node[Array]
    probs: Node[Array]

    def __init__(
        self,
        n: ArrayObject,
        probs: ArrayObject
    ):
        for name, val in [("probs", probs), ("n", n)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val)  # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        n = byo.obj(self.n)
        probs = byo.obj(self.probs)

        return _logprob(x, n, probs)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        probs = byo.obj(self.probs)
        n = byo.obj(self.n)

        return jr.multinomial(key, n, probs, shape=shape)
