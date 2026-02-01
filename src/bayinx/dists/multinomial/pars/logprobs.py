from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed


def _prob(
    x: ArrayLike,
    n: ArrayLike,
    logprobs: ArrayLike
) -> Array:
    # Cast to Array
    x, n, logprobs = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logprobs)

    # Calculate probabilities
    return jnp.exp(_logprob(x, n, logprobs))


def _logprob(
    x: ArrayLike,
    n: ArrayLike,
    logprobs: ArrayLike
) -> Array:
    # Cast to Array
    x, n, logprobs = jnp.asarray(x), jnp.asarray(n), jnp.asarray(logprobs)

    # Calculate log-coefficient and log-probabilities
    log_coeff = gammaln(n + 1) - jnp.sum(gammaln(x + 1), -1)
    return log_coeff + jnp.sum(x * logprobs, -1)


class LogProbsMultinomial(Parameterization):
    """
    A log-probability parameterization of the Multinomial distribution.
    """

    n: Node[Array]
    logprobs: Node[Array]

    def __init__(
        self,
        n: ArrayObject,
        logprobs: ArrayObject,
    ):
        for name, val in [("logprobs", logprobs), ("n", n)]:
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
        logprobs = byo.obj(self.logprobs)

        return _logprob(x, n, logprobs)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        logprobs = byo.obj(self.logprobs)
        n = byo.obj(self.n)

        # Convert log-probabilities to probabilities for sampling
        probs = jnp.exp(logprobs)

        return jr.multinomial(key, n, probs, shape=shape)
