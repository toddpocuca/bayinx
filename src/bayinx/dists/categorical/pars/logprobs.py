from typing import Tuple

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    logprobs: ArrayLike
) -> Array:
    x, logprobs = jnp.asarray(x), jnp.asarray(logprobs)

    return jnp.exp(logprobs[..., x])


def _logprob(
    x: ArrayLike,
    logprobs: ArrayLike
) -> Array:
    x, logprobs = jnp.asarray(x), jnp.asarray(logprobs)

    return logprobs[..., x]


class LogProbsCategorical(Parameterization):
    """
    A log-probability parameterization of the Categorical distribution.
    """

    logprobs: Array

    def __init__(
        self,
        logprobs: ArrayLike,
    ):
        for name, val in [("logprobs", logprobs)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        logprobs = self.logprobs

        return _logprob(x, logprobs)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        logprobs = self.logprobs

        return jr.categorical(key, logprobs, shape=shape)
