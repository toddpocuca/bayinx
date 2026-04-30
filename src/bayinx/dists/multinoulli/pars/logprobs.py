from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    logprobs: ArrayLike
) -> Array:
    # Cast to Array
    x, logprobs = jnp.asarray(x), jnp.asarray(logprobs)

    # Gather probabilities
    return jnp.exp(_logprob(x, logprobs))


def _logprob(
    x: ArrayLike,
    logprobs: ArrayLike
) -> Array:
    # Cast to Array
    x, logprobs = jnp.asarray(x), jnp.asarray(logprobs)

    return jnp.sum(x * logprobs, -1)

class LogProbsMultinoulli(Parameterization):
    """
    A log-probability parameterization of the Multinoulli distribution.
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
        # Extract log-probabilities
        logprobs = self.logprobs

        return _logprob(x, logprobs)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract log-probabilities
        logprobs = self.logprobs

        # Determine number of cases
        n_classes = logprobs.shape[-1]

        return jnn.one_hot(jr.categorical(key, logprobs, shape=shape), n_classes)
