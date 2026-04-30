from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


def _prob(
    x: ArrayLike,
    probs: ArrayLike
) -> Array:
    # Cast to Array
    x, probs = jnp.asarray(x), jnp.asarray(probs)

    # Gather probabilities
    return jnp.prod(probs**x, -1)


def _logprob(
    x: ArrayLike,
    probs: ArrayLike
) -> Array:
    # Cast to Array
    x, probs = jnp.asarray(x), jnp.asarray(probs)

    return jnp.sum(x * jnp.log(probs), -1)

class ProbsMultinoulli(Parameterization):
    """
    A probability parameterization of the Multinoulli distribution.
    """

    probs: Array

    def __init__(
        self,
        probs: ArrayLike,
    ):
        for name, val in [("probs", probs)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract probabilities
        probs = self.probs

        return _logprob(x, probs)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract log-probabilities
        logprobs = jnp.log(self.probs)

        # Determine number of cases
        n_classes = logprobs.shape[-1]

        return jnn.one_hot(jr.categorical(key, logprobs, shape=shape), n_classes)
