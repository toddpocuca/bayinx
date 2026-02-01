from typing import Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

import bayinx.ops as byo
from bayinx.core.distribution import Parameterization
from bayinx.core.node import Node
from bayinx.core.types import ArrayObject
from bayinx.nodes import Observed


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

    logprobs: Node[Array]

    def __init__(
        self,
        logprobs: ArrayObject,
    ):
        for name, val in [("logprobs", logprobs)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract log-probabilities
        logprobs = byo.obj(self.logprobs)

        return _logprob(x, logprobs)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract log-probabilities
        logprobs = byo.obj(self.logprobs)

        # Determine number of cases
        n_classes = logprobs.shape[-1]

        return jnn.one_hot(jr.categorical(key, logprobs, shape=shape), n_classes)
