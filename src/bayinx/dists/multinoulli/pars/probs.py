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

    probs: Node[Array]

    def __init__(
        self,
        probs: ArrayObject,
    ):
        for name, val in [("probs", probs)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val) # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract probabilities
        probs = byo.obj(self.probs)

        return _logprob(x, probs)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        # Extract log-probabilities
        logprobs = jnp.log(byo.obj(self.probs))

        # Determine number of cases
        n_classes = logprobs.shape[-1]

        return jnn.one_hot(jr.categorical(key, logprobs, shape=shape), n_classes)
