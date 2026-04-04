from typing import Tuple

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

    logprobs: Node[Array]

    def __init__(
        self,
        logprobs: ArrayObject,
    ):
        for name, val in [("logprobs", logprobs)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val)  # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        logprobs = byo.obj(self.logprobs)

        return _logprob(x, logprobs)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        logprobs = byo.obj(self.logprobs)

        return jr.categorical(key, logprobs, shape=shape)
