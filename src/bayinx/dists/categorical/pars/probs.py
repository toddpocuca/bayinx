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
    probs: ArrayLike
) -> Array:
    x, probs = jnp.asarray(x), jnp.asarray(probs)

    return probs[..., x]


def _logprob(
    x: ArrayLike,
    probs: ArrayLike
) -> Array:
    x, probs = jnp.asarray(x), jnp.asarray(probs)

    return jnp.log(probs[..., x])


class ProbsCategorical(Parameterization):
    """
    A probability parameterization of the Categorical distribution.
    """

    probs: Node[Array]

    def __init__(
        self,
        probs: ArrayObject
    ):
        for name, val in [("probs", probs)]:
            if isinstance(val, Node):
                if isinstance(byo.obj(val), ArrayLike):
                    # Cast to array
                    val = byo.asarray(val)  # type: ignore

                    setattr(self, name, val)
            else:
                setattr(self, name, Observed(jnp.asarray(val)))

    def logprob(self, x: ArrayLike) -> Scalar:
        probs = byo.obj(self.probs)

        return _logprob(x, probs)

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray):
        probs = byo.obj(self.probs)

        return jr.categorical(key, jnp.log(probs), shape=shape)
