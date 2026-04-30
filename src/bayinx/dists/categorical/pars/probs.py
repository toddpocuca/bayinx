import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


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

    probs: Array

    def __init__(
        self,
        probs: ArrayLike
    ):
        for name, val in [("probs", probs)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        probs = self.probs

        return _logprob(x, probs)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        probs = self.probs

        return jr.categorical(key, jnp.log(probs), shape=shape)
