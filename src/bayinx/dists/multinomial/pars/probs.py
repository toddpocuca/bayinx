import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gamma, gammaln
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


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

    n: Array
    probs: Array

    def __init__(
        self,
        n: ArrayLike,
        probs: ArrayLike
    ):
        for name, val in [("probs", probs), ("n", n)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        n = self.n
        probs = self.probs

        return _logprob(x, n, probs)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        probs = self.probs
        n = self.n

        return jr.multinomial(key, n, probs, shape=shape)
