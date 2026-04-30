import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.distribution import Parameterization


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

    n: Array
    logprobs: Array

    def __init__(
        self,
        n: ArrayLike,
        logprobs: ArrayLike,
    ):
        for name, val in [("logprobs", logprobs), ("n", n)]:
            # Cast to array
            val = jnp.asarray(val)

            setattr(self, name, val)

    def logprob(self, x: ArrayLike) -> Scalar:
        # Extract parameters
        n = self.n
        logprobs = self.logprobs

        return _logprob(x, n, logprobs)

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray):
        # Extract parameters
        logprobs = self.logprobs
        n = self.n

        # Convert log-probabilities to probabilities for sampling
        probs = jnp.exp(logprobs)

        return jr.multinomial(key, n, probs, shape=shape)
