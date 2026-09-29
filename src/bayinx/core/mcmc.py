from abc import abstractmethod
from functools import partial
from typing import Callable, Self

import equinox as eqx
import jax
import jax.lax as lax
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from bayinx.core.model import Model
from bayinx.core.sampler import Sampler


class MCMC[M: Model](Sampler[M]):
    """
    An abstract base class used to define MCMC methods.

    Attributes:
        dim: The dimension of the parameter space.
        _unflatten: A function to transform draws from the variational distribution back to a `Model`.
        _static: The static component of a partitioned `Model` used to initialize the `Variational` object.
        n_chains: The number of chains used to generate samples.
        states: The current states of the chain(s).
    """
    n_chains: int = eqx.field(static=True)
    states: Float[Array, "n_chains n_dims"]

    @abstractmethod
    def step(self, key: PRNGKeyArray) -> Self:
        """
        One transition of the chain(s).

        Parameters:
            key: The PRNG key.
        """
        pass

    @abstractmethod
    def warmup(self, key: PRNGKeyArray, n_warmup: int) -> Self:
        """
        An adaptation "warm-up" phase to configure any hyper-parameters.

        Parameters:
            state: The current state of the chain.
            key: The PRNG key.
        """
        pass

    def sample_draws(self, n_draws: int, key: PRNGKeyArray = jr.key(0)) -> tuple[Self, Array]:
        """
        Sample (flattened) posterior draws.

        Parameters:
            n_draws: The number of draws to sample.
            key: The PRNG key used to seed any downstream RNGs.
        """
        # Partition chain
        dyn, static = eqx.partition(self, self.filter_spec)

        keys = jr.split(key, n_draws)
        def scan_fn(dyn, key):
            # Reconstruct chain
            chain: Self = eqx.combine(dyn, static)

            # Perform one-step transition
            chain = chain.step(key)

            # Re-partition the updated chain
            dyn, _ = eqx.partition(chain, self.filter_spec)

            return dyn, chain.states
        dyn, draws = lax.scan(scan_fn, dyn, keys)

        # Reconstruct final chain
        self: Self = eqx.combine(dyn, static)

        return self, draws

    def sample_parameter(self, name: str, n_draws: int, key: PRNGKeyArray = jr.PRNGKey(0)) -> tuple[Self, Array]:
        """
        Sample a parameter from the posterior.

        Parameters:
            name: The name of the parameter.
            n_draws: The number of draws.
            key: The PRNG key to seed downstream RNGs.
        """
        return self.sample_predictive(lambda model, _: getattr(model, name), n_draws, key)

    def sample_predictive(self, func: Callable[[M, PRNGKeyArray], PyTree[Array]], n_draws: int, key: PRNGKeyArray = jr.PRNGKey(0)) -> tuple[Self, Array]:
        """
        Sample from the posterior predictive.

        Parameters:
            func: The predictive (function) that takes in a model and a PRNG key, and returns a PyTree of arrays.
            n_draws: The number of draws.
            key: The PRNG key to seed downstream RNGs.
        """
        # Determine batch size
        batch_size = self.n_chains

        # Partition chain to isolate static structures from dynamic arrays
        dyn, static = eqx.partition(self, self.filter_spec)

        # Get per batch keys
        per_batch_keys = jr.split(key, n_draws // batch_size)

        @partial(jax.vmap, in_axes = (0, 0))
        def reconstruct_and_query(draw: Array, key: PRNGKeyArray) -> PyTree[Array]:
            # Reconstruct model
            model = self.reconstruct_model(draw).constrain()[0]

            # Evaluate callable
            obj = func(model, key)

            return obj

        # Sample in batches
        def batched_sample(dyn_carry, per_batch_key: PRNGKeyArray) -> tuple[Self, PyTree[Array]]:
            # Reconstruct chain for the current step
            chain: Self = eqx.combine(dyn_carry, static)

            # Perform a one-step transition
            chain = chain.step(per_batch_key)

            # Extract states as draws
            draws = chain.states

            # Generate keys for each draw
            within_batch_keys = jr.split(per_batch_key, batch_size)

            # Re-partition the updated chain
            dyn, _ = eqx.partition(chain, self.filter_spec)

            return dyn, reconstruct_and_query(draws, within_batch_keys)

        # Generate samples of the posterior/posterior predictive
        dyn, predictive_draws = lax.scan(
            f = batched_sample,
            init = dyn,  # Pass the pure JAX array tree here
            xs = per_batch_keys
        )

        # Reconstruct final chain state
        self: Self = eqx.combine(dyn, static)

        # Reshape to remove batch axis
        predictive_draws = jt.map(lambda x: x.reshape(-1, *x.shape[2:]), predictive_draws, is_leaf = lambda x: isinstance(x, Array))

        return self, predictive_draws
