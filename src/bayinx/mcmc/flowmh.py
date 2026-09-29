from functools import partial
from typing import Callable, Self

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from bayinx.core.mcmc import MCMC
from bayinx.core.model import Model
from bayinx.core.variational import Variational
from bayinx.vi.normalizing_flow import NormalizingFlow


class FlowMH[M: Model](MCMC[M]):
    """
    A flow-augmented metropolis-hastings MCMC method.

    Attributes:
        dim: The dimension of the parameter space.
        _unflatten: A function to transform draws from the variational distribution back to a `Model`.
        _static: The static component of a partitioned `Model` used to initialize the `Variational` object.
        n_chains: The number of MCMC chains used to generate samples.
        states: The current states of the chain(s).
        nf: The normalizing flow approximation used to augment the posterior.
        logprobs: Cached log-probabilities of the current state(s) to avoid re-computation.
        target_rate: The target acceptance rate.
        step_size: The step-size (standard deviation) of the proposal distribution.

    """
    nf: NormalizingFlow[M, Variational[M]]
    logprobs: Float[Array, " n_chains"]
    target_rate: float
    step_size: float

    def __init__(self, nf: NormalizingFlow[M, Variational[M]], n_chains: int = 1, key: PRNGKeyArray = jr.key(0)):
        self.dim = nf.dim
        self._unflatten = nf._unflatten
        self._static = nf._static
        self.n_chains = n_chains
        self.nf = nf

        # Initialize current state(s)
        self.states = jr.normal(key, (n_chains, self.dim))

        # Cache the starting log probabilities
        self.logprobs = self.__augmented_eval(self.states)

        # Compute approximate optimal target acceptance rate
        self.target_rate = 0.234 + 0.207 / (self.dim ** 0.5)

        # Compute optimal step size
        self.step_size = 2.38 / (self.dim ** 0.5)

    @property
    def filter_spec(self) -> Self:
        # Generate empty specification
        filter_spec: Self = jt.map(lambda _: False, self)

        # Update logprobs and states as dynamic
        filter_spec = eqx.tree_at(
            lambda mcmc: (mcmc.states, mcmc.logprobs),
            filter_spec,
            (True, True)
        )

        return filter_spec

    def __augmented_eval(self, draws: Array) -> Array:
        """
        Evaluates the augmented posterior log density for a batch of draws.

        Parameters:
            draws: A batch of draws from the augmented latent space with shape `(batch_size, dim)`.

        Returns:
            The associated augmented log densities with shape `(batch_size,)`.
        """
        # Apply forward-flow
        draws, total_log_jacs = self.nf.forward_and_eval(draws)

        # Evaluate model posterior
        posterior_evals = self.eval_model(draws)

        # Return augmented posterior log density
        return posterior_evals + total_log_jacs

    def step(self, key: PRNGKeyArray) -> Self:
        """
        One transition of the chain(s) in the augmented latent space.

        Parameters:
            key: The PRNG key.
        """
        k1, k2 = jr.split(key)

        # Propose new states in the late=nt space
        proposals = self.states + jr.normal(k1, self.states.shape) * self.step_size

        # Evaluate augmented log densities for all proposed states
        proposal_logprobs = self.__augmented_eval(proposals)

        # Compute Metropolis acceptance probabilities
        accept_probs = jnp.minimum(1.0, jnp.exp(proposal_logprobs - self.logprobs))

        # Sample acceptance/rejection
        accepted = jr.bernoulli(k2, accept_probs, shape = (self.n_chains, ))

        # Update latent positions and cache the new log probabilities
        new_states = jnp.where(jnp.expand_dims(accepted, -1), proposals, self.states)
        new_logprobs = jnp.where(accepted, proposal_logprobs, self.logprobs)

        # Update chains
        self = eqx.tree_at(
            lambda m: (m.states, m.logprobs),
            self,
            (new_states, new_logprobs)
        )

        return self

    def warmup(self, key: PRNGKeyArray, n_warmup: int) -> Self:
        """
        A burn-in phase to allow the chain(s) to converge to the high-density region
        using the fixed theoretical optimal step size.
        """
        return self

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

        # Push draws through normalizing flow
        draws = self.nf.forward(draws)

        # Reconstruct final chain
        self: Self = eqx.combine(dyn, static)

        return self, draws

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

            # Push draws through normalizing flow
            draws = self.nf.forward(draws)

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
