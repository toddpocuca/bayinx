from abc import abstractmethod
from functools import partial
from typing import Callable, Self, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import optax as opx
from jaxtyping import Array, Bool, PRNGKeyArray, PyTree, Scalar
from optax import GradientTransformation, OptState

from bayinx.core.model import Model
from bayinx.core.progress import close_progress, update_progress
from bayinx.core.sampler import Sampler


class Variational[M: Model](Sampler):
    """
    An abstract base class used to define variational inference methods.

    Attributes:
        dim: The dimension of the parameter space.
        _unflatten: A function to transform draws from the variational distribution back to a `Model`.
        _static: The static component of a partitioned `Model` used to initialize the `Variational` object.
    """

    @property
    @abstractmethod
    def n_pars(self) -> int:
        """
        Number of variational parameters.
        """
        pass

    @abstractmethod
    def eval(self, draws: Array) -> Array:
        """
        Evaluate the variational distribution at `draws`.
        """
        pass

    @abstractmethod
    def elbo(self, n: int, batch_size: int, key: PRNGKeyArray) -> Array:
        """
        Evaluate the ELBO.
        """
        pass

    @abstractmethod
    def elbo_grad(self, n: int, batch_size: int, stl: bool, key: PRNGKeyArray) -> M:
        """
        Evaluate the gradient of the ELBO.
        """
        pass

    @abstractmethod
    def elbo_and_grad(self, n: int, batch_size: int, stl: bool, key: PRNGKeyArray) -> Tuple[Scalar, M]:
        """
        Evaluate the ELBO and its gradient.
        """
        pass

    @eqx.filter_jit
    def fit(
        self,
        max_iters: int = 50_000,
        learning_rate: float = 1e-3,
        grad_draws: int = 1,
        batch_size: int = 1,
        stl: bool = True,
        key: PRNGKeyArray = jr.key(0),
        verbose: bool = True,
        print_rate: int = 5000
    ) -> Self:
        """
        Optimize the variational distribution.

        # Parameters:
        - max_iters: The maximum number of iterations for optimization.
        - `learning_rate`: The initial learning rate for the optimizer.
        - `tolerance`: The tolerance for the ELBO used for early stopping.
        - `grad_draws`: The number of draws used to compute the ELBO gradient.
        - `batch_size`: The maximum number of draws ever in memory used to compute the ELBO gradient.
        - `stl`: Whether to use the Stick-the-Landing estimator.
        - `key`: The PRNG key used during optimization.
        - `verbose`: Whether to print a progress bar.
        - `print_rate`: The number of iterations between updates for the progress bar.
        """
        # Create unique identifier for optimization loop
        loop_id = jr.key_data(key).sum()

        # Determine actual batch size for ELBO & gradient computations
        grad_batch_size = grad_draws if batch_size >= grad_draws else batch_size

        # Partition variational
        dyn, static = eqx.partition(self, self.filter_spec)

        # Initialize optimizer without a learning rate or time-dependent schedule
        optim: GradientTransformation = opx.chain(
            opx.zero_nans(),
            opx.clip_by_global_norm(1.0),
            opx.rmsprop(learning_rate=learning_rate, decay=0.99),
            opx.scale(-1.0)
        )
        opt_state: OptState = optim.init(dyn)

        # Initialize progress bar
        if verbose:
            update_progress(loop_id, 0, max_iters, "Fitting Variational Approximation", print_rate)

        LoopState = tuple[Self, OptState, Scalar, PRNGKeyArray]
        # Helper functions for optimization loop
        @eqx.filter_jit(donate = 'all')
        def condition(state: LoopState) -> Bool[Array, ""]:
            # Unpack iteration state
            dyn, opt_state, i, key = state

            return i < max_iters

        @eqx.filter_jit(donate = 'all')
        def body(state: LoopState) -> LoopState:
            # Unpack iteration state
            dyn, opt_state, i, key = state

            # Update iteration
            i = i + 1

            # Update progress bar
            if verbose:
                update_progress(loop_id, i, max_iters, "Fitting Variational Approximation", print_rate)

            # Update PRNG key
            key, _ = jr.split(key)

            # Reconstruct variational
            vari: Self = eqx.combine(dyn, static)

            # Compute ELBO gradient
            update: M = vari.elbo_grad(grad_draws, grad_batch_size, stl, key)

            # Transform update through optimizer
            update, opt_state = optim.update( # type: ignore
                update, opt_state, dyn # type: ignore
            )

            # Update variational distribution
            dyn: Self = eqx.apply_updates(dyn, update)

            return dyn, opt_state, i, key

        # Run optimization loop
        dyn, _, iter, _ = lax.while_loop(
            cond_fun=condition,
            body_fun=body,
            init_val=(dyn, opt_state, jnp.array(0, jnp.uint32), key),
        )

        # Close progress bar
        if verbose:
            close_progress(loop_id, iter)

        # Return optimized variational approximation
        return eqx.combine(dyn, static)

    def sample_parameter(self, name: str, n_draws: int, batch_size: int = 1, key: PRNGKeyArray = jr.PRNGKey(0)) -> Array:
        """
        Sample a parameter from the posterior.

        Parameters:
            name: The name of the parameter.
            n_draws: The number of draws.
            batch_size: The maximum number of draws (full instances of a model) ever in memory.
            key: The PRNG key to seed downstream RNGs.

        """
        return self.sample_predictive(lambda model, key: getattr(model, name), n_draws, batch_size, key)

    def sample_predictive(self, func: Callable[[M, PRNGKeyArray], PyTree[Array]], n_draws: int, batch_size: int = 1, key: PRNGKeyArray = jr.PRNGKey(0)) -> Array:
        """
        Sample from the posterior predictive.

        Parameters:
            func: The predictive (function) that takes in a model and a PRNG key, and returns a PyTree of arrays.
            n_draws: The number of draws.
            batch_size: The maximum number of draws (full instances of a model) ever in memory.
            key: The PRNG key to seed downstream RNGs.
        """
        # Split key
        per_batch_keys = jr.split(key, n_draws // batch_size)

        @partial(jax.vmap, in_axes = (0, 0))
        def reconstruct_and_query(draw: Array, key: PRNGKeyArray) -> PyTree[Array]:
            model = self.reconstruct_model(draw).constrain()[0]

            # Evaluate callable
            obj = func(model, key)

            return obj

        # Sample in batches
        def batched_sample(per_batch_key: PRNGKeyArray) -> PyTree[Array]:
            # Sample draws
            draws = self.sample_draws(batch_size, key = per_batch_key)

            # Generate keys for each draw
            within_batch_keys = jr.split(per_batch_key, batch_size)

            return reconstruct_and_query(draws, within_batch_keys)

        # Generate samples of the posterior/posterior predictive
        post_draws: PyTree[Array] = lax.map(
            batched_sample,
            per_batch_keys
        )

        # Reshape to remove batch axis
        post_draws = jt.map(lambda x: x.reshape(-1, *x.shape[2:]), post_draws, is_leaf = lambda x: isinstance(x, Array))

        return post_draws
