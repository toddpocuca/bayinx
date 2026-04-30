from abc import abstractmethod
from functools import partial
from typing import Callable, Self, Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax as opx
from jaxtyping import Array, Bool, PRNGKeyArray, Scalar
from optax import GradientTransformation, OptState

from bayinx.core.model import Model
from bayinx.core.progress import close_progress, update_progress


class Variational[M: Model](eqx.Module):
    """
    An abstract base class used to define variational inference methods.

    Attributes:
        dim: The dimension of the parameter space.
        _unflatten: A function to transform draws from the variational distribution back to a `Model`.
        _static: The static component of a partitioned `Model` used to initialize the `Variational` object.
    """
    dim: int
    _unflatten: Callable[[Array], M]
    _static: M

    @property
    @abstractmethod
    def n_pars(self) -> int:
        """
        Number of variational parameters.
        """
        pass

    @property
    @abstractmethod
    def filter_spec(self) -> Self:
        """
        Filter specification for dynamic and static components of the
        `Variational` object.
        """
        pass

    @abstractmethod
    def sample(self, n: int, key: PRNGKeyArray = jr.PRNGKey(0)) -> Array:
        """
        Sample from the variational distribution.
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
    def reconstruct_model(self, draw: Array) -> M:
        # Unflatten variational draw
        model: M = self._unflatten(draw)

        # Combine with static components
        model: M = eqx.combine(model, self._static)

        return model

    @eqx.filter_jit
    @partial(jax.vmap, in_axes=(None, 0))
    def eval_model(self, draws: Array) -> Array:
        """
        Reconstruct models from variational draws and evaluate their posterior.

        # Parameters
        - `draws`: A set of variational draws.
        """
        # Unflatten variational draw
        model: M = self.reconstruct_model(draws)

        # Evaluate posterior
        return model()

    @eqx.filter_jit
    def fit(
        self,
        max_iters: int,
        learning_rate: float,
        tolerance: float,
        grad_draws: int,
        batch_size: int,
        stl: bool = False,
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

        # Construct scheduler
        lr_schedule: Callable = opx.warmup_cosine_decay_schedule(
            jnp.finfo(jnp.array(0.0)).eps.item(),
            learning_rate,
            int(max_iters * 0.1),
            max_iters - int(max_iters * 0.1),
            jnp.finfo(jnp.array(0.0)).eps.item()
        )

        # Initialize optimizer
        optim: GradientTransformation = opx.chain(
            opx.zero_nans(),
            opx.adamax(lr_schedule),
            opx.ema(0.99),
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
