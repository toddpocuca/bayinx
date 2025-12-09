from abc import abstractmethod
from functools import partial
from typing import Callable, Generic, Self, Tuple, TypeVar

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import optax as opx
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar
from optax import GradientTransformation, OptState

from bayinx.core.model import Model

M = TypeVar("M", bound=Model)

class Variational(eqx.Module, Generic[M]):
    """
    An abstract base class used to define variational methods.

    # Attributes
    - `dim`: The dimension of the support.
    - `_unflatten`: A function to transform draws from the variational distribution back to a `Model`.
    - `_static`: The static component of a partitioned `Model` used to initialize the `Variational` object.
    """
    dim: int
    _unflatten: Callable[[Array], M]
    _static: M

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
    def elbo_grad(self, n: int, batch_size: int, key: PRNGKeyArray) -> M:
        """
        Evaluate the gradient of the ELBO.
        """
        pass

    @abstractmethod
    def elbo_and_grad(self, n: int, batch_size: int, key: PRNGKeyArray) -> Tuple[Scalar, PyTree]:
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
        key: PRNGKeyArray = jr.key(0),
    ) -> Self:
        """
        Optimize the variational distribution.
        """
        # Partition variational
        dyn, static = eqx.partition(self, self.filter_spec)

        # Construct scheduler
        schedule = opx.cosine_decay_schedule(
            learning_rate,
            max_iters
        )

        # Initialize optimizer
        optim: GradientTransformation = opx.chain(
            opx.scale(-1.0), opx.adam(schedule, nesterov = True) # replace learning_rate with scheduler
        )
        opt_state: OptState = optim.init(dyn)

        def condition(state: Tuple[Self, OptState, Scalar, PRNGKeyArray]):
            # Unpack iteration state
            dyn, opt_state, i, key = state

            return i < max_iters

        def body(state: Tuple[Self, OptState, Scalar, PRNGKeyArray]):
            # Unpack iteration state
            dyn, opt_state, i, key = state

            # Update iteration
            i = i + 1

            # Update PRNG key
            key, _ = jr.split(key)

            # Reconstruct variational
            vari: Self = eqx.combine(dyn, static)

            # Compute gradient of the ELBO for update
            update: M = vari.elbo_grad(grad_draws, batch_size, key)

            # Transform update through optimizer
            update, opt_state = optim.update( # type: ignore
                update, opt_state, eqx.filter(dyn, dyn.filter_spec) # type: ignore
            )

            # Update variational distribution
            dyn: Self = eqx.apply_updates(dyn, update)

            return dyn, opt_state, i, key

        # Run optimization loop
        dyn = lax.while_loop(
            cond_fun=condition,
            body_fun=body,
            init_val=(dyn, opt_state, jnp.array(0, jnp.uint32), key),
        )[0]

        # Return optimized variational
        return eqx.combine(dyn, static)
