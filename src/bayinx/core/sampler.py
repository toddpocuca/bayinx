from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Self

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree

from bayinx.core.model import Model


class Sampler[M: Model](eqx.Module):
    """
    An abstract base class used to define sampling methods.

    Attributes:
        dim: The dimension of the parameter space.
        _unflatten: A function to transform flattened draws back to the structure of the `Model`.
        _static: The static component of a partitioned `Model` used to initialize the `Sampler` object.
    """
    dim: int
    _unflatten: Callable[[Array], M]
    _static: M

    @property
    @abstractmethod
    def filter_spec(self) -> Self:
        """
        Filter specification for dynamic and static components of the `Sampler` object.
        """
        pass

    @abstractmethod
    def sample_draws(self, n_draws: int, key: PRNGKeyArray = jr.PRNGKey(0), *args, **kwargs) -> Any:
        """
        Sample (flattened) posterior draws.
        """
        pass

    @abstractmethod
    def sample_parameter(self, name: str, n_draws: int, key: PRNGKeyArray = jr.PRNGKey(0), *args, **kwargs) -> Any:
        """
        Sample a parameter from the posterior.
        """
        pass

    @abstractmethod
    def sample_predictive(self, func: Callable[[M, PRNGKeyArray], PyTree[Array]], n_draws: int, key: PRNGKeyArray = jr.PRNGKey(0), *args, **kwargs) -> Any:
        """
        Sample from the posterior predictive.
        """
        pass

    def reconstruct_model(self, draw: Array) -> M:
        # Unflatten variational draw
        model: M = self._unflatten(draw)

        # Combine with static components
        model: M = eqx.combine(model, self._static)

        return model

    @partial(jax.vmap, in_axes=(None, 0))
    def eval_model(self, draws: Array) -> Array:
        """
        Reconstruct models from flattened draws and evaluate their posterior log density.

        Parameters:
            draws: A set of draws.
        """
        # Unflatten variational draw
        model: M = self.reconstruct_model(draws)

        # Evaluate posterior
        return model()
