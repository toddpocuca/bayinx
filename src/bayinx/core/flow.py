from abc import ABC, abstractmethod
from typing import Callable, Dict, Self, Tuple

import equinox as eqx
import jax.tree as jt
import jax.tree_util as jtu
from jaxtyping import Array, PyTree


class FlowLayer(eqx.Module):
    """
    An abstract base class for a flow layer.

    Attributes:
        params: The parameters of the diffeomorphism.
        constraints: The parameter-wise constraining transformations for each parameter.
        static: Whether the flow layer is frozen (parameters are not subject to further optimization).
    """

    params: dict[str, PyTree]
    constraints: dict[str, Callable[[PyTree], Array]]
    static: bool
    dim: int

    @abstractmethod
    def forward(self, draws: Array) -> Array:
        """
        Computes the forward transformation of `draws`.
        """
        pass

    @abstractmethod
    def reverse(self, draws: Array) -> Array:
        """
        Computes the reverse transformation of `draws`.
        """
        pass

    @abstractmethod
    def forward_and_adjust(self, draws: Array) -> Tuple[Array, Array]:
        """
        Computes the log-Jacobian adjustment at `draws` and applies the forward transformation.

        # Returns
            A tuple containing the transformed draws and log-Jacobian adjustments.
        """
        pass

    @abstractmethod
    def reverse_and_adjust(self, draws: Array) -> Tuple[Array, Array]:
        """
        Computes the log-Jacobian adjustment at `draws` and applies the reverse transformation.

        # Returns
            A tuple containing the transformed draws and log-Jacobian adjustments.
        """
        pass

    # Default filter specification
    @property
    def filter_spec(self) -> "FlowLayer":
        """
        Generates a filter specification to subset relevant parameters for the flow.
        """
        # Generate empty specification
        filter_spec = jtu.tree_map(lambda _: False, self)

        if self.static is False:
            # Specify parameters
            filter_spec = eqx.tree_at(
                lambda flow: flow.params,
                filter_spec,
                replace=jt.map(eqx.is_inexact_array, self.params),
            )

        return filter_spec

    def constrain_params(self: Self) -> Dict[str, PyTree]:
        """
        Constrain flow parameters to the appropriate domain.

        # Returns
        The constrained parameters of the diffeomorphism.
        """
        params = self.params.copy()

        for par, map in self.constraints.items():
            params[par] = map(params[par])

        return params

    def transform_params(self: Self) -> Dict[str, PyTree]:
        """
        Apply a custom transformation to `params` if needed. Defaults to `constrain_params()`.

        # Returns
        The transformed parameters of the diffeomorphism.
        """
        return self.constrain_params()


class FlowSpec(ABC):
    """
    A specification for a flow layer.
    """

    @abstractmethod
    def construct(self, dim: int) -> FlowLayer: ...
