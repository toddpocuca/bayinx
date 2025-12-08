from abc import abstractmethod
from typing import TYPE_CHECKING, Tuple

import equinox as eqx
from jaxtyping import PyTree, Scalar

if TYPE_CHECKING:
    from bayinx.core.types import T


class Constraint(eqx.Module):
    """
    Abstract base class for defining constraints (for stochastic nodes).
    """

    @abstractmethod
    def constrain(self, obj: "T", filter_spec: PyTree) -> Tuple["T", Scalar]:
        """
        Applies the constraining transformation to the leaves of a `PyTree` and
        computes the log-Jacobian adjustment of the transformation.

        # Parameters
        - `obj`: The unconstrained values.
        - `filter_spec`: The filter specification for `obj`.

        # Returns
        A tuple containing:
            - A `PyTree` with its leaves now constrained.
            - A scalar `Array` containing the log-Jacobian adjustment of the
                transformation.
        """
        pass

    @abstractmethod
    def check(self, obj: "T", filter_spec: PyTree) -> bool:
        """
        Checks if the constraint is held for all leaves of a `PyTree`.

        # Parameters
        - `obj`: The unconstrained values.
        - `filter_spec`: The filter specification for `obj`.

        # Returns
        A tuple containing:
            - A `PyTree` with its leaves now constrained.
            - A scalar `Array` containing the log-Jacobian adjustment of the
                transformation.
        """
        pass
