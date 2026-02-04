from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree, Scalar

from bayinx.core.constraint import Constraint


class Identity(Constraint):
    """
    Does nothing.
    """

    @eqx.filter_jit
    def constrain[T: PyTree](self, obj: T, filter_spec: PyTree) -> Tuple[T, Scalar]:
        """
        Applies the identity transformation (does nothing) and computes its log-Jacobian adjustment (0).

        # Returns
        A tuple containing:
            - The same `PyTree`.
            - A scalar `Array` containing 0.
        """
        log_jac: Scalar = jnp.array(0.0)

        return obj, log_jac

    def check[T: PyTree](self, obj: T, filter_spec: PyTree) -> bool:
        """
        Checks for nothing.

        # Returns
        `True`
        """
        return True
