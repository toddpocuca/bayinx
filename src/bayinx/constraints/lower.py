from typing import Any, Tuple

import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree, Scalar, ScalarLike

from bayinx.core.constraint import Constraint
from bayinx.core.types import T


class Lower(Constraint):
    """
    Enforces a lower bound on the parameter.

    # Attributes
    - `lb`: The lower bound.
    """

    lb: Scalar

    def __init__(self, lb: ScalarLike):
        self.lb = jnp.asarray(lb)

    def constrain(self, obj: T, filter_spec: T) -> Tuple[T, Scalar]:
        """
        Applies the exponential transformation to the leaves of a `PyTree` and
        computes the log-Jacobian adjustment of the transformation.

        # Parameters
        - `x`: The unconstrained `PyTree`.

        # Returns
        A tuple containing:
            - A `PyTree` with its values `x` now satisfying `lb <= x`.
            - A scalar `Array` containing the log-absolute-Jacobian of the
                transformation.
        """
        log_jac: Scalar = jnp.array(0.0)

        def constrain_leaf(leaf: Any, filter: bool):
            nonlocal log_jac  # Reference outer variable

            if filter:
                # Apply transformation
                constrained = jnp.exp(leaf) + self.lb

                # Accumulate Jacobian adjustment
                log_jac = log_jac + jnp.sum(leaf)

                return constrained
            else:
                return leaf

        # Constrain leaves
        obj = jt.map(constrain_leaf, obj, filter_spec)

        return obj, log_jac

    def check(self, obj: T, filter_spec: PyTree) -> bool:
        """
        Checks if all relevant leaves of `obj` are greater than or equal to `lb`.
        """
        def check_leaf(leaf: Any, filter: bool):
            if filter:
                # Check constraint
                return jnp.all(leaf >= self.lb)
            else:
                return True

        # Check leaves
        obj = jt.map(check_leaf, obj, filter_spec)
        return jt.all(obj)

    def __repr__(self):
        return f"Lower({self.lb.item()})"
