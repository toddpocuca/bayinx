from typing import Any, Tuple

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree, Scalar

from bayinx.core.constraint import Constraint


class Interval(Constraint):
    """
    Enforces that the parameter lies in the (lb, ub) interval using a scaled
    and shifted sigmoid transformation.

    # Attributes
    - `lb`: The lower bound.
    - `ub`: The upper bound.
    """

    lb: Scalar
    ub: Scalar

    def __init__(self, lb: int | float | Scalar, ub: int | float | Scalar):
        lb, ub = float(lb), float(ub)

        self.lb = jnp.asarray(lb)
        self.ub = jnp.asarray(ub)

    @eqx.filter_jit(donate = 'all')
    def constrain[T: PyTree](self, obj: T, filter_spec: PyTree) -> Tuple[T, Scalar]:
        """
        Applies the scaled sigmoid transformation to the leaves of a `PyTree` and
        computes the log-Jacobian adjustment.

        # Parameters
        - `obj`: The unconstrained `PyTree` (values are in R).

        # Returns
        A tuple containing:
            - A `PyTree` with each leaf `x` now satisfying `lb < x < ub`.
            - A scalar `Array` containing the log-absolute-Jacobian of the
              transformation.
        """
        log_jac: Scalar = jnp.array(0.0)

        def constrain_leaf(leaf: Any, filter: bool):
            nonlocal log_jac  # Reference outer variable

            if filter:
                # Apply transformation ----
                constrained = self.lb + (self.ub - self.lb) * jnn.sigmoid(-leaf)

                # Accumulate log-Jacobian adjustment ----
                log_jac = log_jac + (jnp.log(constrained - self.lb) +
                    jnp.log(self.ub - constrained) -
                    jnp.log(self.ub - self.lb)
                ).sum()

                return constrained
            else:
                return leaf

        # Constrain leaves
        obj = jt.map(constrain_leaf, obj, filter_spec)

        return obj, log_jac

    def check[T: PyTree](self, obj: T, filter_spec: PyTree) -> bool:
        """
        Checks if all relevant leaves of `obj` are in the interval [lb, ub].
        """
        def check_leaf(leaf: Any, filter: bool):
            if filter:
                # Check constraint
                return jnp.all((self.lb <= leaf) & (leaf <= self.ub))
            else:
                return True

        # Check leaves
        obj = jt.map(check_leaf, obj, filter_spec)
        return jt.all(obj)

    def __repr__(self):
        return f"Interval({self.lb.item(), self.ub.item()})"
