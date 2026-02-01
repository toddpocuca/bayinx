from typing import Any, Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree, Scalar

from bayinx.core.constraint import Constraint


class LogSimplex(Constraint):
    """
    Enforces a log-simplex constraint on a node.

    # Attributes
    - `total`: The total sum in the exponentiated space.
    """

    total: Scalar

    def __init__(self, total: bool | int | float | Scalar = 1.0):
        self.total = jnp.asarray(float(total))

    def constrain[T: PyTree](self, obj: T, filter_spec: PyTree) -> Tuple[T, Scalar]:
        """
        Applies a partial inverse isometric log-ratio transformation to the leaves of a `PyTree` and computes the log-Jacobian adjustment.

        # Parameters
        - `obj`: The unconstrained `PyTree` (values are in R).

        # Returns
        A tuple containing:
            - A `PyTree` with each leaf `x` now satisfying `sum(exp(x), axis = -1) == total`.
            - A scalar `Array` containing the log-absolute-Jacobian of the
              transformation.
        """
        log_jac: Scalar = jnp.array(0.0)

        def constrain_leaf(leaf: Any, filter: bool):
            if not filter:
                return leaf

            # Apply constraining transformation ----
            N = leaf.shape[-1]

            # Construct centred basis
            idxs = jnp.arange(1, N + 1)
            scaled_leaf = leaf * jnp.reciprocal(jnp.sqrt(idxs * (idxs + 1)))

            # Compute reverse cumulative sum
            s = jnp.flip(
                jnp.cumsum(
                    jnp.flip(
                        scaled_leaf, axis=-1
                    ), axis=-1
                ), axis=-1
            )
            s = jnp.pad(s, ((0, 0),) * (s.ndim - 1) + ((0, 1),))

            # Construct zero-sum vector
            z = jnp.concatenate([s[..., 0:1], s[..., 1:] - (idxs * scaled_leaf)], axis=-1)

            # Compute constrained leaf
            log_norm_simplex = jnn.log_softmax(z, axis=-1)
            constrained = log_norm_simplex + jnp.log(self.total)

            # Accumulate log-Jacobian adjustment ----
            nonlocal log_jac
            log_jac_const = 0.5 * jnp.log(N + 1)

            # Sum contributions
            log_jac += jnp.sum(log_jac_const)

            return constrained

        obj = jt.map(constrain_leaf, obj, filter_spec)
        return obj, log_jac

    def check[T: PyTree](self, obj: T, filter_spec: PyTree) -> bool:
        eps = jnp.finfo(jnp.array(0.0)).eps

        def check_leaf(leaf: Any, filter: bool):
            if not filter:
                return True

            exp_sums_to_total = jnp.allclose(jnp.sum(jnp.exp(leaf), axis=-1), self.total, rtol=eps)

            return exp_sums_to_total

        checked: PyTree[bool] = jt.map(check_leaf, obj, filter_spec)
        return jt.all(checked)
