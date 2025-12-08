from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer, FlowSpec


def _h(x: Array) -> Array:
    return jnp.tanh(x)

def _dh(x: Array) -> Array:
    return 1.0 - jnp.tanh(x)**2


class PlanarLayer(FlowLayer):
    """
    A Planar flow.

    # Attributes
    - `params`: The parameters of the Planar flow (w, u_hat, b).
    - `constraints`: The constraining transformations for the parameters.
    - `static`: Whether the flow layer is frozen.
    """

    params: Dict[str, Array]
    constraints: Dict[str, Callable[[Array], Array]]
    static: bool

    def __init__(self, dim: int, key: PRNGKeyArray = jr.key(0)):
        """
        Initializes a Planar flow.

        # Parameters
        - `dim`: The dimension of the parameter space.
        """
        self.static = False
        # Split key
        k1, k2, k3 = jr.split(key, 3)

        # Initialize parameters
        self.params = {
            "w": jr.normal(k1, (dim,)) / dim**0.5,
            "u_hat": jr.normal(k2, (dim,)) / dim**0.5,
            "b": jr.normal(k3, ()) / dim**0.5,
        }

        self.constraints = {}

    def __forward(self, draw: Float[Array, " n_dim"]) -> Float[Array, " n_dim"]:
        params = self.transform_params()

        # Extract parameters
        w: Float[Array, " n_dim"] = params["w"]
        u: Float[Array, " n_dim"] = params["u"]
        b: Scalar = params["b"]

        # Compute inner term
        a = draw.dot(w) + b

        # Compute nonlinear stretch
        h = _h(a)

        # Compute forward transformation
        draw = draw + u * h

        return draw

    @eqx.filter_jit
    def forward(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        f = jax.vmap(self.__forward, 0)
        return f(draws)

    def __adjust(self, draw: Float[Array, "n_dim"]) -> Scalar: # noqa
        params = self.transform_params()

        # Extract parameters
        w: Float[Array, " n_dim"] = params["w"]
        u: Float[Array, " n_dim"] = params["u"]
        b: Scalar = params["b"]

        # Compute inner term
        a = draw.dot(w) + b

        # Compute log-Jacobian adjustment
        log_jac: Scalar = jnp.log(1.0 + u.dot(_dh(a) * w))

        assert log_jac.shape == ()
        return log_jac

    @eqx.filter_jit
    def adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        f = jax.vmap(self.__adjust, 0)
        return f(draws)

    def __forward_and_adjust(self, draw: Float[Array, " n_dim"]) -> Tuple[Float[Array, " n_dim"], Scalar]:
        params = self.transform_params()

        # Extract parameters
        w: Float[Array, " n_dim"] = params["w"]
        u: Float[Array, " n_dim"] = params["u"]
        b: Scalar = params["b"]

        # Compute inner term
        a = draw.dot(w) + b

        # Compute forward transformation
        draw = draw + u * _h(a)

        # Compute log-Jacobian adjustment
        log_jac: Scalar = jnp.log(1.0 + u.dot(_dh(a) * w))

        assert len(draw.shape) == 1
        assert log_jac.shape == ()

        return draw, log_jac

    @eqx.filter_jit
    def forward_and_adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Tuple[Float[Array, "n_draws n_dim"], Scalar]:
        f = jax.vmap(self.__forward_and_adjust, 0)
        return f(draws)


    def transform_params(self) -> Dict[str, Array]:
        """
        Applies the affine constraint to u_hat to compute the constrained parameter u,
        and returns all parameters.

        # Returns
        A dictionary of parameters including the computed, constrained 'u'.
        """
        constrained_params = self.constrain_params()

        # Extract parameters
        w: Array = constrained_params["w"]
        u_hat: Array = constrained_params["u_hat"]

        # Compute the constrained u
        u = u_hat + (-1.0 - jnp.dot(w, u_hat)) / (jnp.sum(w**2) + 1e-6) * w
        constrained_params["u"] = u

        return constrained_params

class Planar(FlowSpec):
    """
    A specification for the Planar flow.
    """
    def __init__(self):
        pass

    def construct(self, dim: int) -> PlanarLayer:
        return PlanarLayer(dim)
