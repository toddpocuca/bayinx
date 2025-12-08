from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer, FlowSpec


class FullAffineLayer(FlowLayer):
    """
    A full affine flow.

    # Attributes
    - `params`: The parameters of the full affine flow.
    - `constraints`: The constraining transformations for the parameters of the full affine flow.
    - `static`: Whether the flow layer is frozen (parameters are not subject to further optimization).
    """

    params: Dict[str, Array]
    constraints: Dict[str, Callable[[Array], Array]]
    static: bool


    def __init__(self, dim: int, key: PRNGKeyArray):
        """
        Initializes a full affine flow.

        # Parameters
        - `dim`: The dimension of the parameter space.
        """
        self.static = False

        # Split key
        k1, k2 = jr.split(key)

        # Initialize parameters
        self.params = {
            "shift": jr.normal(key, (dim, )) / dim**0.5,
            "scale": jr.normal(key, (dim, dim)) / dim**0.5,
        }

        # Define constraints
        if dim == 1:
            self.constraints = {"scale": jnp.exp}
        else:
            def constrain_scale(scale: Array):
                # Extract diagonal and apply exponential
                diag: Array = jnp.exp(jnp.diag(scale))

                # Return matrix with modified diagonal
                return jnp.fill_diagonal(jnp.tril(scale), diag, inplace=False)

            self.constraints = {"scale": constrain_scale}

    @eqx.filter_jit
    def forward(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        params = self.transform_params()

        # Extract parameters
        shift: Float[Array, " n_dim"] = params["shift"]
        scale: Float[Array, "n_dim n_dim"] = params["scale"]

        # Compute forward transformation
        draws = (scale @ draws.T).T + shift

        return draws

    def __adjust(self, draw: Float[Array, " n_dim"]) -> Float[Array, " n_dim"]:
        params = self.transform_params()

        # Extract parameters
        scale: Float[Array, "n_dim n_dim"] = params["scale"]

        # Compute log-Jacobian adjustments
        log_jac: Array = jnp.log(jnp.diag(scale)).sum()

        assert log_jac.shape == ()

        return log_jac

    @eqx.filter_jit
    def adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        f = jax.vmap(self.__adjust, 0)
        return f(draws)

    def __forward_and_adjust(self, draw: Float[Array, " n_dim"]) -> Tuple[Float[Array, " n_dim"], Scalar]:
        params = self.transform_params()

        assert len(draw.shape) == 1

        # Extract parameters
        shift: Float[Array, " n_dim"] = params["shift"]
        scale: Float[Array, "n_dim n_dim"] = params["scale"]

        # Compute forward transformation
        draw = (scale @ draw.T).T + shift

        assert len(draw.shape) == 1

        # Compute log_jac
        log_jac: Scalar = jnp.log(jnp.diag(scale)).sum()

        assert log_jac.shape == ()

        return draw, log_jac

    @eqx.filter_jit
    def forward_and_adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Tuple[Float[Array, "n_draws n_dim"], Scalar]:
        f = jax.vmap(self.__forward_and_adjust, 0)
        return f(draws)


class FullAffine(FlowSpec):
    key: PRNGKeyArray

    def __init__(self, key: PRNGKeyArray = jr.key(0)):
        self.key = key

    def construct(self, dim: int) -> FullAffineLayer:
        return FullAffineLayer(dim, self.key)
