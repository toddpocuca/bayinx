from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer, FlowSpec


class DiagAffineLayer(FlowLayer):
    """
    A diagonal (element-wise) affine flow.

    # Attributes
    - `params`: The parameters of the diagonal affine flow.
    - `constraints`: The constraining transformations for the parameters of the diagonal affine flow.
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
            "shift": jr.normal(k1, (dim,)) / dim**0.5,
            "scale": jr.normal(k2, (dim,)) / dim**0.5,
        }

        # Define constraints
        self.constraints = {"scale": jnp.exp}


    def __forward(self, draw: Float[Array, " D"]) -> Float[Array, " D"]:
        params = self.transform_params()

        assert len(draw.shape) == 1

        # Extract parameters
        shift: Float[Array, " dim"] = params["shift"]
        scale: Float[Array, " dim"] = params["scale"]

        # Compute forward transformation
        draw = draw * scale + shift

        return draw

    @eqx.filter_jit
    def forward(self, draws: Float[Array, "draws dim"]) -> Float[Array, "draws dim"]:
        f = jax.vmap(self.__forward, 0)
        return f(draws)

    def __adjust(self, draw: Float[Array, " dim"]) -> Float[Array, " dim"]:
        params = self.transform_params()

        # Extract parameters
        scale: Float[Array, " dim"] = params["scale"]

        # Compute log-Jacobian adjustment
        log_jac: Array = jnp.log(scale).sum()

        assert log_jac.shape == ()

        return log_jac

    @eqx.filter_jit
    def adjust(self, draws: Float[Array, "draws dim"]) -> Float[Array, "draws dim"]:
        f = jax.vmap(self.__adjust, 0)
        return f(draws)

    def __forward_and_adjust(self, draw: Float[Array, " dim"]) -> Tuple[Float[Array, " dim"], Scalar]:
        params = self.transform_params()

        assert len(draw.shape) == 1

        # Extract parameters
        shift: Float[Array, " dim"] = params["shift"] # noqa
        scale: Float[Array, " dim"] = params["scale"] # noqa

        # Compute forward transformation
        draw = scale * draw + shift

        assert len(draw.shape) == 1

        # Compute log-Jacobian adjustment
        log_jac: Scalar = jnp.log(scale).sum()

        assert log_jac.shape == ()

        return draw, log_jac

    @eqx.filter_jit
    def forward_and_adjust(self, draws: Float[Array, "draws dim"]) -> Tuple[Float[Array, "draws dim"], Scalar]:
        f = jax.vmap(self.__forward_and_adjust, 0)
        return f(draws)


class DiagAffine(FlowSpec):
    """
    A specification for the diagonal affine flow.
    """
    key: PRNGKeyArray
    def __init__(self, key: PRNGKeyArray = jr.key(0)):
        self.key = key

    def construct(self, dim: int) -> DiagAffineLayer:
        return DiagAffineLayer(dim, self.key)
