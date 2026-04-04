from typing import Callable, Dict, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer, FlowSpec


class DiagAffineLayer(FlowLayer):
    """
    A diagonal (element-wise) affine flow layer.

    # Attributes
    - `params`: The parameters of the diagonal affine flow.
    - `constraints`: The constraining transformations for the parameters of the diagonal affine flow.
    - `static`: Whether the flow layer is frozen (parameters are not subject to further optimization).
    """

    params: Dict[str, Array]
    constraints: Dict[str, Callable[[Array], Array]]
    static: bool
    dim: int

    def __init__(self, dim: int, key: PRNGKeyArray):
        """
        Initializes a diagonal affine flow.

        # Parameters
        - `dim`: The dimension of the parameter space.
        """
        self.static = False
        self.dim = dim

        # Split key
        k1, k2 = jr.split(key)

        # Initialize parameters
        self.params = {
            "shift": jr.normal(k1, (dim,)) / dim**0.5,
            "scale": jr.normal(k2, (dim,)) / dim**0.5,
        }

        # Define constraints
        self.constraints = {"scale": jnp.exp}

    @eqx.filter_jit
    def forward(self, draws: Float[Array, "n_draws n_dims"]) -> Float[Array, "n_draws n_dims"]:
        # Get constrained parameters
        params = self.transform_params()

        # Extract relevant parameters
        shift: Float[Array, " n_dims"] = params["shift"]
        scale: Float[Array, " n_dims"] = params["scale"]

        # Compute forward transformation
        draws = draws * scale + shift

        return draws

    @eqx.filter_jit
    def reverse(self, draws: Float[Array, "n_draws n_dims"]) -> Float[Array, "n_draws n_dims"]:
        # Get constrained parameters
        params = self.transform_params()

        # Extract relevant parameters
        shift: Float[Array, " n_dims"] = params["shift"]
        scale: Float[Array, " n_dims"] = params["scale"]

        return (draws - shift) / scale

    @eqx.filter_jit
    def forward_and_adjust(self, draws: Float[Array, "n_draws n_dims"]) -> Tuple[Float[Array, "n_draws n_dims"], Scalar]:
        # Get constrained parameters
        params = self.transform_params()

        # Extract relevant parameters
        shift: Float[Array, " n_dims"] = params["shift"]
        scale: Float[Array, " n_dims"] = params["scale"]

        # Compute log-Jacobian adjustments
        log_jacs: Array = jnp.full(draws.shape[0], jnp.log(scale).sum())

        # Compute forward transformation
        draws = draws * scale + shift

        # Shape checks
        assert len(draws.shape) == 2
        assert len(log_jacs.shape) == 1

        return draws, log_jacs

    @eqx.filter_jit
    def reverse_and_adjust(self, draws: Float[Array, "n_draws n_dims"]) -> Tuple[Float[Array, "n_draws n_dims"], Float[Array, " n_draws"]]:
        # Get constrained parameters
        params = self.transform_params()

        # Extract relevant parameters
        shift: Float[Array, " n_dims"] = params["shift"]
        scale: Float[Array, " n_dims"] = params["scale"]

        # Compute log-Jacobian adjustments
        log_jacs: Array = jnp.full(draws.shape[0], -jnp.log(scale).sum())

        # Compute reverse transformation
        draws = (draws - shift) / scale

        return draws, log_jacs

class DiagAffine(FlowSpec):
    """
    A specification for the diagonal affine flow.

    Definition:
        $T(\\mathbf{z}) = \\mathbf{d} \\odot \\mathbf{z} + \\mathbf{c}$

        Where $\\mathbf{z} \\in \\mathbb{R}^D$, $\\mathbf{d} \\in \\mathbb{R}^{D}$ is non-negative, and $\\mathbf{c} \\in \\mathbb{R}^D$.

    Attributes:
        key: The PRNG key used to generate the diagonal affine flow layer.
    """
    key: PRNGKeyArray
    def __init__(self, key: PRNGKeyArray = jr.key(0)):
        self.key = key

    def construct(self, dim: int) -> DiagAffineLayer:
        return DiagAffineLayer(dim, self.key)
