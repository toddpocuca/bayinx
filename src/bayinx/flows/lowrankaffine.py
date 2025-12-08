from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import scan
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer, FlowSpec


def _mat_op(
    carry: Float[Array, " rank"],
    vars: Tuple[
        Float[Array, " rank"],
        Float[Array, " rank"],
        Scalar,
        Scalar,
        Scalar
    ]
) -> Tuple[Float[Array, " rank"], Scalar]:
    """
    The implicit matrix operations compute:

        x = Lz + b = (D + W)z + b = Dz + (UV^T * M)z + b

    Where W = UV^T * M is the low-rank representation of the strictly lower triangular matrix W,
        parameterized by U and V, and an implicit mask M that zeros out upper triangular elements.

    In the implementation only U, V, diag(D), and b are needed to complete the matrix operations.
    """
    # Unpack state
    U_r, V_r, z_i, d_i, b_i = vars
    h_i = carry

    # Compute partial product
    h_next = h_i + z_i * V_r

    # Compute i-th element of transformed draw:
    x_i = b_i + z_i * d_i + U_r.dot(h_i)

    return h_next, x_i

class LowRankAffineLayer(FlowLayer):
    """
    A low-rank affine flow.

    # Attributes
    - `params`: A dictionary containing the shift and low-rank representation of the scale parameters.
    - `constraints`: A dictionary of constraining transformations.
    - `static`: Whether the flow layer is frozen (parameters are not subject to further optimization).
    - `rank`: Rank of the scale transformation.
    """

    rank: int

    def __init__(self, dim: int, rank: int, key: PRNGKeyArray = jr.key(0)):
        """
        Initializes a low-rank affine flow.

        # Parameters
        - `dim`: The dimension of the parameter space.
        - `rank`: The rank of the (implicit) scale matrix.
        """
        self.static = False
        self.rank = rank

        # Split key
        k1, k2, k3, k4 = jr.split(key, 4)

        # Initialize parameters
        self.params = {
            "shift": jr.normal(k1, (dim, )) / dim**0.5,
            "diag_scale": jr.normal(k2, (dim, )) / dim**0.5,
            "offdiag_scale": (
                jr.normal(k3, (dim, rank)) / dim**0.5,
                jr.normal(k4, (dim, rank)) / dim**0.5
            )
        }

        # Define constraints
        self.constraints = {"diag_scale": jnp.exp}


    def __forward(self, draw: Float[Array, " n_dim"]) -> Float[Array, " n_dim"]:
        params = self.transform_params()

        # Extract parameters
        shift: Array = params["shift"]
        diag: Array = params["diag_scale"]
        U, V = params["offdiag_scale"]

        # Compute forward transformation
        _, draw = scan(
            f=_mat_op,
            init=jnp.zeros((self.rank, )),
            xs= (U, V, draw, diag, shift)
        )

        return draw

    @eqx.filter_jit
    def forward(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        f = jax.vmap(self.__forward, 0)
        return f(draws)

    def __adjust(self, draw: Float[Array, " n_dim"]) -> Scalar:
        params = self.transform_params()

        diag: Array = params["diag_scale"]

        # Compute log-Jacobian adjustment
        log_jac: Scalar = jnp.log(diag).sum()

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
        shift: Array = params["shift"]
        diag: Array = params["diag_scale"]
        U, V = params["offdiag_scale"]

        # Compute log-Jacobian adjustment
        log_jac: Scalar = jnp.log(diag).sum()

        # Compute forward transformation
        _, draw = scan(
            f=_mat_op,
            init=jnp.zeros((self.rank, )),
            xs= (U, V, draw, diag, shift)
        )

        assert len(draw.shape) == 1
        assert log_jac.shape == ()

        return draw, log_jac

    @eqx.filter_jit
    def forward_and_adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Tuple[Float[Array, "n_draws n_dim"], Scalar]:
        f = jax.vmap(self.__forward_and_adjust, 0)
        return f(draws)

class LowRankAffine(FlowSpec):
    rank: int

    def __init__(self, rank: int):
        self.rank = rank

    def construct(self, dim: int) -> LowRankAffineLayer:
        if (self.rank > (dim - 1)/2):
            raise ValueError(f"Rank {self.rank} is large, consider using a full affine flow instead.")

        return LowRankAffineLayer(dim, self.rank)
