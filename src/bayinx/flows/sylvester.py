from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import scan
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer, FlowSpec


def _h(x: Array) -> Array:
    return jnp.tanh(x)

def _dh(x: Array) -> Array:
    return 1.0 - jnp.square(jnp.tanh(x))

def _construct_orthogonal(vectors: Float[Array, " dim rank"]) -> Float[Array, " dim rank"]:
    """
    Constructs a thin orthogonal matrix using Householder reflections.
    """
    D, M = vectors.shape

    # Initialize Q as the thin identity matrix
    Q_initial = jnp.eye(D, M)

    def apply_reflection(Q_current, v_k):
        v_norm_sq = jnp.sum(v_k**2)
        tau = 2.0 / v_norm_sq

        a_T = jnp.dot(v_k, Q_current)

        update = jnp.outer(v_k, tau * a_T)

        Q_next = Q_current - update

        return Q_next, None

    Q_final, _ = scan(
        f=apply_reflection,
        init=Q_initial,
        xs=vectors.T
    )

    return Q_final


class SylvesterLayer(FlowLayer):
    """
    A Sylvester flow.

    # Attributes
    - `params`: Dictionary containing raw parameters.
    - `constraints`: Dictionary of constraining transformations.
    - `static`: Whether the flow layer is frozen.
    - `rank`: The rank (M) of the transformation.
    """

    rank: int
    params: Dict[str, Array]
    constraints: Dict[str, Callable[[Array], Array]]
    static: bool

    def __init__(self, dim: int, rank: int, key: PRNGKeyArray = jr.key(0)):
        """
        Initializes the Sylvester flow.

        # Parameters
        - `dim`: The dimension of the parameter space (D).
        - `rank`: The number of hidden units/reflections (M).
        """
        self.static = False
        self.rank = rank

        # Split key
        k1, k2, k3, k4 = jr.split(key, 4)

        # Initialize parameters
        self.params = {
            "Q": jr.normal(k1, (dim, rank)) * 0.1 / (dim * rank)**0.5,
            "R1": jr.normal(k2, (rank, rank)) * 0.1 / rank,
            "R2": jr.normal(k3, (rank, rank)) * 0.1 / rank,
            "shift": jr.normal(k4, (rank,)) * 0.1 / rank**0.5,
            "scale": 0.55 + jr.normal(k4, (rank,)) * 0.1 / rank**0.5
        }

        self.constraints = {
            "Q": lambda hvecs: hvecs / jnp.linalg.norm(hvecs, axis = 0),
            "R1": jnp.triu,
            "R2": jnp.triu,
            "scale": jnp.exp
        }

    def transform_params(self) -> Dict[str, Array]:
        """
        Constructs the orthonormal matrix Q, and constrains R2 based on R1 to ensure invertibility.
        """
        params = self.constrain_params()
        eps = jnp.finfo(jnp.array(0.0)).eps.item()

        # Extract relevant parameters
        hvecs = params["Q"]
        R1 = params["R1"]
        R2 = params["R2"]

        # Construct orthogonal matrix from householder vectors
        params["Q"] = _construct_orthogonal(hvecs)

        # Constrain diagonals of R2 to ensure invertibility
        R1_diag = jnp.diag(R1)
        R1_diag = jnp.sign(R1_diag) * (jnp.maximum(jnp.abs(R1_diag), eps))
        R2_diag = (jnp.exp(jnp.diag(R2)) - 1.0) / R1_diag
        params["R2"] = jnp.fill_diagonal(R2, R2_diag, inplace = False)

        return params

    def __forward(self, draw: Float[Array, " n_dim"], params: Dict[str, Array]) -> Float[Array, " n_dim"]:
        # Extract parameters
        Q: Float[Array, "dim rank"] = params["Q"]
        R1: Float[Array, "rank rank"] = params["R1"]
        R2: Float[Array, "rank rank"] = params["R2"]
        shift: Float[Array, " rank"] = params["shift"]
        scale: Float[Array, " rank"] = params["scale"]

        # Compute forward transform
        draw = draw + Q @ R1 @ (scale * _h(R2 @ Q.T @ (draw - shift) / scale))

        return draw

    @eqx.filter_jit
    def forward(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        # Extract constrained parameters
        params = self.transform_params()

        f = jax.vmap(self.__forward, (0, None))
        return f(draws, params)

    def __adjust(self, draw: Float[Array, " n_dim"], params: Dict[str, Array]) -> Scalar:
        # Extract parameters
        Q: Float[Array, "dim rank"] = params["Q"]
        R1: Float[Array, "rank rank"] = params["R1"]
        R2: Float[Array, "rank rank"] = params["R2"]
        shift: Float[Array, " rank"] = params["shift"]
        scale: Float[Array, " rank"] = params["scale"]

        # Compute inner derivative term
        dh_term = _dh(R2 @ Q.T @ (draw - shift) / scale)

        # Extract diagonals
        R1_diag = jnp.diag(R1)
        R2_diag = jnp.diag(R2)

        # Compute log-Jacobian adjustment
        derivs = 1 + dh_term * R2_diag * R1_diag
        log_jac = -jnp.sum(jnp.log(jnp.abs(derivs)))

        assert log_jac.shape == ()

        return log_jac

    @eqx.filter_jit
    def adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        # Extract constrained parameters
        params = self.transform_params()

        f = jax.vmap(self.__adjust, (0, None))
        return f(draws, params)

    def __forward_and_adjust(self, draw: Float[Array, " n_dim"], params: Dict[str, Array]) -> Tuple[Float[Array, " n_dim"], Scalar]:
        # Extract parameters
        Q: Float[Array, "dim rank"] = params["Q"]
        R1: Float[Array, "rank rank"] = params["R1"]
        R2: Float[Array, "rank rank"] = params["R2"]
        shift: Float[Array, " rank"] = params["shift"]
        scale: Float[Array, " rank"] = params["scale"]

        # Compute shared inner term
        shared_term = R2 @ Q.T @ (draw - shift) / scale

        # Compute derivative term
        dh_term = _dh(shared_term)

        # Extract diagonals
        R1_diag = jnp.diag(R1)
        R2_diag = jnp.diag(R2)

        # Compute log-Jacobian adjustment
        derivs = 1 + dh_term * R2_diag * R1_diag
        log_jac = -jnp.sum(jnp.log(jnp.abs(derivs)))

        # Compute forward transform
        draw = draw + Q @ R1 @ (scale * _h(shared_term))

        assert len(draw.shape) == 1
        assert log_jac.shape == ()

        return draw, log_jac

    @eqx.filter_jit
    def forward_and_adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Tuple[Float[Array, "n_draws n_dim"], Scalar]:
        # Extract constrained parameters
        params = self.transform_params()

        f = jax.vmap(self.__forward_and_adjust, (0, None))
        return f(draws, params)

class Sylvester(FlowSpec):
    """
    A specification for the Sylvester flow.

    Definition:
        $T(\\mathbf{z}) = \\mathbf{z} + \\mathbf{Q} \\mathbf{R}_1 h(\\mathbf{R}_2 \\mathbf{Q}^\\top \\mathbf{z} + \\mathbf{b})$

    Attributes:
        rank: The rank of the Sylvester flow (number of hidden units).
        key: The PRNG key used to generate a Sylvester flow layer.
    """
    rank: int
    key: PRNGKeyArray

    def __init__(self, rank: int, key: PRNGKeyArray = jr.key(0)):
        """
        Initializes the specification for a Sylvester flow.

        Parameters:
            rank: The rank of the transformation (number of hidden units).
            key: A PRNG key used to generate the flow layer.
        """
        self.rank = rank
        self.key = key

    def construct(self, dim: int) -> SylvesterLayer:
        """
        Constructs a Sylvester flow layer.

        Parameters:
            dim: The dimension of the parameter space.

        Returns:
            A SylvesterLayer of dimension `dim` and rank `self.rank`.
        """
        if self.rank > dim:
            raise ValueError(f"Rank {self.rank} cannot be greater than dimension {dim}.")

        return SylvesterLayer(dim, self.rank, self.key)
