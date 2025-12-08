from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import scan
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer, FlowSpec


def _h(x: Array) -> Array:
    """Non-linearity for Sylvester flow."""
    return jnp.tanh(x)


def _dh(x: Array) -> Array:
    """Derivative of the non-linearity."""
    return 1.0 - jnp.tanh(x)**2


def _construct_orthogonal(vectors: Float[Array, " dim rank"]) -> Float[Array, " dim rank"]:
    """
    Constructs a D x M orthogonal matrix (Q) using Householder reflections.
    """
    D, M = vectors.shape

    # Initialize Q as the thin identity matrix (first M columns of I_D)
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

        k1, k2, k3, k4 = jr.split(key, 4)

        # Initialize parameters
        # hvecs: D x M matrix where each column is a Householder vector
        self.params = {
            "hvecs": jr.normal(k1, (dim, rank)) / dim**0.5,
            "r1": jr.normal(k2, (rank, rank)) / rank**0.5,
            "r2": jr.normal(k3, (rank, rank)) / rank**0.5,
            "b": jr.normal(k4, (rank,)) / rank**0.5,
        }

        # Constraint for Upper Triangular matrices with positive diagonal
        def constrain_triangular(matrix: Array):
            # Extract diagonal and apply exponential to ensure invertibility
            diag: Array = jnp.exp(jnp.diag(matrix))
            # Return upper triangular matrix with modified diagonal
            return jnp.fill_diagonal(jnp.triu(matrix), diag, inplace=False)

        self.constraints = {
            "r1": constrain_triangular,
            "r2": constrain_triangular
        }

    def transform_params(self) -> Dict[str, Array]:
        """
        Applies constraints and constructs the orthogonal matrix Q.
        """
        constrained = self.constrain_params()

        # Construct orthogonal matrix
        q = _construct_orthogonal(constrained["hvecs"])
        constrained["q"] = q

        return constrained

    def __forward(self, draw: Float[Array, " n_dim"]) -> Float[Array, " n_dim"]:
        params = self.transform_params()

        # Extract parameters
        Q: Float[Array, "dim rank"] = params["q"]
        R1: Float[Array, "rank rank"] = params["r1"]
        R2: Float[Array, "rank rank"] = params["r2"]
        b: Float[Array, " rank"] = params["b"]

        # Compute inner terms
        y = R2.dot(Q.T.dot(draw)) + b
        h_y = _h(y)

        # Compute forward transform
        draw = draw + Q.dot(R1.dot(h_y))

        return draw

    @eqx.filter_jit
    def forward(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        f = jax.vmap(self.__forward, 0)
        return f(draws)

    def __adjust(self, draw: Float[Array, " n_dim"]) -> Scalar:
        params = self.transform_params()

        # Extract parameters
        Q: Float[Array, "dim rank"] = params["q"]
        R1: Float[Array, "rank rank"] = params["r1"]
        R2: Float[Array, "rank rank"] = params["r2"]
        b: Float[Array, " rank"] = params["b"]

        # Recompute the argument to the nonlinearity
        term = R2.dot(Q.T.dot(draw)) + b
        diag_h_prime = _dh(term)

        # Diagonal of R1 and R2
        d_r1 = jnp.diag(R1)
        d_r2 = jnp.diag(R2)

        # Diagonal term of the matrix (I + diag(h') R2 R1)
        # diag_term_i = 1 + h'_i * (R2_ii * R1_ii)
        diag_term = 1.0 + diag_h_prime * d_r2 * d_r1

        log_jac = jnp.sum(jnp.log(diag_term))

        assert log_jac.shape == ()

        return log_jac

    @eqx.filter_jit
    def adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        f = jax.vmap(self.__adjust, 0)
        return f(draws)

    def __forward_and_adjust(self, draw: Float[Array, " n_dim"]) -> Tuple[Float[Array, " n_dim"], Scalar]:
        params = self.transform_params()

        Q: Float[Array, "dim rank"] = params["q"]
        R1: Float[Array, "rank rank"] = params["r1"]
        R2: Float[Array, "rank rank"] = params["r2"]
        b: Float[Array, " rank"] = params["b"]

        # --- Forward ---
        q_z = jnp.dot(Q.T, draw)
        arg = jnp.dot(R2, q_z) + b

        # h(arg)
        h_y = _h(arg)

        # Update
        term = jnp.dot(R1, h_y)
        draw_new = draw + jnp.dot(Q, term)

        # --- Adjust ---
        # Derivative h'(arg)
        diag_h_prime = _dh(arg)

        # Diagonals of triangular matrices
        d_r1 = jnp.diag(R1)
        d_r2 = jnp.diag(R2)

        # Log-det of triangular matrix
        diag_term = 1.0 + diag_h_prime * d_r2 * d_r1
        log_jac = jnp.sum(jnp.log(jnp.abs(diag_term)))

        assert len(draw_new.shape) == 1
        assert log_jac.shape == ()

        return draw_new, log_jac

    @eqx.filter_jit
    def forward_and_adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Tuple[Float[Array, "n_draws n_dim"], Scalar]:
        f = jax.vmap(self.__forward_and_adjust, 0)
        return f(draws)


class Sylvester(FlowSpec):
    """
    A specification for the Sylvester flow.
    """
    rank: int

    def __init__(self, rank: int):
        self.rank = rank

    def construct(self, dim: int) -> SylvesterLayer:
        if self.rank > dim:
            raise ValueError(f"Rank {self.rank} cannot be greater than dimension {dim}.")

        return SylvesterLayer(dim, self.rank)
