from typing import Callable, Tuple

import equinox as eqx
import equinox.nn as enn
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from bayinx.core.flow import FlowLayer, FlowSpec


class AffineNeuralTransform(eqx.Module):
    """
    A neural network that learns outputs the parameters of an affine transformation.

    # Attributes
    - `nn`: The neural network.
    - `out_dim`: The
    """
    nn: enn.MLP
    trans_dim: int

    def __init__(self, cond_dim: int, trans_dim: int, hidden_size: int, depth: int, activation: Callable[[Array], Array], key: PRNGKeyArray):
        n_params = 2*trans_dim

        self.nn = enn.MLP(cond_dim, n_params, hidden_size, depth, activation = activation, key = key)
        self.trans_dim = trans_dim

    @eqx.filter_vmap(in_axes = (None, 0))
    def __call__(self, x: Array) -> dict[str, Array]:
        # Compute raw transformation parameters
        raw_params = self.nn(x)

        # Format into dictionary
        params = {
            'shift': raw_params[0:self.trans_dim],
            'log_scale': 5 * jnp.tanh(raw_params[self.trans_dim:] / 5)
        }

        return params


class CNeuralAffineLayer(FlowLayer):
    """
    A (coupling) neural affine flow layer, AKA real NVP.

    # Attributes
    - `params`: Stores the neural network that outputs the parameters of the transformation.
    - `constraints`: Empty dictionary.
    - `static`: Whether the flow layer is frozen (parameters are not subject to further optimization).
    - `cond_indices`: Indices for the "conditioning" elements.
    - `trans_indices`: Indices for the "transformed" elements.
    - `dim`: The dimension of the parameter space.
    """

    params: dict[str, AffineNeuralTransform]
    constraints: dict
    static: bool
    cond_indices: Array
    trans_indices: Array
    dim: int

    def __init__(self, dim: int, hidden_size: int, depth: int, activation: Callable[[Array], Array], flip: bool, key: PRNGKeyArray):
        """
        Initializes a coupling-neural affine flow.

        # Parameters
        - `dim`: The dimension of the parameter space.
        - `hidden_size`: The size of the hidden layers.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function used between layers.
        - `flip`: Whether to flip the mask.
        - `key`: The PRNG key.
        """
        self.static = False
        self.dim = dim

        # Partition parameter space
        if flip:
            mask = (jnp.arange(dim) % 2 != 0)
        else:
            mask = (jnp.arange(dim) % 2 == 0)

        # Extract indices
        self.cond_indices = jnp.where(mask)[0]
        self.trans_indices = jnp.where(~mask)[0]

        # Get input and output dimensions for the neural network transform
        cond_dim: int = int(jnp.sum(mask))
        trans_dim: int = dim - cond_dim

        # Initialize parameters
        self.params = {
            "transform": AffineNeuralTransform(cond_dim, trans_dim, hidden_size, depth, activation = activation, key = key)
        }
        self.constraints = {}


    @eqx.filter_jit
    def forward(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        # Extract neural transform
        transform = self.params["transform"]

        # Subset conditioning and transformed elements
        cond_draws: Float[Array, "n_draws n_conds"] = draws[:, self.cond_indices]
        trans_draws: Float[Array, "n_draws n_trans"] = draws[:, self.trans_indices]

        # Get the learned transformation parameters
        params: dict[str, Array] = transform(cond_draws)

        # Extract relevant parameters
        shift: Float[Array, "n_draws n_trans"] = params['shift']
        log_scale: Float[Array, "n_draws n_trans"] = params['log_scale']

        # Compute forward transformation
        trans_draws = jnp.exp(log_scale) * trans_draws + shift
        draws = draws.at[:, self.trans_indices].set(trans_draws)

        return draws

    @eqx.filter_jit
    def reverse(self, draws: Float[Array, "n_draws n_dim"]) -> Float[Array, "n_draws n_dim"]:
        # Extract neural transform
        transform = self.params["transform"]

        # Subset conditioning and transformed elements from model parameters
        cond_draws: Float[Array, "n_draws n_conds"] = draws[:, self.cond_indices]
        trans_draws: Float[Array, "n_draws n_trans"] = draws[:, self.trans_indices]

        # Get the learned transformation parameters
        params: dict[str, Array] = transform(cond_draws)

        # Extract relevant parameters
        shift: Float[Array, "n_draws n_trans"] = params['shift']
        log_scale: Float[Array, "n_draws n_trans"] = params['log_scale']

        # Compute reverse transformation
        trans_draws = (trans_draws - shift) / jnp.exp(log_scale)
        draws = draws.at[:, self.trans_indices].set(trans_draws)

        return draws

    @eqx.filter_jit
    def forward_and_adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Tuple[Float[Array, "n_draws n_dim"], Float[Array, " n_draws"]]:
        # Extract neural transform
        transform = self.params["transform"]

        # Subset conditioning and transformed elements
        cond_draws: Float[Array, "n_draws n_conds"] = draws[:, self.cond_indices]
        trans_draws: Float[Array, "n_draws n_trans"] = draws[:, self.trans_indices]

        # Get the learned transformation parameters
        params: dict[str, Array] = transform(cond_draws)

        # Extract relevant parameters
        shift: Float[Array, "n_draws n_trans"] = params['shift']
        log_scale: Float[Array, "n_draws n_trans"] = params['log_scale']

        # Compute log-Jacobian adjustments
        log_jacs = jnp.sum(log_scale, axis = 1)

        # Compute forward transformation
        trans_draws = jnp.exp(log_scale) * trans_draws + shift
        draws = draws.at[:, self.trans_indices].set(trans_draws)

        return draws, log_jacs

    @eqx.filter_jit
    def reverse_and_adjust(self, draws: Float[Array, "n_draws n_dim"]) -> Tuple[Float[Array, "n_draws n_dim"], Float[Array, " n_draws"]]:
        # Extract neural transform
        transform = self.params["transform"]

        # Subset conditioning and transformed elements from model parameters
        cond_draws: Float[Array, "n_draws n_conds"] = draws[:, self.cond_indices]
        trans_draws: Float[Array, "n_draws n_trans"] = draws[:, self.trans_indices]

        # Get the learned transformation parameters
        params: dict[str, Array] = transform(cond_draws)

        # Extract relevant parameters
        shift: Float[Array, "n_draws n_trans"] = params['shift']
        log_scale: Float[Array, "n_draws n_trans"] = params['log_scale']

        # Compute log-Jacobian adjustments
        log_jacs = -jnp.sum(log_scale, axis = 1)

        # Compute reverse transformation
        trans_draws = (trans_draws - shift) / jnp.exp(log_scale)
        draws = draws.at[:, self.trans_indices].set(trans_draws)

        return draws, log_jacs


class CNeuralAffine(FlowSpec):
    """
    A specification for the coupling-neural affine flow, also known as the real-valued non-volume-preserving (real NVP) flow.

    Definition:
        Given an input $\\mathbf{z}$ with $D$ elements, we first partition them into $\\mathbf{z}_{1:d}$ and $\\mathbf{z}_{d+1:D}$.
        Then the transformation is given as:

        $$\\begin{aligned}
        T(\\mathbf{z}_{1:d}) &= \\mathbf{z}_{1:d} \\\\
        T(\\mathbf{z}_{d+1:D}) &= \\mathbf{s}(\\mathbf{z}_{1:d}) \\odot \\mathbf{z}_{d+1:D} + \\mathbf{c}(\\mathbf{z}_{1:d})
        \\end{aligned}$$

        Where $\\mathbf{z} \\in \\mathbb{R}^D$, $\\mathbf{s} \\in \\mathbb{R}^{D/2}$ is non-negative, and $\\mathbf{c} \\in \\mathbb{R}^{D/2}$.
        Both $\\mathbf{s}$ and $\\mathbf{c}$ are joint outputs of a neural network.

    Attributes:
        hidden_size: The width of the hidden layers for the neural network.
        depth: The number of hidden layers for the neural network.
        activation: The activation function to be used in between layers for the neural network.
        flip: Whether to flip the mask (which elements are the 'conditioner's and which are the 'transformer's).
        key: The PRNG key used to generate the flow layer.
    """
    hidden_size: int
    depth: int
    activation: Callable[[Array], Array]
    flip: bool
    key: PRNGKeyArray

    def __init__(self, hidden_size: int = 16, depth: int = 2, activation: Callable[[Array], Array] = jnn.relu, flip: bool = False, key: PRNGKeyArray = jr.key(0)):
        self.hidden_size = hidden_size
        self.depth = depth
        self.activation = activation
        self.flip = flip
        self.key = key

    def construct(self, dim: int) -> CNeuralAffineLayer:
        return CNeuralAffineLayer(dim, self.hidden_size, self.depth, self.activation, self.flip, self.key)
