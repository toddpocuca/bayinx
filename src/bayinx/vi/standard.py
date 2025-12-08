from typing import Self, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar

from bayinx.core.variational import M, Variational
from bayinx.dists.normal.pars.mean_scale import _logprob


class Standard(Variational[M]):
    """
    A standard normal approximation to a posterior distribution.

    # Attributes
    - `dim`: The dimension of the support.
    """

    def __init__(self, model: M):
        """
        Constructs a standard normal approximation to a posterior distribution.

        # Parameters
        - `model`: A probabilistic `Model` object.
        """
        # Partition model
        params, self._static = eqx.partition(model, model.filter_spec)

        # Flatten params component
        params, self._unflatten = ravel_pytree(params)

        # Store dimension of parameter space
        self.dim = jnp.size(params)


    @eqx.filter_jit
    def sample(self, n: int, key: PRNGKeyArray = jr.PRNGKey(0)) -> Array:
        # Sample variational draws
        draws: Array = jr.normal(key=key, shape=(n, self.dim))

        # Shape checks
        assert len(draws.shape) == 2

        return draws

    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        return _logprob(
            x=draws,
            mean=0.0,
            scale=1.0,
        ).sum(axis=1)

    @property
    def filter_spec(self):
        filter_spec = jtu.tree_map(lambda _: False, self)

        return filter_spec

    @eqx.filter_jit
    def elbo(self, n: int, batch_size: int, key: PRNGKeyArray) -> Scalar:
        dyn, static = eqx.partition(self, self.filter_spec)

        @eqx.filter_jit
        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            vari = eqx.combine(dyn, static)

            # Sample draws from variational distribution
            draws: Array = vari.sample(n, key)

            # Evaluate posterior density for each draw
            posterior_evals: Array = vari.eval_model(draws)

            # Evaluate variational density for each draw
            variational_evals: Array = vari.eval(draws)

            # Evaluate ELBO
            return jnp.mean(posterior_evals - variational_evals)

        return elbo(dyn, n, key)

    @eqx.filter_jit
    def elbo_grad(self, n: int, batch_size: int, key: PRNGKeyArray) -> Self:
        raise RuntimeError("Do not use the 'elbo_grad' method for a Standard variational approximation. It has no variational parameters.")
        return self

    def elbo_and_grad(self, n: int, batch_size: int, key: PRNGKeyArray) -> Tuple[Scalar, PyTree]:
        """
        Evaluate the ELBO and its gradient.
        """
        raise RuntimeError("Do not use the 'elbo_and_grad' method for a Standard variational approximation. It has no variational parameters.")
        return self.elbo(n, key), self
