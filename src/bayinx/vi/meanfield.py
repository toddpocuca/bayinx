from typing import Generic, Self

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PRNGKeyArray, Scalar

from bayinx.core.variational import M, Variational
from bayinx.dists.normal.pars.mean_scale import _logprob


class MeanField(Variational, Generic[M]):
    """
    A fully factorized Gaussian approximation to a posterior distribution.

    # Attributes
    - `dim`: The dimension of the support.
    - `mean`: The mean of the unconstrained approximation.
    - `log_std` The log-transformed standard deviation of the unconstrained approximation.
    """

    mean: Array
    log_std: Array

    def __init__(self, model: M):
        """
        Constructs an unoptimized meanfield posterior approximation.

        # Parameters
        - `model`: A probabilistic `Model` object.
        - `init_log_std`: The initial log-transformed standard deviation of the Gaussian approximation.
        """
        # Partition model
        params, self._static = eqx.partition(model, model.filter_spec)

        # Flatten params component
        params, self._unflatten = ravel_pytree(params)

        # Initialize variational parameters
        self.mean = params
        self.log_std = jnp.full(params.size, 0.0)

    @property
    def filter_spec(self):
        # Generate empty specification
        filter_spec = jtu.tree_map(lambda _: False, self)

        # Specify variational parameters
        filter_spec = eqx.tree_at(
            lambda mf: mf.mean,
            filter_spec,
            replace=True,
        )
        filter_spec = eqx.tree_at(
            lambda mf: mf.log_std,
            filter_spec,
            replace=True,
        )

        return filter_spec

    @eqx.filter_jit
    def sample(self, n: int, key: PRNGKeyArray = jr.PRNGKey(0)) -> Array:
        # Sample variational draws
        draws: Array = (
            jr.normal(key=key, shape=(n, self.mean.size)) * jnp.exp(self.log_std)
            + self.mean
        )

        return draws

    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        return _logprob(
            x=draws,
            mean=self.mean,
            scale=jnp.exp(self.log_std),
        ).sum(axis=1)

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
        dyn, static = eqx.partition(self, self.filter_spec)

        @eqx.filter_jit
        @eqx.filter_grad
        def elbo_grad(dyn: Self, n: int, key: PRNGKeyArray):
            vari = eqx.combine(dyn, static)

            # Sample draws from variational distribution
            draws: Array = vari.sample(n, key)

            # Evaluate posterior density for each draw
            posterior_evals: Array = vari.eval_model(draws)

            # Evaluate variational density for each draw
            variational_evals: Array = vari.eval(draws)

            # Evaluate ELBO
            return jnp.mean(posterior_evals - variational_evals)

        return elbo_grad(dyn, n, key)
