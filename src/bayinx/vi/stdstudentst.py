from typing import Callable, Self, Tuple

import equinox as eqx
import jax.flatten_util as jfu
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jssp
import jax.tree_util as jtu
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from bayinx.core.model import Model
from bayinx.core.variational import Variational
from bayinx.dists.studentst.pars.loc_scale_df import _logprob


class StandardStudentsT[M: Model](Variational[M]):
    """
    A standard Student's T approximation of a posterior distribution with learnable degrees of freedom.
    """
    log_df: Float[Array, " n_dims"]


    @property
    def df(self) -> Float[Array, " n_dims"]:
        # Constrain degrees of freedom
        return jnp.exp(self.log_df) + 1.0

    def __init__(self, model: M, initial_df: float = 10.0):
        """
        Constructs a standard Student's T approximation to a posterior distribution.

        # Parameters
        - `model`: A probabilistic `Model` object.
        - `initial_df`: The initial degrees of freedom.
        """
        # Partition model
        params, self._static = eqx.partition(model, model.filter_spec)

        # Flatten params component
        params, self._unflatten = jfu.ravel_pytree(params)

        # Store dimension of parameter space
        self.dim = jnp.size(params)

        # Initialize degrees of freedom
        self.log_df = jnp.full(self.dim, jnp.log(initial_df))

    @eqx.filter_jit
    def sample(self, n: int, sir: bool = True, key: PRNGKeyArray = jr.PRNGKey(0)) -> Array:
        # Split key
        k1, k2, k3 = jr.split(key, 3)

        # Generate standard normal noise
        normal_draws = jr.normal(k1, shape=(n, self.dim))

        # Generate Gamma noise through a differentiable sampler
        shape = self.df / 2.0
        rate = self.df / 2.0
        gamma_draws = jr.gamma(k2, shape, shape=(n, self.dim)) / rate

        # Compute variational draws
        draws = normal_draws * 1.0 / jnp.sqrt(gamma_draws)

        # Apply sampling-importance-resampling if requested
        if sir:
            # Evaluate variational density
            variational_evals = self.eval(draws)

            # Evaluate posterior density
            posterior_evals = self.eval_model(draws)

            # Compute importance weights
            log_weights = posterior_evals - variational_evals
            weights = jnp.exp(log_weights - jssp.logsumexp(log_weights))

            # Re-sample draws
            draws = jr.choice(k3, draws, shape = (n, ), p = weights, axis = 0)

        assert len(draws.shape) == 2
        return draws

    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        return _logprob(draws, self.df, 0.0, 1.0).sum(axis=1)

    @property
    def filter_spec(self) -> Self:
        # Generate empty specification
        filter_spec: Self = jtu.tree_map(lambda _: False, self)

        # Specify degrees of freedom
        filter_spec = eqx.tree_at(
            lambda vari: vari.log_df,
            filter_spec,
            True
        )

        return filter_spec

    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        return _logprob(
            draws,
            self.df,
            0.0,
            1.0,
        ).sum(axis=1)

    @eqx.filter_jit
    def elbo(self, n: int, batch_size: int, key: PRNGKeyArray) -> Scalar:
        dyn, static = eqx.partition(self, self.filter_spec)

        # Define ELBO function
        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)

            # Split keys
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(_, batch_key):
                # Draw from variational distribution
                draws = self.sample(batch_size, batch_key)

                # Evaluate posterior and variational densities
                batched_vari_evals = self.eval(draws)
                batched_post_evals = self.eval_model(draws)

                # Compute ELBO estimate
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return None, batched_elbo_evals

            elbo_evals = lax.scan(
                batched_elbo,
                init=None,
                xs=keys,
                length=n // batch_size
            )[1]

            # Compute ELBO estimate
            return jnp.mean(elbo_evals)

        # Compute ELBO estimate
        return elbo(dyn, n, key)

    @eqx.filter_jit
    def elbo_grad(self, n: int, batch_size: int, key: PRNGKeyArray) -> Self:
        dyn, static = eqx.partition(self, self.filter_spec)

        # Define ELBO function
        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)

            # Split keys
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(_, batch_key):
                # Draw from variational distribution
                draws = self.sample(batch_size, batch_key)

                # Evaluate posterior and variational densities
                batched_vari_evals = self.eval(draws)
                batched_post_evals = self.eval_model(draws)

                # Compute ELBO estimate
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return None, batched_elbo_evals

            elbo_evals = lax.scan(
                batched_elbo,
                init=None,
                xs=keys,
                length=n // batch_size
            )[1]

            # Compute ELBO estimate
            return jnp.mean(elbo_evals)

        # Map to its gradient
        elbo_grad: Callable[
            [Self, int, PRNGKeyArray], Self
        ] = eqx.filter_grad(elbo)

        # Compute ELBO gradient estimate
        return elbo_grad(dyn, n, key)

    @eqx.filter_jit
    def elbo_and_grad(self, n: int, batch_size: int, key: PRNGKeyArray) -> Tuple[Scalar, Self]:
        dyn, static = eqx.partition(self, self.filter_spec)

        # Define ELBO function
        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)

            # Split keys
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(_, batch_key):
                # Draw from variational distribution
                draws = self.sample(batch_size, batch_key)

                # Evaluate posterior and variational densities
                batched_vari_evals = self.eval(draws)
                batched_post_evals = self.eval_model(draws)

                # Compute ELBO estimate
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return None, batched_elbo_evals

            elbo_evals = lax.scan(
                batched_elbo,
                init=None,
                xs=keys,
                length=n // batch_size
            )[1]

            # Compute ELBO estimate
            return jnp.mean(elbo_evals)

        # Map to its value & gradient
        elbo_and_grad: Callable[
            [Self, int, PRNGKeyArray], Tuple[Scalar, Self]
        ] = eqx.filter_value_and_grad(elbo)

        return elbo_and_grad(dyn, n, key)
