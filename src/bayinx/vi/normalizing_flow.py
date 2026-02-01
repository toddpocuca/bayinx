from typing import Callable, Optional, Self

import equinox as eqx
import jax.flatten_util as jfu
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer
from bayinx.core.model import Model
from bayinx.core.variational import Variational


class NormalizingFlow[M: Model](Variational[M]):
    """
    An ordered collection of diffeomorphisms that map a base distribution to a variational approximation.

    Attributes:
        dim: The dimension of the support.
        base: A base variational distribution.
        flows: An ordered collection of continuously parameterized diffeomorphisms.
        static_base: Whether the base distribution is fixed during optimization.
    """
    flows: list[FlowLayer]
    base: Variational[M]
    static_base: bool

    def __init__(
        self,
        base: Variational[M],
        flows: list[FlowLayer],
        static_base: bool = False,
        model: Optional[M] = None,
        _static: Optional[M] = None,
        _unflatten: Optional[Callable[[Array], M]] = None
    ):
        """
        Constructs an unoptimized normalizing flow posterior approximation.

        # Parameters
        - `base`: The base variational distribution.
        - `flows`: A list of flows.
        - `model`: A probabilistic `Model` object.
        """
        if model is not None:
            # Partition model
            params, self._static = eqx.partition(model, model.filter_spec)

            # Flatten params component
            _, self._unflatten = jfu.ravel_pytree(params)
        elif _static is not None and _unflatten is not None:
            self._static = _static
            self._unflatten = _unflatten
        else:
            raise ValueError("Either 'model' or '_static' and '_unflatten' must be specified.")

        self.dim = base.dim
        self.base = base
        self.flows = flows
        self.static_base = static_base

    @property
    def filter_spec(self) -> Self:
        # Generate empty specification
        filter_spec: Self = jtu.tree_map(lambda _: False, self)

        # Specify variational parameters based on each flow's filter spec.
        filter_spec: Self = eqx.tree_at(
            lambda vari: vari.flows,
            filter_spec,
            replace=[flow.filter_spec for flow in self.flows],
        )

        # Specify parameters for the base distribution if dynamic
        if self.static_base is False:
            filter_spec: Self = eqx.tree_at(
                lambda vari: vari.base,
                filter_spec,
                replace=self.base.filter_spec,
            )

        return filter_spec

    @eqx.filter_jit
    def sample(
        self,
        n: int,
        key: PRNGKeyArray = jr.PRNGKey(0)
    ) -> Array:
        # Sample from the base distribution
        draws: Array = self.base.sample(n, key = key)

        # Apply forward transformations
        for map in self.flows:
            draws = map.forward(draws)

        assert len(draws.shape) == 2
        return draws


    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        raise RuntimeError("Computing the variational density efficiently requires access to an analytic reverse flow, which many useful flows do not have.")
        return jnp.full(draws.shape[0], jnp.nan) # dont use this method

    @eqx.filter_jit
    def _eval(self, base_draws: Array, return_draws: bool = False) -> tuple[Array, Array] | tuple[Array, Array, Array]:
        """
        Evaluate the posterior and variational densities together with draws of the base distribution to avoid extra compute.

        # Parameters
        - `base_draws`: Draws from the base variational distribution.

        # Returns
        The posterior and variational densities as JAX Arrays (accompanied by the variational draws if requested).
        """
        # Evaluate base density
        variational_evals: Array = self.base.eval(base_draws)

        draws = base_draws
        for map in self.flows:
            # Apply transformation
            draws, log_jacs = map.forward_and_adjust(draws)

            # Adjust variational density
            variational_evals = variational_evals + log_jacs

        # Evaluate posterior at the variational draws
        posterior_evals = self.eval_model(draws)

        if return_draws:
            return posterior_evals, variational_evals, draws
        else:
            return posterior_evals, variational_evals

    @eqx.filter_jit
    def elbo(self, n: int, batch_size: int, key: PRNGKeyArray = jr.PRNGKey(0)) -> Scalar:
        dyn, static = eqx.partition(self, self.filter_spec)

        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)

            # Split keys
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(carry: None, batch_key: PRNGKeyArray) -> tuple[None, Array]:
                # Draw from (base) variational distribution
                draws: Array = self.base.sample(batch_size, key = batch_key)

                # Evaluate posterior and variational densities
                batched_post_evals, batched_vari_evals = self._eval(draws)

                # Compute ELBO estimate
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return None, batched_elbo_evals

            elbo_evals = lax.scan(
                batched_elbo,
                init=None,
                xs=keys,
                length=n // batch_size
            )[1]

            # Compute average of ELBO estimates
            return jnp.mean(elbo_evals)

        return elbo(dyn, n, key)

    @eqx.filter_jit
    def elbo_grad(self, n: int, batch_size: int, key: PRNGKeyArray) -> Self:
        dyn, static = eqx.partition(self, self.filter_spec)

        # Define ELBO function
        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)

            # Split key
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(batch_key: PRNGKeyArray) -> Array:
                # Draw from variational distribution
                draws: Array = self.base.sample(batch_size, key = batch_key)

                # Evaluate posterior and variational densities
                batched_post_evals, batched_vari_evals = self._eval(draws)

                # Compute batched ELBO evals
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return batched_elbo_evals

            # Compute ELBO evals
            elbo_evals = lax.map(batched_elbo, keys)

            # Average ELBO estimates
            elbo_est = jnp.mean(elbo_evals)

            return elbo_est

        # Map to its gradient
        elbo_grad: Callable[
            [Self, int, PRNGKeyArray], Self
        ] = eqx.filter_grad(elbo)

        return elbo_grad(dyn, n, key)

    @eqx.filter_jit
    def elbo_and_grad(self, n: int, batch_size: int, key: PRNGKeyArray) -> tuple[Scalar, Self]:
        dyn, static = eqx.partition(self, self.filter_spec)

        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)

            # Split keys
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(carry: None, batch_key: PRNGKeyArray) -> tuple[None, Array]:
                # Draw from variational distribution
                draws: Array = self.base.sample(batch_size, key = batch_key)

                # Evaluate posterior and variational densities
                batched_post_evals, batched_vari_evals = self._eval(draws)

                # Compute ELBO estimate
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return None, batched_elbo_evals

            # Compute ELBO evals
            elbo_evals = lax.scan(
                batched_elbo,
                init=None,
                xs=keys,
                length=n // batch_size
            )[1].flatten()

            # Compute average of ELBO estimates
            return jnp.mean(elbo_evals)

        # Map to its value & gradient
        elbo_and_grad: Callable[
            [Self, int, PRNGKeyArray], tuple[Scalar, Self]
        ] = eqx.filter_value_and_grad(elbo)

        return elbo_and_grad(dyn, n, key)
