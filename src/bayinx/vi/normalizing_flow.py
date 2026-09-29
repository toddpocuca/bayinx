from typing import Callable, Self

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer, FlowSpec
from bayinx.core.model import Model
from bayinx.core.variational import Variational


class NormalizingFlow[M: Model, V: Variational](Variational[M]):
    """
    An ordered collection of diffeomorphisms that map a base distribution to a variational approximation.

    Attributes:
        dim: The dimension of the parameter space.
        _unflatten: A function to transform draws from the variational distribution back to a `Model`.
        _static: The static component of a partitioned `Model` used to initialize the `Variational` object.
        base: A base variational distribution.
        flows: An ordered collection of continuously parameterized diffeomorphisms.
        static_base: Whether the base distribution is fixed during optimization.
    """
    flows: list[FlowLayer]
    base: V
    static_base: bool

    def __init__(
        self,
        base: V,
        flow_specs: list[FlowSpec],
        static_base: bool = False,
    ):
        """
        Constructs an unoptimized normalizing flow posterior approximation.

        # Parameters
        - `base`: The base variational distribution.
        - `flows`: A list of flows.
        """
        self.dim = base.dim
        self._static = base._static
        self._unflatten = base._unflatten
        self.base = base
        self.flows = [spec.construct(self.dim) for spec in flow_specs]
        self.static_base = static_base

    @property
    def n_pars(self) -> int:
        """
        Calculates the total number of variational parameters across all layers and the base.
        """
        # Count parameters in the flow layers
        total_flow_pars = 0
        for layer in self.flows:
            params = eqx.filter(layer, layer.filter_spec).params

            # Sum the number of elements of all leaves in the tree
            total_flow_pars += sum(
                x.size for x in jtu.tree_leaves(params) if isinstance(x, Array)
            )

        # Count parameters in base distribution if learnable
        base_pars = 0 if self.static_base else self.base.n_pars

        return total_flow_pars + base_pars

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
    def sample_draws(
        self,
        n: int,
        key: PRNGKeyArray = jr.PRNGKey(0)
    ) -> Array:
        # Sample from the base distribution
        draws: Array = self.base.sample_draws(n, key = key)

        # Apply forward transformations
        for map in self.flows:
            draws = map.forward(draws)

        assert len(draws.shape) == 2
        return draws


    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        """
        Evaluate the variational density at `draws`.

        # Parameters
        - `draws`: Draws of the variational distribution.

        # Returns
            The variational density at `draws`.
        """
        base_draws, total_log_jacs = self.reverse_and_eval(draws)

        # Evaluate base variational density
        base_evals = self.base.eval(base_draws)

        # Accumulate log-Jacobian adjustment: P_X (x) = P_Y (y) / |det d/dy[f^-1](y)| ==> P_Y (y) = P_X (x) * |det d/dy[f^-1](y)|
        variational_evals = base_evals + total_log_jacs

        return variational_evals

    @eqx.filter_jit
    def __eval(self, base_draws: Array, return_draws: bool = False) -> tuple[Array, Array]:
        """
        Evaluate the posterior and variational densities together with draws of the base distribution to avoid extra compute.

        # Parameters
        - `base_draws`: Draws from the base variational distribution.

        # Returns
        The posterior and variational densities as JAX Arrays.
        """
        # Evaluate base density
        variational_evals = self.base.eval(base_draws)

        # Apply forward-flow
        draws, total_log_jacs = self.forward_and_eval(base_draws)

        # Accumulate Jacobian adjustment: P_Y (y) = P_X (x) / |det d/dx[f](x)|
        variational_evals -= total_log_jacs

        # Evaluate posterior at the variational draws
        posterior_evals = self.eval_model(draws)

        return posterior_evals, variational_evals

    @eqx.filter_jit
    def elbo(self, n: int, batch_size: int, key: PRNGKeyArray = jr.PRNGKey(0)) -> Scalar:
        dyn, static = eqx.partition(self, self.filter_spec)

        # Define ELBO function
        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)

            # Split key
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(batch_key: PRNGKeyArray) -> Array:
                # Draw from variational distribution
                draws: Array = self.base.sample_draws(batch_size, key = batch_key)

                # Evaluate posterior and variational densities
                batched_post_evals, batched_vari_evals = self.__eval(draws)

                # Compute batched ELBO evals
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return batched_elbo_evals

            # Compute ELBO evals
            elbo_evals = lax.map(batched_elbo, keys)

            # Average ELBO estimates
            elbo_est = jnp.mean(elbo_evals)

            return elbo_est

        return elbo(dyn, n, key)

    @eqx.filter_jit
    def elbo_grad(self, n: int, batch_size: int, stl: bool, key: PRNGKeyArray) -> Self:
        dyn, static = eqx.partition(self, self.filter_spec)
        stopped_dyn = lax.stop_gradient(dyn)

        # Define ELBO function
        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)
            stopped_self = eqx.combine(stopped_dyn, static)

            # Split key
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(batch_key: PRNGKeyArray) -> Array:
                if stl:
                    # Draw from variational distribution
                    draws: Array = self.sample_draws(batch_size, key = batch_key)

                    # Evaluate posterior density
                    batched_post_evals = self.eval_model(draws)

                    # Evaluate variational density
                    batched_vari_evals = stopped_self.eval(draws) # STL estimator

                    # Compute batched ELBO evals
                    batched_elbo_evals: Array = batched_post_evals - batched_vari_evals
                else:
                    # Draw from base distribution
                    base_draws: Array = self.base.sample_draws(batch_size, key = batch_key)

                    # Evaluate posterior and variational densities together from base samples
                    batched_post_evals, batched_vari_evals = self.__eval(base_draws)

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
    def elbo_and_grad(self, n: int, batch_size: int, stl: bool, key: PRNGKeyArray) -> tuple[Scalar, Self]:
        dyn, static = eqx.partition(self, self.filter_spec)
        stopped_dyn = lax.stop_gradient(dyn)

        # Define ELBO function
        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)
            stopped_self = eqx.combine(stopped_dyn, static)

            # Split key
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(batch_key: PRNGKeyArray) -> Array:
                if stl:
                    # Draw from variational distribution
                    draws: Array = self.sample_draws(batch_size, key = batch_key)

                    # Evaluate posterior density
                    batched_post_evals = self.eval_model(draws)

                    # Evaluate variational density
                    batched_vari_evals = stopped_self.eval(draws) # STL estimator

                    # Compute batched ELBO evals
                    batched_elbo_evals: Array = batched_post_evals - batched_vari_evals
                else:
                    # Draw from base distribution
                    base_draws: Array = self.base.sample_draws(batch_size, key = batch_key)

                    # Evaluate posterior and variational densities together from base samples
                    batched_post_evals, batched_vari_evals = self.__eval(base_draws)

                    # Compute batched ELBO evals
                    batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return batched_elbo_evals

            # Compute ELBO evals
            elbo_evals = lax.map(batched_elbo, keys)

            # Average ELBO estimates
            elbo_est = jnp.mean(elbo_evals)

            return elbo_est

        # Map to its value & gradient
        elbo_and_grad: Callable[
            [Self, int, PRNGKeyArray], tuple[Scalar, Self]
        ] = eqx.filter_value_and_grad(elbo)

        return elbo_and_grad(dyn, n, key)

    def forward(self, base_draws: Array) -> Array:
        """
        Applies the forward flow at `base_draws`.

        Parameters:
            base_draws: Draws from the base distribution.

        Returns:
            The forward-flow transformed base draws (equivalent to draws from the variational distribution).
        """
        # Apply forward transformations
        draws = base_draws
        for map in self.flows:
            draws = map.forward(draws)

        return draws

    def forward_and_eval(self, base_draws: Array) -> tuple[Array, Array]:
        """
        Applies the forward flow and accumulates the log-Jacobian adjustment.

        Parameters:
            base_draws: Draws from the base distribution.

        Returns:
            The forward-flow transformed base draws and the associated log-Jacobian adjustment.
        """
        total_log_jacs: Array = jnp.zeros(base_draws.shape[0])

        draws = base_draws
        for map in self.flows:
            # Apply transformation
            draws, log_jacs = map.forward_and_adjust(draws)

            # Accumulate log-Jacobian adjustments
            total_log_jacs += log_jacs

        return draws, total_log_jacs

    def reverse(self, draws: Array) -> Array:
        """
        Applies the reverse flow at `draws`.

        Parameters:
            draws: Draws from the variational distribution.

        Returns:
            The reverse-flow transformed draws.
        """
        # Apply reverse transformations
        for map in reversed(self.flows):
            draws = map.forward(draws)

        return draws

    def reverse_and_eval(self, draws: Array) -> tuple[Array, Array]:
        """
        Applies the reverse flow and evaluates the variational density at `draws`.

        Parameters:
            draws: Draws from the variational distribution.

        Returns:
            The reverse-flow transformed draws (equivalent to draws from the base distribution).
        """
        total_log_jacs: Array = jnp.zeros(draws.shape[0])

        for map in reversed(self.flows):
            # Apply reverse transformation and accumulate the log-Jacobian adjustments
            draws, log_jacs = map.reverse_and_adjust(draws)

            # Accumulate log-Jacobian adjustments
            total_log_jacs += log_jacs

        return draws, total_log_jacs
