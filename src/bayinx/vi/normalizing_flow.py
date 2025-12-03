from typing import Callable, Optional, Self, Tuple

import equinox as eqx
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray, Scalar

from bayinx.core.flow import FlowLayer
from bayinx.core.variational import M, Variational


class NormalizingFlow(Variational[M]):
    """
    An ordered collection of diffeomorphisms that map a base distribution to a
    normalized approximation of a posterior distribution.

    # Attributes
    - `dim`: The dimension of the support.
    - `base`: A base variational distribution.
    - `flows`: An ordered collection of continuously parameterized diffeomorphisms.
    """
    flows: list[FlowLayer]
    base: Variational[M]

    def __init__(
        self,
        base: Variational[M],
        flows: list[FlowLayer],
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

        return filter_spec

    @eqx.filter_jit
    def sample(self, n: int, key: PRNGKeyArray = jr.PRNGKey(0)) -> Array:
        # Sample from the base distribution
        draws: Array = self.base.sample(n, key)

        assert len(draws.shape) == 2

        # Apply forward transformations
        for map in self.flows:
            draws = map.forward(draws)

        assert len(draws.shape) == 2

        return draws

    @eqx.filter_jit
    def eval(self, draws: Array) -> Array:
        raise RuntimeError("Evaluating the variational density for a normalizing flow requires an analytic inverse to exist, which many useful flows do not have. Therefore, do not use this method.")
        return jnp.full(draws.shape[0], jnp.nan)

    @eqx.filter_jit
    def __eval(self, draws: Array) -> Tuple[Array, Array]:
        """
        Evaluate the posterior and variational densities together at the
        transformed `draws` to avoid extra compute.

        # Parameters
        - `draws`: Draws from the base variational distribution.

        # Returns
        The posterior and variational densities as JAX Arrays.
        """
        # Evaluate base density
        variational_evals: Array = self.base.eval(draws)

        # Shape checks
        assert len(variational_evals.shape) == 1
        assert len(draws.shape) == 2

        for map in self.flows:
            # Apply transformation
            draws, ljas = map.forward_and_adjust(draws)
            assert len(draws.shape) == 2
            assert len(ljas.shape) == 1

            # Adjust variational density
            variational_evals = variational_evals - ljas

        # Evaluate posterior at final variational draws
        posterior_evals = self.eval_model(draws)

        # Shape checks
        assert len(posterior_evals.shape) == 1
        assert len(variational_evals.shape) == 1
        assert posterior_evals.shape == variational_evals.shape

        return posterior_evals, variational_evals

    @eqx.filter_jit
    def elbo(self, n: int, batch_size: int, key: PRNGKeyArray = jr.PRNGKey(0)) -> Scalar:
        dyn, static = eqx.partition(self, self.filter_spec)

        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)

            # Split keys
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(carry: None, key: PRNGKeyArray) -> Tuple[None, Array]:
                # Draw from variational distribution
                draws: Array = self.base.sample(batch_size, key)

                # Evaluate posterior and variational densities
                batched_post_evals, batched_vari_evals = self.__eval(draws)

                # Compute ELBO estimate
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return None, batched_elbo_evals

            elbo_evals = scan(
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
            def batched_elbo(carry: None, key: PRNGKeyArray) -> Tuple[None, Array]:
                # Draw from variational distribution
                draws: Array = self.base.sample(batch_size, key)

                # Evaluate posterior and variational densities
                batched_post_evals, batched_vari_evals = self.__eval(draws)

                # Compute ELBO estimate
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return None, batched_elbo_evals

            elbo_evals = scan(
                batched_elbo,
                init=None,
                xs=keys,
                length=n // batch_size
            )[1]

            # Compute average of ELBO estimates
            return jnp.mean(elbo_evals)

        # Map to its gradient
        elbo_grad: Callable[
            [Self, int, PRNGKeyArray], Self
        ] = eqx.filter_grad(elbo)

        return elbo_grad(dyn, n, key)

    @eqx.filter_jit
    def elbo_and_grad(self, n: int, batch_size: int, key: PRNGKeyArray) -> Tuple[Scalar, Self]:
        dyn, static = eqx.partition(self, self.filter_spec)

        def elbo(dyn: Self, n: int, key: PRNGKeyArray) -> Scalar:
            self = eqx.combine(dyn, static)

            # Split keys
            keys = jr.split(key, n // batch_size)

            # Split ELBO calculation into batches
            def batched_elbo(carry: None, key: PRNGKeyArray) -> Tuple[None, Array]:
                # Draw from variational distribution
                draws: Array = self.base.sample(batch_size, key)

                # Evaluate posterior and variational densities
                batched_post_evals, batched_vari_evals = self.__eval(draws)

                # Compute ELBO estimate
                batched_elbo_evals: Array = batched_post_evals - batched_vari_evals

                return None, batched_elbo_evals

            elbo_evals = scan(
                batched_elbo,
                init=None,
                xs=keys,
                length=n // batch_size
            )[1]

            # Compute average of ELBO estimates
            return jnp.mean(elbo_evals)

        # Map to its value & gradient
        elbo_and_grad: Callable[
            [Self, int, PRNGKeyArray], Tuple[Scalar, Self]
        ] = eqx.filter_value_and_grad(elbo)

        return elbo_and_grad(dyn, n, key)
