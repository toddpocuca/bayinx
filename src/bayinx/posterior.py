from functools import partial
from typing import Any, Callable, Literal, Optional, Sequence, Type

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar
from scipy.stats import genpareto

from bayinx.core.flow import FlowSpec
from bayinx.core.model import Model
from bayinx.vi.normalizing_flow import NormalizingFlow
from bayinx.vi.stdstudentst import StandardStudentsT

# Public
__all__ = ["Posterior"]

class Posterior[M: Model]():
    """
    The posterior distribution for a model.

    Attributes:
        vari: The variational approximation of the posterior.
        config: The configuration for the posterior.
    """
    vari: NormalizingFlow[M]

    def __init__(self, model_def: Type[M], **kwargs: Any):
        """
        Initialize the posterior distribution.

        Arguments:
            model_def: The model class.
            kwargs: Additional shapes, data, and/or toy parameter objects to pass through for model construction.
        """
        # Construct toy model
        model = model_def(**kwargs)

        # Construct standard normal base distribution
        base = StandardStudentsT(model)

        # Construct default normalizing flow
        self.vari = NormalizingFlow( # type: ignore
            base = base,
            flows = [],
            model = model
        )

    def configure(
        self,
        flowspecs: Sequence[FlowSpec],
        insert: Literal['prepend', 'append'] = 'append'
    ):
        """
        Configure the variational approximation.

        Parameters:
            flowspecs: The specification for a sequence of flows.
            insert: Whether to insert the collection of flows before
        """
        # Append new NF architecture
        if flowspecs is not None:
            # Initialize NF architecture
            flows = [
                flowspec.construct(self.vari.dim) for flowspec in flowspecs
            ]

            # Freeze current flows
            for flow in self.vari.flows:
                object.__setattr__(flow, 'static', True) # kind of illegal but I need to avoid copies

            # Append new flows
            if insert == 'append':
                self.vari.flows.extend(flows)
            elif insert == 'prepend':
                object.__setattr__(self.vari, 'flows', flows + self.vari.flows) # i know i know...


    def fit(
        self,
        max_iters: int = 100_000,
        learning_rate: None | float = None,
        tolerance: None | float = None,
        grad_draws: int = 1,
        batch_size: int = 1,
        stl: bool = True,
        key: PRNGKeyArray = jr.key(0),
        verbose: bool = True,
        print_rate: int = 5000
    ):
        """
        Optimize the variational approximation.

        Parameters:
            max_iters: The maximum number of iterations for optimization.
            learning_rate: The initial learning rate for the optimizer.
            tolerance: The tolerance for the ELBO used for early stopping.
            grad_draws: The number of draws used to compute the ELBO gradient.
            batch_size: The maximum number of draws ever in memory used to compute the ELBO gradient.
            stl: Whether to use the Stick-the-Landing estimator.
            key: The PRNG key used during optimization.
            verbose: Whether to print a progress bar.
            print_rate: The number of iterations between updates for the progress bar.
        """
        # Include settings
        if learning_rate is None:
            learning_rate = 0.1 / self.vari.n_pars**0.5

        # Optimize variational approximation with user-specified flows
        self.vari = self.vari.fit(
            max_iters,
            learning_rate,
            tolerance,
            grad_draws,
            batch_size,
            stl,
            key,
            verbose,
            print_rate
        )

    def __reg_sample(
        self,
        func: Callable[[M, PRNGKeyArray], PyTree[Array]],
        n_draws: int,
        batch_size: int,
        key: PRNGKeyArray
    ) -> Array:
        vari = self.vari

        # Split keys
        per_batch_keys = jr.split(key, n_draws // batch_size)

        @partial(jax.vmap, in_axes = (0, 0))
        def reconstruct_and_query(draw: Array, key: PRNGKeyArray) -> PyTree[Array]:
            model = vari.reconstruct_model(draw).constrain()[0]

            # Evaluate callable
            obj = func(model, key)

            return obj

        # Sample in batches
        def batched_sample(per_batch_key: PRNGKeyArray) -> PyTree[Array]:
            # Sample draws
            draws = vari.sample(batch_size, key = per_batch_key)

            # Generate keys for each draw
            within_batch_keys = jr.split(per_batch_key, batch_size)

            return reconstruct_and_query(draws, within_batch_keys)

        # Generate samples of the posterior/posterior predictive
        post_draws: PyTree[Array] = lax.map(
            batched_sample,
            per_batch_keys
        )

        # Reshape to remove batch axis
        post_draws = jt.map(lambda x: x.reshape(-1, *x.shape[2:]), post_draws, is_leaf = lambda x: isinstance(x, Array))

        return post_draws

    def __sir_sample(
        self,
        func: Callable[[M, PRNGKeyArray], Array],
        n_draws: int,
        batch_size: int,
        key: PRNGKeyArray
    ) -> Array:
        vari = self.vari

        # Split key for sampling & resampling
        s_key, rs_key = jr.split(key)

        # Split keys across batches
        per_batch_keys = jr.split(s_key, n_draws // batch_size)

        @partial(jax.vmap, in_axes = (0, 0))
        def reconstruct_and_query(draw: Array, key: PRNGKeyArray) -> PyTree[Array]:
            model = vari.reconstruct_model(draw).constrain()[0]

            # Evaluate callable
            obj = func(model, key)

            return obj

        # Sample in batches
        def batched_sample(per_batch_key: PRNGKeyArray) -> tuple[PyTree[Array], Array]:
            # Sample draws from the base distribution
            base_draws = vari.base.sample(batch_size, key = per_batch_key)

            # Evaluate base density
            vari_evals = vari.base.eval(base_draws)

            # Apply forward transformations
            draws = base_draws
            for map in vari.flows:
                # Apply transformation
                draws, log_jacs = map.forward_and_adjust(draws)

                # Adjust variational density
                vari_evals = vari_evals + log_jacs

            # Evaluate posterior at variational draws
            post_evals = vari.eval_model(draws)

            # Compute unnormalized importance weight
            log_uweight = post_evals - vari_evals

            # Generate within-batch keys
            within_batch_keys = jr.split(per_batch_key, batch_size)

            return (reconstruct_and_query(draws, within_batch_keys), log_uweight)

        # Get posterior samples with importance weights
        post_draws, log_uweights = lax.map(
            batched_sample,
            per_batch_keys
        )

        # Reshape to remove batch axis
        post_draws = jt.map(lambda x: x.reshape(-1, *x.shape[2:]), post_draws, is_leaf = lambda x: isinstance(x, Array))
        log_uweights = log_uweights.reshape(-1, *log_uweights.shape[2:])

        # Re-sample draws using the Gumbel-max trick
        indices = jr.categorical(rs_key, log_uweights, shape=(n_draws,))
        post_draws = jt.map(lambda leaf: leaf[indices], post_draws)

        return post_draws


    def sample(
        self,
        attr: str,
        n_draws: int,
        batch_size: Optional[int] = 1,
        sir: bool = False,
        key: PRNGKeyArray = jr.key(0)
    ) -> PyTree[Array]:
        """
        Sample a variable from the posterior distribution.

        Parameters:
            attr: The name of the attribute of the model.
            n_draws: The number of draws to sample from the posterior.
            batch_size: The maximum number of full draws (essentially full instances of a model) ever in memory.
            sir: Whether to use sampling-importance-resampling.
            key: The PRNG key used to generate samples.
        """
        if batch_size is None or batch_size > n_draws:
            batch_size = n_draws
        else:
            batch_size = batch_size

        # Construct callable to extract node
        def func(model, key):
            return getattr(model, attr)

        if sir:
            # Do sampling-importance-resampling
            return self.__sir_sample(func, n_draws, batch_size, key)
        else:
            # Do regular sampling
            return self.__reg_sample(func, n_draws, batch_size, key)


    def predictive(
        self,
        func: Callable[[M, PRNGKeyArray], PyTree[Array]],
        n_draws: int,
        batch_size: None | int = None,
        sir: bool = False,
        key: PRNGKeyArray = jr.key(0)
    ) -> PyTree[Array]:
        """
        Generate predictives from the posterior distribution.

        Parameters:
            func: A function that maps the model and a PRNG key to an output.
            n_draws: The number of draws to sample from the posterior.
            batch_size: The maximum number of full draws (essentially full instances of a model) ever in memory.
            sir: Whether to use sampling-importance-resampling.
            key: The PRNG key used to generate samples.
        """
        if batch_size is None or batch_size > n_draws:
            batch_size = n_draws

        if sir:
            # Do sampling-importance-resampling
            return self.__sir_sample(func, n_draws, batch_size, key)
        else:
            # Do regular sampling
            return self.__reg_sample(func, n_draws, batch_size, key)


    def prop_ess(
        self,
        n_draws: int = 10_000,
        batch_size: None | int = 1,
        key: PRNGKeyArray = jr.key(0)
    ) -> Scalar:
        """
        Compute the proportional Effective Sample Size (ESS) for the variational approximation.

        The proportional ESS is the ratio of the ESS to the total number of draws ($ESS / S$).
        It measures the efficiency of the variational distribution as an importance sampling proposal for the true posterior.

        - ESS $\\approx$ 1.0: The variational approximation is a close match to the posterior.
        - ESS $\\ll$ 1.0: The approximation does not reflect the posterior well.

        Parameters:
            n_draws: The number of draws $S$ used to estimate the importance weights.
            batch_size: The maximum number of full model instances processed in memory at once.
            key: The PRNG key used for sampling from the base distribution.

        Returns:
            A scalar representing the proportional ESS, bounded between $1/S$ and $1.0$.
        """
        # Extract approximation
        vari = self.vari

        # Clip batch size if needed
        if batch_size is None or batch_size > n_draws:
            batch_size = n_draws

        # Split keys across batches
        per_batch_keys = jr.split(key, n_draws // batch_size)

        def batched_sample_and_compute(per_batch_key: PRNGKeyArray) -> Array:
            # Sample draws from the base distribution
            base_draws = vari.base.sample(batch_size, key = per_batch_key)

            # Compute posterior and variational densities together
            post_evals, vari_evals = vari._eval(base_draws)

            # Compute unnormalized importance weights
            log_uweights = post_evals - vari_evals

            return log_uweights

        # Compute unnormalized importance weights
        log_uweights: Array = lax.map(
            batched_sample_and_compute,
            per_batch_keys
        ).flatten()

        # Normalize importance weights
        log_weights: Array = log_uweights - jax.nn.logsumexp(log_uweights)

        # Compute log-ESS
        log_ess = -jax.nn.logsumexp(2.0 * log_weights)

        return jnp.exp(log_ess - jnp.log(n_draws))

    def pareto_k(
        self,
        n_draws: int = 10_000,
        batch_size: None | int = 1,
        key: PRNGKeyArray = jr.key(0)
    ) -> float:
        """
        Compute the Pareto k diagnostic for variational inference.

        The Pareto-smoothed importance sampling (PSIS) diagnostic gives a goodness of fit
        measurement for joint distributions. The estimated continuous hat_k value
        identifies the discrepancy between the approximate and true distribution.

        - k < 0.5: Fast convergence rate; the variational approximation is close to the true density.
        - 0.5 <= k < 0.7: Useful finite sample convergence rates; the approximation is acceptable.
        - k >= 0.7: Convergence rate becomes impractically slow; the approximation is unreliable.

        Parameters:
            n_draws: The number of draws to sample from the base distribution.
            batch_size: The maximum number of full draws ever in memory.
            key: The PRNG key used to generate samples.

        Returns:
            The estimated shape parameter k of the generalized Pareto distribution.
        """
        # Extract approximation
        vari = self.vari

        # Clip batch size if needed
        if batch_size is None or batch_size > n_draws:
            batch_size = n_draws

        # Split keys across batches
        per_batch_keys = jr.split(key, n_draws // batch_size)

        def batched_sample_and_compute(per_batch_key: PRNGKeyArray) -> Array:
            # Sample draws from the base distribution
            base_draws = vari.base.sample(batch_size, key=per_batch_key)

            # Compute posterior and variational densities together
            post_evals, vari_evals = vari._eval(base_draws)

            # Compute unnormalized log importance weights
            log_uweights = post_evals - vari_evals

            return log_uweights

        # Compute unnormalized log importance weights
        log_uweights: Array = lax.map(
            batched_sample_and_compute,
            per_batch_keys
        ).flatten()

        # Shift log weights for numerical stability
        log_weights_shifted = log_uweights - jnp.max(log_uweights)
        r = jnp.exp(log_weights_shifted)

        # Sort the importance ratios
        r_sorted = jnp.sort(r)

        # M is empirically set as min(S/5, 3 * sqrt(S))
        M = int(min(n_draws / 5.0, 3.0 * n_draws**0.5))

        # Fit generalized Pareto distribution to the M largest r_s
        tail_r = r_sorted[-M:]

        # Convert to numpy for scipy's genpareto fit
        tail_r_np = np.asarray(tail_r)

        # Shift the tail by the threshold
        threshold = tail_r_np[0]
        tail_shifted = tail_r_np - threshold

        # Report the shape parameter k
        k, loc, scale = genpareto.fit(tail_shifted, floc=0)

        return float(k)
