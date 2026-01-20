from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jssp
from jaxtyping import Array, PRNGKeyArray

import bayinx.ops as byo
from bayinx.core.flow import FlowSpec
from bayinx.core.model import Model
from bayinx.core.node import Node
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
    config: Dict[str, Any]

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
        self.vari = NormalizingFlow(
            base = base,
            flows = [],
            model = model
        )

        # Include default attributes
        self.config = {
            "learning_rate": 0.1 / self.vari.dim**0.5,
            "tolerance": 1e-4,
            "grad_draws": 4,
            "max_batch_size": 4
        }


    def configure(
        self,
        flowspecs: Optional[List[FlowSpec]] = None,
        learning_rate: Optional[float] = None,
        tolerance: Optional[float] = None,
        grad_draws: Optional[int] = None,
        max_batch_size: Optional[int] = None
    ):
        """
        Configure the variational approximation.

        Parameters:
            flowspecs: The specification for a sequence of flows.
            learning_rate: The initial learning rate for the optimizer.
            tolerance: The tolerance for the ELBO used for early stopping.
            grad_draws: The number of draws used to compute the ELBO gradient.
            max_batch_size: The maximum number of draws ever in memory used to compute the ELBO gradient.
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
            self.vari.flows.extend(flows)

        # Include other settings
        if learning_rate is not None:
            self.config["learning_rate"] = learning_rate
        if tolerance is not None:
            self.config["tolerance"] = tolerance
        if grad_draws is not None:
            self.config["grad_draws"] = grad_draws
        if max_batch_size is not None:
            self.config["max_batch_size"] = max_batch_size


    def fit(
        self,
        max_iters: int = 100_000,
        learning_rate: Optional[float] = None,
        tolerance: Optional[float] = None,
        grad_draws: Optional[int] = None,
        max_batch_size: Optional[int] = None,
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
            max_batch_size: The maximum number of draws ever in memory used to compute the ELBO gradient.
            verbose: Whether to print a progress bar.
            print_rate: The number of iterations between updates for the progress bar.
        """
        # Include settings
        if learning_rate is not None:
            self.config["learning_rate"] = learning_rate
        if tolerance is not None:
            self.config["tolerance"] = tolerance
        if grad_draws is not None:
            self.config["grad_draws"] = grad_draws
        if max_batch_size is not None:
            self.config["max_batch_size"] = max_batch_size

        # Optimize variational approximation with user-specified flows
        self.vari = self.vari.fit(
            max_iters,
            self.config["learning_rate"],
            self.config["tolerance"],
            self.config["grad_draws"],
            self.config["max_batch_size"],
            key,
            verbose,
            print_rate
        )

    def __reg_sample(
        self,
        func: Callable[[M, PRNGKeyArray], Node[Array] | Array],
        n_draws: int,
        batch_size: int,
        key: PRNGKeyArray
    ) -> Array:
        vari = self.vari

        # Split keys
        per_batch_keys = jr.split(key, n_draws // batch_size)

        @partial(jax.vmap, in_axes = (0, 0))
        def reconstruct_and_query(draw: Array, key: PRNGKeyArray) -> Array:
            model = vari.reconstruct_model(draw).constrain()[0]

            # Evaluate callable
            obj = func(model, key)

            # Coerce from Node if needed
            if isinstance(obj, Node):
                obj = byo.obj(obj)

                if isinstance(obj, Array):
                    return obj
                else:
                    raise TypeError("Return type of 'sample' & 'predictive' must be 'Node[Array]' or 'Array'.")
            elif isinstance(obj, Array):
                return obj
            else:
                raise TypeError("Return type of 'sample' & 'predictive' must be 'Node[Array]' or 'Array'.")

        # Sample in batches
        def batched_sample(per_batch_key: PRNGKeyArray) -> Array:
            # Sample draws
            draws = vari.sample(batch_size, key = per_batch_key)

            # Generate keys for each draw
            within_batch_keys = jr.split(per_batch_key, batch_size)

            return reconstruct_and_query(draws, within_batch_keys)

        # Generate samples of the posterior/posterior predictive
        post_draws: Array = lax.map(
            batched_sample,
            per_batch_keys
        )

        # Reshape to remove batch axis
        post_draws = post_draws.reshape(-1, *post_draws.shape[2:])

        return post_draws


    def __sir_sample(
        self,
        func: Callable[[M, PRNGKeyArray], Node[Array] | Array],
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
        def reconstruct_and_query(draw: Array, within_batch_key: PRNGKeyArray) -> Array:
            """
            Reconstruct model and evaluate query (either extract a node for 'sample' or compute a posterior predictive for 'predictive').
            """
            # Reconstruct model from variational draw
            model = self.vari.reconstruct_model(draw).constrain()[0]

            # Evaluate callable
            obj = func(model, within_batch_key)

            # Coerce from Node if needed
            if isinstance(obj, Node):
                obj = byo.obj(obj)

                if isinstance(obj, Array):
                    return obj
                else:
                    raise TypeError("Return type of 'node' argument for '.sample' & 'func' argument for '.predictive' must be 'Node[Array]' or 'Array'.")
            elif isinstance(obj, Array):
                return obj
            else:
                raise TypeError("Return type of 'sample' & 'predictive' must be 'Node[Array]' or 'Array'.")

        # Sample in batches
        def batched_sample(per_batch_key: PRNGKeyArray) -> tuple[Array, Array]:
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
        post_draws = post_draws.reshape(-1, *post_draws.shape[2:])
        log_uweights = log_uweights.reshape(-1, *log_uweights.shape[2:])

        # Normalize and exponentiate to get importance weights
        weights = jnp.exp(log_uweights - jssp.logsumexp(log_uweights))

        # Re-sample draws
        post_draws = jr.choice(
            rs_key,
            post_draws,
            shape = (n_draws, ),
            p = weights,
            axis = 0
        )

        return post_draws


    def sample(
        self,
        node: str,
        n_draws: int,
        max_batch_size: Optional[int] = 1,
        sir: bool = False,
        key: PRNGKeyArray = jr.key(0)
    ) -> Array:
        """
        Sample a node from the posterior distribution.

        Parameters:
            node: The name of the node.
            n_draws: The number of draws to sample from the posterior.
            max_batch_size: The maximum number of draws ever in memory.
            sir: Whether to use sampling-importance-resampling.
            key: The PRNG key used to generate samples.
        """
        if max_batch_size is None or max_batch_size > n_draws:
            batch_size = n_draws
        else:
            batch_size = max_batch_size

        # Construct callable to extract node
        def func(model, key):
            return byo.obj(getattr(model, node))

        if sir:
            # Do sampling-importance-resampling
            return self.__sir_sample(func, n_draws, batch_size, key)
        else:
            # Do regular sampling
            return self.__reg_sample(func, n_draws, batch_size, key)


    def predictive(
        self,
        func: Callable[[M, PRNGKeyArray], Node[Array] | Array],
        n_draws: int,
        max_batch_size: Optional[int] = None,
        sir: bool = True,
        key: PRNGKeyArray = jr.key(0)
    ) -> Array:
        """
        Generate predictives from the posterior distribution.

        Parameters:
            func: A function that maps the model and a PRNG key to a predictive.
            n_draws: The number of draws to sample from the posterior.
            max_batch_size: The maximum number of draws ever in memory.
            sir: Whether to use sampling-importance-resampling.
            key: The PRNG key used to generate samples.
        """
        if max_batch_size is None or max_batch_size > n_draws:
            batch_size = n_draws
        else:
            batch_size = max_batch_size

        if sir:
            # Do sampling-importance-resampling
            return self.__sir_sample(func, n_draws, batch_size, key)
        else:
            # Do regular sampling
            return self.__reg_sample(func, n_draws, batch_size, key)
