
from functools import partial
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type

import jax
import jax.random as jr
from jax.lax import scan
from jaxtyping import Array, PRNGKeyArray

from bayinx.core.flow import FlowSpec
from bayinx.core.node import Node
from bayinx.core.variational import M
from bayinx.vi.normalizing_flow import NormalizingFlow
from bayinx.vi.stdstudentst import StandardStudentsT

# Public
__all__ = ["Posterior"]

class Posterior(Generic[M]):
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

    def sample(
        self,
        node: str,
        n_draws: int,
        max_batch_size: Optional[int] = None,
        sir: bool = True,
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
        if max_batch_size is None:
            max_batch_size = n_draws

        # Split keys
        keys = jr.split(key, n_draws // max_batch_size)

        # Reconstruct model and grab node
        @partial(jax.vmap, in_axes = 0)
        def reconstruct_and_subset(draw: Array):
            model = self.vari.reconstruct_model(draw).constrain()[0]

            return getattr(model, node).obj

        # Sample in batches
        def batched_sample(carry: None, key: PRNGKeyArray):
            # Sample draws
            draws = self.vari.sample(max_batch_size, sir = sir, key = key)

            return None, reconstruct_and_subset(draws)

        # Get posterior samples
        posterior_draws = scan(
            batched_sample,
            init=None,
            xs=keys,
            length=n_draws // max_batch_size
        )[1].squeeze()

        return posterior_draws


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
        if max_batch_size is None:
            max_batch_size = n_draws

        # Split keys
        keys = jr.split(key, n_draws // max_batch_size)

        # Reconstruct a model and generate a sample of the posterior predictive
        @partial(jax.vmap, in_axes = (0, 0))
        def reconstruct_and_predict(draw: Array, key: PRNGKeyArray) -> Array:
            model = self.vari.reconstruct_model(draw).constrain()[0]

            # Compute predictive
            obj = func(model, key)

            # Coerce from Node if needed
            if isinstance(obj, Node):
                obj: Array = obj.obj # type: ignore

            return obj

        # Compute posterior predictive in batches
        def batched_sample(carry: None, key: PRNGKeyArray) -> Tuple[None, Array]:
            # Sample draws
            draws = self.vari.sample(max_batch_size, sir = sir, key = key)

            # Generate additional keys for each draw
            keys = jr.split(key, max_batch_size)

            return None, reconstruct_and_predict(draws, keys)

        # Generate samples of the posterior predictive
        predictive_draws: Array = scan(
            batched_sample,
            init=None,
            xs=keys,
            length=n_draws // max_batch_size
        )[1].squeeze()

        return predictive_draws
