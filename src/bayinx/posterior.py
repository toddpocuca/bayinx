
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
from bayinx.vi.standard import Standard


class Posterior(Generic[M]):
    """
    The posterior distribution for a model.

    # Attributes
    - `vari`: The variational approximation of the posterior.
    - `config` The configuration for the posterior.
    """
    vari: NormalizingFlow[M]
    config: Dict[str, Any]

    def __init__(self, model_def: Type[M], **kwargs: Any):
        # Construct toy model
        model = model_def(**kwargs)

        # Construct standard normal base distribution
        base = Standard(model)

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
            "grad_draws": 1,
            "batch_size": 1
        }


    def configure(
        self,
        flowspecs: Optional[List[FlowSpec]] = None,
        learning_rate: Optional[float] = None,
        tolerance: Optional[float] = None,
        grad_draws: Optional[int] = None,
        batch_size: Optional[int] = None
    ):
        """
        Configure the variational approximation.

        # Parameters
        - `flowspecs`: The specification for a sequence of flows.
        """
        # Append new NF architecture
        if flowspecs is not None:
            # Initialize NF architecture
            flows = [
                flowspec.construct(self.vari.dim) for flowspec in flowspecs
            ]

            if isinstance(self.vari, Standard):
                # Construct new normalizing flow
                self.vari = NormalizingFlow(
                    base = self.vari,
                    flows = flows,
                    _static = self.vari._static,
                    _unflatten = self.vari._unflatten
                )
            elif isinstance(self.vari, NormalizingFlow):
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
        if batch_size is not None:
            self.config["batch_size"] = batch_size


    def fit(
        self,
        max_iters: int = 50_000,
        learning_rate: Optional[float] = None,
        tolerance: Optional[float] = None,
        grad_draws: Optional[int] = None,
        batch_size: Optional[int] = None,
        key: PRNGKeyArray = jr.key(0),
    ):
        # Include settings
        if learning_rate is not None:
            self.config["learning_rate"] = learning_rate
        if tolerance is not None:
            self.config["tolerance"] = tolerance
        if grad_draws is not None:
            self.config["grad_draws"] = grad_draws
        if batch_size is not None:
            self.config["batch_size"] = batch_size

        if isinstance(self.vari, Standard):
            # Construct default sequence of optimization
            print("TODO")
        else:
            # Optimize variational approximation with user-specified flows
            self.vari = self.vari.fit(
                max_iters,
                self.config["learning_rate"],
                self.config["tolerance"],
                self.config["grad_draws"],
                self.config["batch_size"],
                key
            )

    def sample(
        self,
        node: str,
        n_draws: int,
        batch_size: Optional[int] = None,
        key: PRNGKeyArray = jr.key(0)
    ) -> Array:
        """
        Sample a node from the posterior distribution.

        # Parameters
        - `node`: The name of the node.
        - `n_draws`: The number of draws from the posterior.
        - `batch_size`: The number of draws for the full model ever initialized in memory at once.
        - `key`: The PRNG key.
        """
        if batch_size is None:
            batch_size = n_draws

        # Split keys
        keys = jr.split(key, n_draws // batch_size)

        @partial(jax.vmap, in_axes = 0)
        def reconstruct_and_subset(draw: Array):
            model = self.vari.reconstruct_model(draw).constrain()[0]

            return getattr(model, node).obj

        def batched_sample(carry: None, key: PRNGKeyArray):
            # Sample draws
            draws = self.vari.sample(batch_size, key)

            return None, reconstruct_and_subset(draws)

        posterior_draws = scan(
            batched_sample,
            init=None,
            xs=keys,
            length=n_draws // batch_size
        )[1].squeeze()

        return posterior_draws


    def predictive(
        self,
        func: Callable[[M, PRNGKeyArray], Node[Array] | Array],
        n_draws: int,
        batch_size: Optional[int] = None,
        key: PRNGKeyArray = jr.key(0)
    ) -> Array:
        """

        """
        if batch_size is None:
            batch_size = n_draws

        # Split keys
        keys = jr.split(key, n_draws // batch_size)

        @partial(jax.vmap, in_axes = (0, 0))
        def reconstruct_and_predict(draw: Array, key: PRNGKeyArray) -> Array:
            model = self.vari.reconstruct_model(draw).constrain()[0]

            # Compute predictive
            obj = func(model, key)

            # Coerce from Node if needed
            if isinstance(obj, Node):
                obj: Array = obj.obj # type: ignore

            return obj

        def batched_sample(carry: None, key: PRNGKeyArray) -> Tuple[None, Array]:
            # Sample draws
            draws = self.vari.sample(batch_size, key)

            # Generate additional keys for each draw
            keys = jr.split(key, batch_size)

            return None, reconstruct_and_predict(draws, keys)

        posterior_draws: Array = scan(
            batched_sample,
            init=None,
            xs=keys,
            length=n_draws // batch_size
        )[1].squeeze()

        return posterior_draws
