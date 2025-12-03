from typing import Protocol, Tuple

import jax.random as jr
from jaxtyping import PRNGKeyArray, Scalar

from bayinx.core.node import Node


class Distribution(Protocol):
    """
    A protocol used for defining the structure of distributions.
    """

    def logprob(self, node: Node) -> Scalar: ...

    def sample(self, shape: int | Tuple[int, ...], key: PRNGKeyArray = jr.key(0)): ...

    def __rlshift__(self, node: Node):
        """
        Implicitly accumulate the log probability into the current model context.
        """
        from bayinx.core.context import _model_context

        # Evaluate log posterior
        log_prob = self.logprob(node)

        # Accumulate log probability into context
        if hasattr(_model_context, "target"):
            _model_context.target += log_prob
        else:
            raise RuntimeError(
                "Model context doesn't exist. Make sure you're calling "
                + "this within the 'model' method."
            )
