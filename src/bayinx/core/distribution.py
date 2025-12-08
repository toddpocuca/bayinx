from typing import Protocol, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import ArrayLike, PRNGKeyArray, Scalar

from bayinx.core.node import Node


class Parameterization(Protocol):
    """
    A protocol used for defining distribution parameterizations.
    """

    def logprob(self, x: ArrayLike) -> Scalar: ...

    def sample(self, shape: Tuple[int, ...], key: PRNGKeyArray): ...


class Distribution(Protocol):
    """
    A protocol used for defining distributions.
    """
    par: Parameterization

    def logprob(self, node: Node) -> Scalar:
        obj, par = (
            node.obj,
            self.par,
        )

        # Filter out irrelevant values
        obj, _ = eqx.partition(obj, node._filter_spec)

        # Compute log probabilities across leaves
        eval_obj = jt.map(lambda x: par.logprob(x).sum(), obj)

        # Compute total sum
        total = jt.reduce_associative(lambda x,y: x + y, eval_obj, identity=0.0)

        return jnp.asarray(total)

    def sample(self, shape: int | Tuple[int, ...], key: PRNGKeyArray = jr.key(0)):
        # Coerce to tuple
        if isinstance(shape, int):
            shape = (shape, )

        return self.par.sample(shape, key)

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
