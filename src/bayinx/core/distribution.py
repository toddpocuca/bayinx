from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, Scalar

from bayinx.core.node import Node
from bayinx.core.utils import _extract_obj


class Parameterization(Protocol):
    """
    A protocol used for defining distribution parameterizations.
    """

    def logprob(self, x: ArrayLike) -> Array: ...

    def sample(self, shape: tuple[int, ...], key: PRNGKeyArray) -> Array: ...


class Distribution(Protocol):
    """
    A protocol used for defining distributions.
    """
    par: Parameterization

    def eval[T: PyTree](self, node: Node[T] | T) -> Scalar:
        """
        Evaluate log-probability accumulation.
        """
        # Evaluate log-probability across object
        obj = self.logprob(node)

        # Compute log probabilities across leaves
        obj = jt.map(jnp.sum, obj)

        # Compute total sum
        total = jt.reduce_associative(lambda x,y: x + y, obj, identity=0.0)

        return jnp.asarray(total)

    def logprob[T: PyTree](self, node: Node[T] | T) -> T:
        """
        Compute log-probability across a PyTree.
        """
        obj, filter_spec = _extract_obj(node)
        par = self.par

        # Filter out irrelevant values
        obj, _ = eqx.partition(obj, filter_spec)

        # Compute log probabilities across leaves
        obj = jt.map(lambda x: par.logprob(x), obj)

        return obj

    def sample(self, shape: int | tuple[int, ...], key: PRNGKeyArray = jr.key(0)):
        # Coerce to tuple
        if isinstance(shape, int):
            shape = (shape, )

        return self.par.sample(shape, key)

    def __rlshift__[T: PyTree](self, node: Node[T] | T):
        """
        Implicitly accumulate the log probability into the current model context.
        """
        from bayinx.core.context import _model_context

        # Evaluate log posterior
        log_prob = self.eval(node)

        # Accumulate log probability into context
        if hasattr(_model_context, "target"):
            _model_context.target += log_prob
        else:
            raise RuntimeError(
                "Model context target doesn't exist. "
                "Make sure you're calling this within the 'model' method."
            )
