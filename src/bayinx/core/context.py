import threading
from contextlib import contextmanager
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Scalar

# Local storage for model context
_model_context = threading.local()


@dataclass
class Target:
    value: Scalar

    # Define relevant methods to get around lack of explicit mutability in JAX
    def __iadd__(self, other):
        self.value = self.value + other
        return self

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return self.value + other


@contextmanager
def model_context():
    """
    Context manager that sets up the implicit `target` accumulator.

    This context allows the `<<` operator for Node classes to automatically
    accumulate log probabilities without requiring explicit handling of target.
    """
    _model_context.target = Target(jnp.array(0.0))

    try:
        yield _model_context.target
    finally: # Remove old context if present
        if hasattr(_model_context, "target"):
            delattr(_model_context, "target")
