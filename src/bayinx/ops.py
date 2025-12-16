import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import ArrayLike, PyTree

from bayinx.core.node import Node
from bayinx.core.utils import _extract_obj

# Public
__all__ = ["exp", "log", "sin", "cos", "tanh", "sigmoid"]

def exp(node: Node) -> Node:
    """
    Apply the exponential transformation (jnp.exp) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply exponential
    new_obj = jt.map(lambda x: jnp.exp(x), obj)

    return Node(new_obj, filter_spec)


def log(node: Node[PyTree[ArrayLike]] | PyTree[ArrayLike]) -> Node[PyTree[ArrayLike]]:
    """
    Apply the natural logarithm transformation (jnp.log) to an object.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply logarithm
    new_obj = jt.map(lambda x: jnp.log(x), obj)

    return Node(new_obj, filter_spec)


def sin(node: Node[PyTree[ArrayLike]] | PyTree[ArrayLike]) -> Node[PyTree[ArrayLike]]:
    """
    Apply the sine transformation (jnp.sin) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply sine
    new_obj = jt.map(lambda x: jnp.sin(x), obj)

    return Node(new_obj, filter_spec)


def cos(node: Node[PyTree[ArrayLike]] | PyTree[ArrayLike]) -> Node[PyTree[ArrayLike]]:
    """
    Apply the cosine transformation (jnp.cos) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply cosine
    new_obj = jt.map(lambda x: jnp.cos(x), obj)

    return Node(new_obj, filter_spec)


def tanh(node: Node[PyTree[ArrayLike]] | PyTree[ArrayLike]) -> Node[PyTree[ArrayLike]]:
    """
    Apply the hyperbolic tangent transformation (jnp.tanh) to a node.
    """
    obj, filter_spec = _extract_obj(node)


    # Apply tanh
    new_obj = jt.map(lambda x: jnp.tanh(x), obj)

    return Node(new_obj, filter_spec)


def sigmoid(node: Node[PyTree[ArrayLike]] | PyTree[ArrayLike]) -> Node[PyTree[ArrayLike]]:
    """
    Apply the sigmoid transformation to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply sigmoid
    new_obj = jt.map(lambda x: 1.0 / (1.0 + jnp.exp(-x)), obj)

    return Node(new_obj, filter_spec)
