import jax.nn as jnn
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import ArrayLike, PyTree

from bayinx.core.node import Node
from bayinx.core.utils import _extract_obj

# Public
__all__ = ["exp", "log", "sin", "cos", "tanh", "sigmoid"]

def exp[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the exponential transformation (jnp.exp) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply exponential
    new_obj = jt.map(lambda x: jnp.exp(x), obj)

    return Node(new_obj, filter_spec)


def log[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the natural logarithm transformation (jnp.log) to an object.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply logarithm
    new_obj = jt.map(lambda x: jnp.log(x), obj)

    return Node(new_obj, filter_spec)


def sin[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the sine transformation (jnp.sin) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply sine
    new_obj = jt.map(lambda x: jnp.sin(x), obj)

    return Node(new_obj, filter_spec)


def cos[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the cosine transformation (jnp.cos) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply cosine
    new_obj = jt.map(lambda x: jnp.cos(x), obj)

    return Node(new_obj, filter_spec)


def tanh[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the hyperbolic tangent transformation (jnp.tanh) to a node.
    """
    obj, filter_spec = _extract_obj(node)


    # Apply tanh
    new_obj = jt.map(lambda x: jnp.tanh(x), obj)

    return Node(new_obj, filter_spec)


def sigmoid[T: PyTree[ArrayLike]](node: Node[T] | T) -> Node[T]:
    """
    Apply the sigmoid transformation to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply sigmoid
    new_obj = jt.map(jnn.sigmoid, obj)

    return Node(new_obj, filter_spec)

def obj[T: PyTree](node: Node[T]) -> T:
    """
    Extract internal object from a node.
    """
    return node.obj
