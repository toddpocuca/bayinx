import jax.numpy as jnp
import jax.tree as jt

from bayinx.core.node import Node
from bayinx.core.utils import _extract_obj


def exp(node: Node) -> Node:
    """
    Apply the exponential transformation (jnp.exp) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply exponential
    new_obj = jt.map(lambda x: jnp.exp(x), obj)

    return type(node)(new_obj, filter_spec)


def log(node: Node) -> Node:
    """
    Apply the natural logarithm transformation (jnp.log) to a node.
    Handles input value restrictions (must be positive).
    """
    obj, filter_spec = _extract_obj(node)

    # Apply logarithm
    new_obj = jt.map(lambda x: jnp.log(x), obj)

    return type(node)(new_obj, filter_spec)


def sin(node: Node) -> Node:
    """
    Apply the sine transformation (jnp.sin) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply sine
    new_obj = jt.map(lambda x: jnp.sin(x), obj)

    return type(node)(new_obj, filter_spec)


def cos(node: Node) -> Node:
    """
    Apply the cosine transformation (jnp.cos) to a node.
    """
    obj, filter_spec = _extract_obj(node)

    # Apply cosine
    new_obj = jt.map(lambda x: jnp.cos(x), obj)

    return type(node)(new_obj, filter_spec)


def tanh(node: Node) -> Node:
    """
    Apply the hyperbolic tangent transformation (jnp.tanh) to a node.
    """
    obj, filter_spec = _extract_obj(node)


    # Apply tanh
    new_obj = jt.map(lambda x: jnp.tanh(x), obj)

    return type(node)(new_obj, filter_spec)


def sigmoid(node: Node) -> Node:
    """
    Apply the sigmoid (logistic) transformation to a node.
    Sigmoid formula: 1 / (1 + exp(-x))
    """
    obj, filter_spec = _extract_obj(node)

    # Apply sigmoid
    new_obj = jt.map(lambda x: 1.0 / (1.0 + jnp.exp(-x)), obj)

    return type(node)(new_obj, filter_spec)
