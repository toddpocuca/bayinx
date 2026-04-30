from abc import abstractmethod
from dataclasses import field, fields
from typing import Literal, Self, TypedDict

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree, Scalar

from bayinx.constraints import Identity, Interval, LogSimplex, Lower, Simplex, Upper
from bayinx.core.constraint import Constraint
from bayinx.core.context import Target, _model_context, model_context
from bayinx.core.utils import _extract_shape_params, _resolve_shape_spec


class NodeMetadata(TypedDict):
    type: Literal["observed", "stochastic"]
    shape: None | int | str | tuple[int | str, ...]
    init: None | PyTree
    constraint: Constraint

def define(
    type: Literal["observed", "stochastic"],
    shape: None | int | str | tuple[int | str, ...] = None,
    init: None | PyTree = None,
    lower: None | float = None,
    upper: None | float = None,
    simplex: None | float | bool = None,
    logsimplex: None | float | bool = None
):
    match (lower, upper, simplex, logsimplex):
        case (float() | int(), None, None, None):
            constraint = Lower(lower)
        case (None, float() | int(), None, None):
            constraint = Upper(upper)
        case (float() | int(), float() | int(), None, None):
            constraint = Interval(lower, upper)
        case (None, None, float() | bool(), None):
            constraint = Simplex(float(simplex)) # type: ignore
        case (None, None, None, float() | bool()):
            constraint = LogSimplex(float(logsimplex)) # type: ignore
        case (None, None, None, None):
            constraint = Identity()
        case (_):
            raise TypeError("Unclear definition.")

    metadata: NodeMetadata = {
        "type": type,
        "shape": shape,
        "init": init,
        "constraint": constraint # type: ignore
    }

    return field(metadata=metadata)

def observed(
    shape: None | int | str | tuple[int | str, ...] = None,
    init: None | PyTree = None,
    lower: None | float = None,
    upper: None | float = None,
    simplex: None | float | bool = None,
    logsimplex: None | float | bool = None
):
    """
    A field specifier marking an observed node (attribute) of a model (class inheriting from `Model`).

    Parameters:
        shape: Specify the shape of the node.
        init: Specify the node's value in the definition.
        lower: Enforce a lower bound.
        upper: Enforce an upper bound.
        simplex: Enforce a simplex constraint where the values sum to `simplex`.
        logsimplex: Enforce a log-simplex constraint where the exponentiated values sum to `logsimplex`.
    """

    return define("observed", shape, init, lower, upper, simplex, logsimplex)

def stochastic(
    shape: None | int | str | tuple[int | str, ...] = None,
    init: None | PyTree = None,
    lower: None | float = None,
    upper: None | float = None,
    simplex: None | float | bool = None,
    logsimplex: None | float | bool = None
):
    """
    A field specifier marking a stochastic node (attribute) of a model (class inheriting from `Model`).

    Parameters:
        shape: Specify the shape of the node.
        init: Specify the node's structure in the definition.
        lower: Enforce a lower bound.
        upper: Enforce an upper bound.
        simplex: Enforce a simplex constraint where the values sum to `simplex`.
        logsimplex: Enforce a log-simplex constraint where the exponentiated values sum to `logsimplex`.
    """

    return define("stochastic", shape, init, lower, upper, simplex, logsimplex)


class Model(eqx.Module):
    """
    A base class used to define probabilistic models.
    """

    def __init_subclass__(cls, **kwargs):
        # Consume 'init' argument before passing it up to Equinox
        kwargs.pop('init', None)
        super().__init_subclass__(**kwargs)

    def __init__(self, **kwargs):
        cls = self.__class__
        all_fields = {f.name for f in fields(cls)}

        # Grab initialized parameters
        init_params: set[str] = {f.name for f in fields(cls) if f.name in kwargs.keys()} # TODO

        # Grab shape parameters from model definition
        shape_params: set[str] = set()
        for node_defn in fields(cls):
            if (shape_spec := node_defn.metadata.get("shape")) is not None:
                shape_params = shape_params | _extract_shape_params(shape_spec)

        allowed_keys = all_fields | shape_params
        extra_keys = set(kwargs.keys()) - allowed_keys
        if extra_keys:
            raise TypeError(f"Model received unexpected keyword arguments: {extra_keys}")

        # Check all shape parameters are passed as arguments
        if not shape_params.issubset(kwargs.keys()):
            missing_params = shape_params - kwargs.keys()
            raise TypeError(
                f"Following shape parameters were not specified during model initialization: '{", ".join(missing_params)}'."
            )


        # Define all initialized dimensions
        shape_values: dict = {
            shape_param: kwargs[shape_param]
            for shape_param in shape_params
        }

        # Auto-initialize parameters based on field metadata and type annotations
        for node_defn in fields(cls):
            # Grab node name and type
            name = node_defn.name
            node_type: Literal["observed", "stochastic"] = node_defn.metadata["type"]

            # Grab shape information if available
            shape_spec: str | None = node_defn.metadata.get("shape")
            shape = _resolve_shape_spec(shape_spec, shape_values)

            # Construct object
            if node_defn.name in init_params: # Initialized in model construction
                obj = kwargs[name]
            elif node_defn.metadata["init"] is not None: # Initialized in model definition
                obj = node_defn.metadata["init"]
            elif node_type == "stochastic" and shape is not None: # Shape for stochastic node defined in model definition
                # Decrement shape for constrained objects
                if isinstance(node_defn.metadata["constraint"], Simplex | LogSimplex):
                    shape = shape[:-1] + (shape[-1] - 1,)

                obj = jnp.zeros(shape)
            else:
                raise ValueError(f"Node '{node_defn.name}' not initialized or defined.")

            # Check shape
            if shape is not None and jnp.shape(obj) != shape:
                raise ValueError(f"Expected shape {shape} for '{node_defn.name}' but got {jnp.shape(obj)}.")

            # Set attribute with constructed object
            setattr(
                self,
                name,
                obj
            )

    @property
    def filter_spec(self) -> Self:
        """
        Generates a filter specification to subset stochastic elements of the model.
        """
        # Generate empty specification
        filter_spec: Self = jt.map(lambda _: False, self)

        for f in fields(self):
            # Extract attribute and type
            node = getattr(self, f.name)
            node_type: Literal["observed", "stochastic"] = f.metadata['type']

            # Check if attribute is stochastic
            if node_type == "stochastic":
                # Update model's filter specification at the node
                filter_spec: Self = eqx.tree_at(
                    lambda model: getattr(model, f.name),
                    filter_spec,
                    replace=jt.map(eqx.is_inexact_array_like, node)
                )

        return filter_spec

    def constrain(self, jacobian: bool = True) -> tuple[Self, Scalar]:
        """
        Constrain nodes to the appropriate domain.

        # Returns
        A tuple containing the constrained `Model` object and the log-Jacobian adjustment.
        """
        model: Self = self
        total: Scalar = jnp.array(0.0)

        for f in fields(self):
            # Extract attribute, type, and constraint
            obj = getattr(self, f.name)
            node_type: Literal["observed", "stochastic"] = f.metadata['type']
            constraint: Constraint = f.metadata["constraint"]

            # Check if node is stochastic
            if node_type == "stochastic":
                # Construct filter specification
                filter_spec = jt.map(eqx.is_inexact_array, obj)

                # Apply constraint
                obj, log_jac = constraint.constrain(obj, filter_spec)

                # Update values with constrained counterpart
                model = eqx.tree_at(
                    where=lambda model: getattr(model, f.name),
                    pytree=model,
                    replace=obj,
                )

                # Adjust posterior density
                if jacobian:
                    total += log_jac

        return model, total

    @abstractmethod
    def model(self, target: Target):
        """
        The logic of the model accumulating the (unnormalized) posterior density into `target`.
        """
        pass

    @eqx.filter_jit
    def __call__(self) -> Scalar:
        with model_context(): # Initialize model context
            target = _model_context.target

            # Constrain the model and accumulate Jacobian adjustments
            self, log_jac = self.constrain()
            target += log_jac

            # Accumulate model log probabilities
            self.model(target)

            # Return the target density
            return target.value
