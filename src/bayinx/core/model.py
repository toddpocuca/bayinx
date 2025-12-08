from abc import abstractmethod
from dataclasses import field, fields
from typing import (
    Dict,
    Optional,
    Self,
    Tuple,
    Type,
    get_origin,
    get_type_hints,
)

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree, Scalar

from bayinx.constraints import Identity, Interval, Lower, Upper
from bayinx.core.context import _model_context, model_context
from bayinx.core.node import Node
from bayinx.core.types import HasConstraint
from bayinx.core.utils import _extract_shape_params, _resolve_shape_spec
from bayinx.nodes import Continuous, Observed, Stochastic


def define(
    shape: Optional[int | str | Tuple[int | str, ...]] = None,
    init: Optional[PyTree] = None,
    lower: Optional[float] = None,
    upper: Optional[float] = None
):
    """
    Define a stochastic node.

    # Parameters
    - `shape`: Specify the shape of the node.
    - `init`: Specify the node in the definition.
    - `lower`: Enforce a lower bound on a stochastic node.
    - `upper`: Enforce an upper bound on a stochastic bode.
    """
    metadata: Dict = {}

    if shape is not None:
        metadata["shape"] = shape
    if init is not None:
        metadata["init"] = init

    match (lower, upper):
        case (float() | int(), None):
            metadata["constraint"] = Lower(lower) # type: ignore
        case (None, float() | int()):
            metadata["constraint"] = Upper(upper) # type: ignore
        case (float() | int(), float() | int()):
            metadata["constraint"] = Interval(lower, upper) # type: ignore
        case (None, None):
            metadata["constraint"] = Identity()
        case (_):
            raise TypeError("TODO.")

    return field(metadata=metadata)


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

        # Grab initialized parameters
        init_params: set[str] = {f.name for f in fields(cls) if f.name in kwargs.keys()}

        # Grab shape parameters from model definition
        shape_params: set[str] = set()
        for node_defn in fields(cls):
            if (shape_spec := node_defn.metadata.get("shape")) is not None:
                shape_params = shape_params | _extract_shape_params(shape_spec)

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

        # Grab node types
        node_types: dict[str, Type[Node]] = {k: get_origin(v) for k, v in get_type_hints(cls).items()}


        # Auto-initialize parameters based on field metadata and type annotations
        for node_defn in fields(cls):
            # Grab node type
            node_type = node_types[node_defn.name]

            # Grab shape information if available
            shape_spec: str | None = node_defn.metadata.get("shape")
            shape = _resolve_shape_spec(shape_spec, shape_values)

            # Construct object
            if node_defn.name in init_params: # Initialized in model construction
                obj = kwargs[node_defn.name]
            elif "init" in node_defn.metadata: # Initialized in model definition
                obj = node_defn.metadata["init"]
            elif issubclass(node_type, Stochastic) and shape is not None: # Shape for stochastic node defined in model definition
                obj = jnp.zeros(shape) # TODO: will change later for discrete objects
            else:
                raise ValueError(f"Node '{node_defn.name}' not initialized or defined.")

            # Check shape
            if shape is not None and jnp.shape(obj) != shape:
                raise ValueError(f"Expected shape {shape} for '{node_defn.name}' but got {jnp.shape(obj)}.")

            if issubclass(node_type, Stochastic):
                if node_type == Continuous:
                    setattr(
                        self,
                        node_defn.name,
                        Continuous(obj, node_defn.metadata["constraint"]),
                    )
                else:
                    TypeError(f"'{node_type.__name__}' is not implemented yet")
            elif issubclass(node_type, Observed):
                # Construct node
                node = Observed(obj)

                # Check constraints (if available)
                if not node_defn.metadata['constraint'].check(node.obj, node._filter_spec):
                    raise ValueError(f"Observed values for '{node_defn.name}' do not satisfy constraint '{node_defn.metadata['constraint']}'.")

                setattr(self, node_defn.name, Observed(obj))
            else:
                raise TypeError(f"{node_defn.name} node is neither Stochastic nor Observed but {node_type.__name__}.")


    @property
    def filter_spec(self) -> Self:
        """
        Generates a filter specification to subset stochastic elements of the model.
        """
        # Generate empty specification
        filter_spec: Self = jt.map(lambda _: False, self)

        for f in fields(self): # type: ignore
            # Extract attribute
            node: Node = getattr(self, f.name)

            # Check if attribute is stochastic
            if isinstance(node, Stochastic):
                # Update model's filter specification at node
                filter_spec: Self = eqx.tree_at(
                    lambda model: getattr(model, f.name),
                    filter_spec,
                    replace=node.filter_spec
                )

        return filter_spec

    def filter_for(self, node_type: Type[Stochastic]) -> Self:
        """
        Generates a filter specification to subset stochastic elements of a certain type of the model.
        """
        # Generate empty specification
        filter_spec: Self = jt.map(lambda _: False, self)

        for f in fields(self): # type: ignore
            # Extract node
            node: Node = getattr(self, f.name)

            if isinstance(node, node_type):
                # Update model's filter specification for node
                filter_spec: Self = eqx.tree_at(
                    lambda model: getattr(model, f.name),
                    filter_spec,
                    replace=node.filter_spec,
                )

        return filter_spec


    def constrain(self, jacobian: bool = True) -> Tuple[Self, Scalar]:
        """
        Constrain nodes to the appropriate domain.

        # Returns
        A tuple containing the constrained `Model` object and the log-Jacobian adjustment.
        """
        model: Self = self
        target: Scalar = jnp.array(0.0)

        for f in fields(self): # type: ignore
            # Extract attribute
            node = getattr(self, f.name)

            # Check if node has a constraint
            if isinstance(node, HasConstraint):
                # Apply constraint
                obj, log_jac = node._constraint.constrain(node.obj, node._filter_spec)

                # Update values with constrained counterpart
                model = eqx.tree_at(
                    where=lambda model: getattr(model, f.name).obj,
                    pytree=model,
                    replace=obj,
                )

                # Adjust posterior density
                if jacobian:
                    target += log_jac

        return model, target

    @abstractmethod
    def model(self, target: Scalar) -> Scalar:
        pass

    @eqx.filter_jit
    def __call__(self) -> Scalar:
        with model_context():
            # Constrain the model and accumulate Jacobian adjustments
            self, target = self.constrain()

            # Accumulate manual increments
            target += self.model(jnp.array(0.0))

            # Accumulate implicit increments
            target += _model_context.target.value

            # Return the accumulated target
            return target
