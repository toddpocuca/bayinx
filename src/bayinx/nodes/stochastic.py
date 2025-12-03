from abc import abstractmethod
from typing import Optional

from bayinx.core.node import Node
from bayinx.core.types import T


class Stochastic(Node[T]):
    """
    A container for stochastic (unobserved) nodes of a probabilistic model.

    Subclasses can be constructed with defined filter specifications (implement the `filter_spec` property).

    # Attributes
    - `obj`: An internal realization of the node.
    - `_filter_spec`: An internal filter specification for `obj`.
    """

    @abstractmethod
    def __init__(
        self,
        obj: T,
        filter_spec: Optional[T],
    ):
        pass
