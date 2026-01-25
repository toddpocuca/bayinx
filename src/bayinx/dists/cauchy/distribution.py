from typing import Optional

from jaxtyping import Array, ArrayLike, Real

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.node import Node

from .pars import LocationScaleCauchy


class Cauchy(Distribution):
    """
    A Cauchy distribution.

    Parameters:
        loc: Parameterizes a Cauchy distribution by its location.
        scale: Parameterizes a Cauchy distribution by its scale.
    """

    par: Parameterization


    def __init__(
        self,
        loc: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        scale: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None
    ):
        if scale is not None:
            self.par = LocationScaleCauchy(loc, scale)
        else:
            raise TypeError(f"Expected rate: {scale} to be not None.")
