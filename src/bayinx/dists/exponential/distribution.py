from typing import Optional

from jaxtyping import Array, ArrayLike, Real

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.node import Node

from .pars import RateExponential, ScaleExponential


class Exponential(Distribution):
    """
    An Exponential distribution.

    Parameters:
        rate: Parameterizes an Exponential distribution by its rate.
        scale: Parameterizes an Exponential distribution by its scale.
    """

    par: Parameterization


    def __init__(
        self,
        rate: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        scale: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None
    ):
        if rate is not None:
            self.par = RateExponential(rate)
        elif scale is not None:
            self.par = ScaleExponential(scale)
        else:
            raise TypeError(f"Expected rate: {rate} or scale: {scale} to be not None.")
