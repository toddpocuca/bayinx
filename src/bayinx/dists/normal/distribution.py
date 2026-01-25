from typing import Optional

from jaxtyping import Array, ArrayLike, Real

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.node import Node

from .pars import LocPrecisionNormal, LocScaleNormal, LocVarNormal


class Normal(Distribution):
    """
    A Normal distribution.

    Parameters:
        loc: Parameterizes a Normal distribution by its location.
        scale: Parameterizes a Normal distribution by its scale (standard-deviation).
        var: Parameterizes a Normal distribution by its variance.
        prec: Parameterizes a Normal distribution by its precision.
    """

    par: Parameterization


    def __init__(
        self,
        loc: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        scale: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        var: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        prec: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None
    ):
        if loc is not None and scale is not None:
            self.par = LocScaleNormal(loc, scale)
        elif loc is not None and var is not None:
            self.par = LocVarNormal(loc, var)
        elif loc is not None and prec is not None:
            self.par = LocPrecisionNormal(loc, prec)
        else:
            raise TypeError(f"Expected loc: {loc}, and at least one of scale: {scale}, var: {var}, prec: {prec} to be not None.")
