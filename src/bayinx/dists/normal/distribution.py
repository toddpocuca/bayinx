from typing import Optional

from jaxtyping import Array, ArrayLike, Real

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.node import Node

from .pars import MeanPrecisionNormal, MeanScaleNormal, MeanVarNormal


class Normal(Distribution):
    """
    A Normal distribution.

    Parameters:
        mean: Parameterizes a Normal distribution by its mean.
        scale: Parameterizes a Normal distribution by its scale (standard-deviation).
        var: Parameterizes a Normal distribution by its variance.
        prec: Parameterizes a Normal distribution by its precision.
    """

    par: Parameterization


    def __init__(
        self,
        mean: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        scale: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        var: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        prec: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None
    ):
        if mean is not None and scale is not None:
            self.par = MeanScaleNormal(mean, scale)
        elif mean is not None and var is not None:
            self.par = MeanVarNormal(mean, var)
        elif mean is not None and prec is not None:
            self.par = MeanPrecisionNormal(mean, prec)
        else:
            raise TypeError(f"Expected mean: {mean}, and at least one of scale: {scale}, var: {var}, prec: {prec} to be not None.")
