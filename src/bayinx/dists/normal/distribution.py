
from jaxtyping import ArrayLike

from bayinx.core.distribution import Distribution, Parameterization

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
        loc: None | ArrayLike = None,
        scale: None | ArrayLike = None,
        var: None | ArrayLike = None,
        prec: None | ArrayLike = None
    ):
        if loc is not None and scale is not None:
            self.par = LocScaleNormal(loc, scale)
        elif loc is not None and var is not None:
            self.par = LocVarNormal(loc, var)
        elif loc is not None and prec is not None:
            self.par = LocPrecisionNormal(loc, prec)
        else:
            raise TypeError(f"Expected loc: {loc}, and at least one of scale: {scale}, var: {var}, prec: {prec} to be not None.")
