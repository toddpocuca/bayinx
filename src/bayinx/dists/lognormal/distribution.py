from jaxtyping import ArrayLike

from bayinx.core.distribution import Distribution, Parameterization

from .pars import LocScaleLogNormal


class LogNormal(Distribution):
    """
    A Log-Normal distribution.

    Parameters:
        loc: Parameterizes a Log-Normal distribution by its location (mean of log X).
        scale: Parameterizes a Log-Normal distribution by its scale (standard-deviation of log X).
    """

    par: Parameterization

    def __init__(
        self,
        loc: None | ArrayLike = None,
        scale: None | ArrayLike = None
    ):
        if loc is not None and scale is not None:
            self.par = LocScaleLogNormal(loc, scale)
        else:
            raise TypeError(f"Expected loc: {loc}, and at least one of scale: {scale} to be not None.")
