
from jaxtyping import ArrayLike

from bayinx.core.distribution import Distribution, Parameterization

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
        rate: None | ArrayLike = None,
        scale: None | ArrayLike = None
    ):
        if rate is not None:
            self.par = RateExponential(rate)
        elif scale is not None:
            self.par = ScaleExponential(scale)
        else:
            raise TypeError(f"Expected rate: {rate} or scale: {scale} to be not None.")
