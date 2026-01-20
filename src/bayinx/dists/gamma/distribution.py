from typing import Optional

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.types import ArrayObject
from bayinx.dists.gamma.pars import MeanShapeGamma, RateShapeGamma, ScaleShapeGamma


class Gamma(Distribution):
    """
    A Gamma distribution.

    Parameters:
        rate: Parameterize the distribution by its rate.
        shape: Parameterize the distribution by its shape.
        scale: Parameterize the distribution by its scale.
        mean: Parameterize the distribution by its mean.
    """

    par: Parameterization


    def __init__(
        self,
        rate: Optional[ArrayObject] = None,
        shape: Optional[ArrayObject] = None,
        scale: Optional[ArrayObject] = None,
        mean: Optional[ArrayObject] = None
    ):
        if rate is not None and shape is not None:
            self.par = RateShapeGamma(rate, shape)
        elif scale is not None and shape is not None:
            self.par = ScaleShapeGamma(scale, shape)
        elif mean is not None and shape is not None:
            self.par = MeanShapeGamma(mean, shape)
        else:
            raise TypeError("Must choose one parameterization out of 'rate' & 'shape', 'scale' & 'shape' and 'mean' & 'shape'.")
