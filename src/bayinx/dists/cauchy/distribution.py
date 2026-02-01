

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.types import ArrayObject

from .pars import LocationScaleCauchy


class Cauchy(Distribution):
    """
    A Cauchy distribution.
    """

    par: Parameterization


    def __init__(
        self,
        loc: ArrayObject,
        scale: ArrayObject
    ):
        """
        Construct a Cauchy distribution by selecting a parameterization.

        Parameters:
            loc: Parameterizes a Cauchy distribution by its location.
            scale: Parameterizes a Cauchy distribution by its scale.
        """
        self.par = LocationScaleCauchy(loc, scale)
