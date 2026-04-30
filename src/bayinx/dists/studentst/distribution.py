
from jaxtyping import ArrayLike

from bayinx.core.distribution import Distribution, Parameterization

from .pars import LocScaleStudentsT


class StudentsT(Distribution):
    """
    A Student's T distribution.

    Parameters:
        df: Parameterizes a Student's T distribution by its degrees of freedom.
        loc: Parameterizes a Student's T distribution by its location.
        scale: Parameterizes a Student's T distribution by its scale.
    """

    par: Parameterization


    def __init__(
        self,
        df: ArrayLike,
        loc: ArrayLike,
        scale: ArrayLike
    ):
        self.par = LocScaleStudentsT(loc, scale, df)
