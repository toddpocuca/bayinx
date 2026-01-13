
from jaxtyping import Array, ArrayLike, Real

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.node import Node

from .pars import LocScaleStudentT


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
        df: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        loc: Real[ArrayLike, "..."] | Node[Real[Array, "..."]],
        scale: Real[ArrayLike, "..."] | Node[Real[Array, "..."]]
    ):
        self.par = LocScaleStudentT(loc, scale, df)
