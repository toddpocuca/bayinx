from typing import Optional

from jaxtyping import Array, ArrayLike, Real

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.node import Node

from .pars import ProbFailureBinomial, ProbSuccessBinomial


class Binomial(Distribution):
    """
    A Binomial distribution.
    """

    par: Parameterization


    def __init__(
        self,
        n: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        p: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        q: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None
    ):
        if n is not None and p is not None:
            self.par = ProbSuccessBinomial(n, p)
        elif n is not None and q is not None:
            self.par = ProbFailureBinomial(n, q)
        else:
            raise TypeError(f"Expected n: {n} and at least one of p: {p}, q: {q} to be not None.")
