from typing import Optional

from jaxtyping import Array, ArrayLike, Real

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.node import Node

from .pars import ProbFailureBernoulli, ProbSuccessBernoulli


class Bernoulli(Distribution):
    """
    A Bernoulli distribution.
    """

    par: Parameterization


    def __init__(
        self,
        p: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        q: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None
    ):
        if p is not None:
            self.par = ProbSuccessBernoulli(p)
        elif q is not None:
            self.par = ProbFailureBernoulli(q)
        else:
            raise TypeError(f"Expected p: {p} or q: {q} to be not None.")
