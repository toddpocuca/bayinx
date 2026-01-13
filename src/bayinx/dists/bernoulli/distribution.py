from typing import Optional

from jaxtyping import Array, ArrayLike, Real

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.node import Node

from .pars import (
    LogitProbFailureBernoulli,
    LogitProbSuccessBernoulli,
    ProbFailureBernoulli,
    ProbSuccessBernoulli,
)


class Bernoulli(Distribution):
    """
    A Bernoulli distribution.

    Parameters:
        p: Parameterizes a Bernoulli distribution by its probability of success.
        q: Parameterizes a Bernoulli distribution by its probability of failure.
        logit_p: Parameterizes a Bernoulli distribution by its logit probability of success.
        logit_q: Parameterizes a Bernoulli distribution by its logit probability of failure.
    """

    par: Parameterization


    def __init__(
        self,
        p: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        q: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        logit_p: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None,
        logit_q: Optional[Real[ArrayLike, "..."] | Node[Real[Array, "..."]]] = None
    ):
        if p is not None:
            self.par = ProbSuccessBernoulli(p)
        elif q is not None:
            self.par = ProbFailureBernoulli(q)
        elif logit_p is not None:
            self.par = LogitProbSuccessBernoulli(logit_p)
        elif logit_q is not None:
            self.par = LogitProbFailureBernoulli(logit_q)
        else:
            raise TypeError(f"Expected at least one of p: {p}, q: {q}, logit_p: {logit_p}, logit_q: {logit_q} to be not None.")
