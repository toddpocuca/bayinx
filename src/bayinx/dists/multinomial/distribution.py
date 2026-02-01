from typing import Optional

from bayinx.core.distribution import Distribution, Parameterization
from bayinx.core.types import ArrayObject

from .pars import LogProbsMultinomial, ProbsMultinomial


class Multinomial(Distribution):
    """
    A Multinomial distribution.
    """

    par: Parameterization

    def __init__(
        self,
        n: ArrayObject,
        probs: Optional[ArrayObject] = None,
        logprobs: Optional[ArrayObject] = None
    ):
        if probs is not None:
            self.par = ProbsMultinomial(n, probs)
        elif logprobs is not None:
            self.par = LogProbsMultinomial(n, logprobs)
        else:
            raise TypeError("Expected 'probs' or 'logprobs' to be not None.")
